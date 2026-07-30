from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable

import torch


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    threshold: float = 0.5
    false_negative_rate: float = 0.0
    pr_auc: float | None = None
    roc_auc: float | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["confusion_matrix"] = {
            "true_negative": self.true_negative,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "true_positive": self.true_positive,
        }
        return value


def binary_metrics_from_logits(
    logits: torch.Tensor | Iterable[float],
    labels: torch.Tensor | Iterable[int],
    *,
    loss: float = 0.0,
    threshold: float = 0.5,
) -> BinaryMetrics:
    logits_tensor = torch.as_tensor(logits, dtype=torch.float32).flatten()
    return binary_metrics_from_probabilities(
        torch.sigmoid(logits_tensor),
        labels,
        loss=loss,
        threshold=threshold,
    )


def binary_metrics_from_probabilities(
    probabilities: torch.Tensor | Iterable[float],
    labels: torch.Tensor | Iterable[int],
    *,
    loss: float = 0.0,
    threshold: float = 0.5,
) -> BinaryMetrics:
    probabilities_tensor = torch.as_tensor(
        probabilities,
        dtype=torch.float64,
    ).flatten()
    labels_tensor = torch.as_tensor(labels, dtype=torch.int64).flatten()
    if probabilities_tensor.numel() != labels_tensor.numel():
        raise ValueError("probabilities and labels must contain the same number of items")
    if probabilities_tensor.numel() == 0:
        raise ValueError("metrics require at least one prediction")
    if not bool(torch.isfinite(probabilities_tensor).all()):
        raise ValueError("probabilities contain NaN or infinity")
    if bool(((probabilities_tensor < 0) | (probabilities_tensor > 1)).any()):
        raise ValueError("probabilities must be in [0, 1]")
    if bool(((labels_tensor != 0) & (labels_tensor != 1)).any()):
        raise ValueError("labels must contain only 0 and 1")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    predictions = probabilities_tensor >= threshold
    positives = labels_tensor == 1
    negatives = ~positives
    tp = int((predictions & positives).sum())
    tn = int((~predictions & negatives).sum())
    fp = int((predictions & negatives).sum())
    fn = int((~predictions & positives).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(1, probabilities_tensor.numel())
    false_negative_rate = fn / (tp + fn) if tp + fn else 0.0
    return BinaryMetrics(
        loss=float(loss),
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        threshold=threshold,
        false_negative_rate=false_negative_rate,
        pr_auc=_average_precision(probabilities_tensor, labels_tensor),
        roc_auc=_roc_auc(probabilities_tensor, labels_tensor),
    )


def calibrate_thresholds(
    probabilities: torch.Tensor | Iterable[float],
    labels: torch.Tensor | Iterable[int],
    *,
    target_recall: float = 0.90,
    default_threshold: float = 0.50,
    loss: float = 0.0,
) -> dict[str, Any]:
    """Evaluate default, best-F1, and target-recall operating points."""

    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be in (0, 1]")
    probabilities_tensor = torch.as_tensor(
        probabilities,
        dtype=torch.float64,
    ).flatten()
    labels_tensor = torch.as_tensor(labels, dtype=torch.int64).flatten()
    if int((labels_tensor == 1).sum()) == 0:
        raise ValueError("Threshold calibration requires at least one positive label")
    candidates = sorted(
        {0.0, 1.0, *(float(value) for value in probabilities_tensor.tolist())}
    )
    evaluated = [
        binary_metrics_from_probabilities(
            probabilities_tensor,
            labels_tensor,
            loss=loss,
            threshold=threshold,
        )
        for threshold in candidates
    ]
    best_f1 = max(
        evaluated,
        key=lambda item: (item.f1, item.recall, item.precision, item.threshold),
    )
    recall_qualified = [
        item for item in evaluated if item.recall + 1e-12 >= target_recall
    ]
    if not recall_qualified:
        raise RuntimeError(
            f"No threshold satisfies target recall {target_recall:.4f}"
        )
    target_recall_metrics = max(
        recall_qualified,
        key=lambda item: (item.precision, item.f1, item.threshold),
    )
    default_metrics = binary_metrics_from_probabilities(
        probabilities_tensor,
        labels_tensor,
        loss=loss,
        threshold=default_threshold,
    )
    return {
        "default": default_metrics.to_dict(),
        "best_f1": best_f1.to_dict(),
        "target_recall": {
            "requested_recall": target_recall,
            **target_recall_metrics.to_dict(),
        },
        "selected_strategy": "target_recall",
        "selected_threshold": target_recall_metrics.threshold,
    }


def _average_precision(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
) -> float | None:
    positive_count = int((labels == 1).sum())
    if positive_count == 0:
        return None
    order = torch.argsort(probabilities, descending=True, stable=True)
    sorted_labels = labels[order]
    cumulative_positive = torch.cumsum(sorted_labels, dim=0)
    ranks = torch.arange(
        1,
        sorted_labels.numel() + 1,
        dtype=torch.float64,
    )
    precision_at_rank = cumulative_positive.to(torch.float64) / ranks
    return float(precision_at_rank[sorted_labels == 1].sum() / positive_count)


def _roc_auc(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
) -> float | None:
    positive_count = int((labels == 1).sum())
    negative_count = int((labels == 0).sum())
    if positive_count == 0 or negative_count == 0:
        return None

    # Average ranks for ties, followed by the Mann-Whitney U statistic.
    order = torch.argsort(probabilities, stable=True)
    sorted_probabilities = probabilities[order]
    ranks = torch.empty_like(sorted_probabilities, dtype=torch.float64)
    start = 0
    while start < sorted_probabilities.numel():
        end = start + 1
        while (
            end < sorted_probabilities.numel()
            and math.isclose(
                float(sorted_probabilities[end]),
                float(sorted_probabilities[start]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        ranks[start:end] = average_rank
        start = end
    original_ranks = torch.empty_like(ranks)
    original_ranks[order] = ranks
    positive_rank_sum = float(original_ranks[labels == 1].sum())
    u_statistic = positive_rank_sum - positive_count * (positive_count + 1) / 2
    return u_statistic / (positive_count * negative_count)
