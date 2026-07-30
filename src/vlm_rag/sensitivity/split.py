from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from typing import Iterable, Mapping


SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True, slots=True)
class DocumentStats:
    doc_id: str
    page_count: int
    positive_count: int

    @property
    def negative_count(self) -> int:
        return self.page_count - self.positive_count


def assign_documents_to_splits(
    page_labels: Iterable[tuple[str, int]],
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, str]:
    """Assign whole documents while approximately stratifying page labels.

    A stable hash creates the initial allocation, then a deterministic local
    search balances pages, positives, negatives, and document counts. The fixed
    seed makes the result reproducible across Python versions and operating
    systems.
    """

    _validate_ratios(ratios)
    grouped: dict[str, list[int]] = defaultdict(list)
    for doc_id, label in page_labels:
        if label not in (0, 1):
            raise ValueError(f"{doc_id}: label must be 0 or 1, got {label!r}")
        grouped[doc_id].append(label)
    if len(grouped) < len(SPLIT_NAMES):
        raise ValueError("At least three documents are required for train/val/test")

    documents = [
        DocumentStats(doc_id, len(labels), sum(labels))
        for doc_id, labels in grouped.items()
    ]
    totals = {
        "pages": sum(doc.page_count for doc in documents),
        "positive": sum(doc.positive_count for doc in documents),
        "negative": sum(doc.negative_count for doc in documents),
        "documents": len(documents),
    }
    targets = {
        split: {
            metric: totals[metric] * ratios[index]
            for metric in totals
        }
        for index, split in enumerate(SPLIT_NAMES)
    }

    current = {
        split: {"pages": 0, "positive": 0, "negative": 0, "documents": 0}
        for split in SPLIT_NAMES
    }
    assignment: dict[str, str] = {}

    # A stable hash supplies an unbiased, cross-platform initial allocation.
    # Starting randomly avoids systematically sending long documents to a
    # particular split. A deterministic local search then improves page and
    # label balance while retaining document-level isolation.
    cumulative = (ratios[0], ratios[0] + ratios[1], 1.0)
    documents_by_id = {document.doc_id: document for document in documents}
    for document in documents:
        unit_value = _stable_unit_interval(document.doc_id, seed)
        if unit_value < cumulative[0]:
            split = "train"
        elif unit_value < cumulative[1]:
            split = "val"
        else:
            split = "test"
        assignment[document.doc_id] = split
        _add_document(current[split], document, direction=1)

    _ensure_non_empty_splits(assignment, documents_by_id, current, targets)

    ordered_documents = sorted(
        documents,
        key=lambda document: (_stable_hash(document.doc_id, seed), document.doc_id),
    )
    for _ in range(10):
        moved = 0
        for document in ordered_documents:
            old_split = assignment[document.doc_id]
            if current[old_split]["documents"] <= 1:
                continue
            best_score = _global_objective(current, targets)
            best_split = old_split
            for candidate_split in SPLIT_NAMES:
                if candidate_split == old_split:
                    continue
                _add_document(current[old_split], document, direction=-1)
                _add_document(current[candidate_split], document, direction=1)
                score = _global_objective(current, targets)
                _add_document(current[candidate_split], document, direction=-1)
                _add_document(current[old_split], document, direction=1)
                if score < best_score - 1e-15:
                    best_score = score
                    best_split = candidate_split
            if best_split != old_split:
                _add_document(current[old_split], document, direction=-1)
                _add_document(current[best_split], document, direction=1)
                assignment[document.doc_id] = best_split
                moved += 1
        if moved == 0:
            break

    return assignment


def split_membership(
    assignments: Mapping[str, str],
) -> dict[str, set[str]]:
    membership = {split: set() for split in SPLIT_NAMES}
    for doc_id, split in assignments.items():
        if split not in membership:
            raise ValueError(f"Unknown split {split!r} for document {doc_id!r}")
        membership[split].add(doc_id)
    return membership


def _validate_ratios(ratios: tuple[float, float, float]) -> None:
    if len(ratios) != len(SPLIT_NAMES):
        raise ValueError("ratios must contain train, val, and test values")
    if any(value <= 0 for value in ratios):
        raise ValueError("all split ratios must be positive")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1, got {sum(ratios):.12f}")


def _stable_hash(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _stable_unit_interval(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest, "big") / float(1 << (8 * len(digest)))


def _add_document(
    bucket: dict[str, int],
    document: DocumentStats,
    direction: int,
) -> None:
    bucket["pages"] += direction * document.page_count
    bucket["positive"] += direction * document.positive_count
    bucket["negative"] += direction * document.negative_count
    bucket["documents"] += direction


def _global_objective(
    current: Mapping[str, Mapping[str, int]],
    targets: Mapping[str, Mapping[str, float]],
) -> float:
    weights = {"pages": 5.0, "positive": 5.0, "negative": 5.0, "documents": 2.0}
    score = 0.0
    for split in SPLIT_NAMES:
        for metric, weight in weights.items():
            denominator = max(float(targets[split][metric]), 1.0)
            error = (current[split][metric] - targets[split][metric]) / denominator
            score += weight * error * error
    return score


def _ensure_non_empty_splits(
    assignment: dict[str, str],
    documents_by_id: Mapping[str, DocumentStats],
    current: dict[str, dict[str, int]],
    targets: Mapping[str, Mapping[str, float]],
) -> None:
    for empty_split in (split for split in SPLIT_NAMES if current[split]["documents"] == 0):
        best_move: tuple[float, str, str] | None = None
        for doc_id, old_split in assignment.items():
            if current[old_split]["documents"] <= 1:
                continue
            document = documents_by_id[doc_id]
            _add_document(current[old_split], document, direction=-1)
            _add_document(current[empty_split], document, direction=1)
            score = _global_objective(current, targets)
            _add_document(current[empty_split], document, direction=-1)
            _add_document(current[old_split], document, direction=1)
            candidate = (score, doc_id, old_split)
            if best_move is None or candidate < best_move:
                best_move = candidate
        if best_move is None:
            raise ValueError("Could not create three non-empty document splits")
        _, doc_id, old_split = best_move
        document = documents_by_id[doc_id]
        _add_document(current[old_split], document, direction=-1)
        _add_document(current[empty_split], document, direction=1)
        assignment[doc_id] = empty_split
