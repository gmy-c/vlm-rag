from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from PIL import Image
import torch

from .dataset import resolve_data_path
from .model import SensitivityClassifier
from .schema import iter_manifest


SUPPORTED_IMAGE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
)


@dataclass(frozen=True, slots=True)
class InferenceItem:
    page_id: str
    resolved_path: Path
    input_path: str
    true_label: int | None = None


@dataclass(frozen=True, slots=True)
class PredictionResult:
    page_id: str
    probability: float
    threshold: float
    predicted_label: int
    input_path: str
    true_label: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class InferenceError:
    page_id: str
    input_path: str
    error_type: str
    error: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


InferenceEvent = PredictionResult | InferenceError


def items_from_manifest(
    manifest_path: Path,
    data_root: Path,
) -> Iterator[InferenceItem]:
    for record in iter_manifest(manifest_path):
        yield InferenceItem(
            page_id=record.page_id,
            resolved_path=resolve_data_path(data_root, record.image_path),
            input_path=record.image_path,
            true_label=record.is_sensitive,
        )


def item_from_image(path: Path) -> InferenceItem:
    resolved = path.expanduser().resolve()
    return InferenceItem(
        page_id=resolved.stem,
        resolved_path=resolved,
        input_path=str(resolved),
    )


def items_from_directory(
    directory: Path,
    *,
    recursive: bool = False,
) -> Iterator[InferenceItem]:
    root = directory.expanduser().resolve()
    iterator = root.rglob("*") if recursive else root.glob("*")
    paths = sorted(
        (
            path
            for path in iterator
            if path.is_file() and path.suffix.casefold() in SUPPORTED_IMAGE_SUFFIXES
        ),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )
    for path in paths:
        relative = path.relative_to(root)
        page_id = relative.with_suffix("").as_posix()
        yield InferenceItem(
            page_id=page_id,
            resolved_path=path,
            input_path=str(path),
        )


def run_batched_inference(
    model: SensitivityClassifier,
    items: Iterable[InferenceItem],
    *,
    batch_size: int = 2,
    threshold: float = 0.5,
) -> Iterator[InferenceEvent]:
    """Yield predictions/errors while keeping at most ``batch_size`` images on GPU."""

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    model.eval()
    item_iterator = iter(items)
    while True:
        batch_items = list(islice(item_iterator, batch_size))
        if not batch_items:
            break
        yield from _infer_one_batch(model, batch_items, threshold=threshold)


def _infer_one_batch(
    model: SensitivityClassifier,
    items: Sequence[InferenceItem],
    *,
    threshold: float,
) -> Iterator[InferenceEvent]:
    events: dict[int, InferenceEvent] = {}
    valid_positions: list[int] = []
    valid_items: list[InferenceItem] = []
    images: list[Image.Image] = []
    for position, item in enumerate(items):
        try:
            with Image.open(item.resolved_path) as source:
                source.load()
                image = source.convert("RGB")
        except Exception as exc:  # Pillow raises decoder-specific exception classes.
            events[position] = InferenceError(
                page_id=item.page_id,
                input_path=item.input_path,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            continue
        valid_positions.append(position)
        valid_items.append(item)
        images.append(image)

    if images:
        processed = model.image_processor(images=images, return_tensors="pt")
        with torch.inference_mode():
            logits = model(processed["pixel_values"])
            probabilities = torch.sigmoid(logits.float()).cpu().tolist()
        for position, item, probability in zip(
            valid_positions,
            valid_items,
            probabilities,
        ):
            events[position] = PredictionResult(
                page_id=item.page_id,
                probability=float(probability),
                threshold=float(threshold),
                predicted_label=int(probability >= threshold),
                input_path=item.input_path,
                true_label=item.true_label,
            )
        del processed, logits, probabilities

    for position in range(len(items)):
        yield events[position]
