from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any, Callable, Sequence

from .schema import SensitivityRecord, load_manifest


def resolve_data_path(data_root: Path, manifest_path: str) -> Path:
    """Resolve a portable manifest path without allowing data-root traversal."""

    relative = PurePosixPath(manifest_path)
    if relative.is_absolute() or ".." in relative.parts or "\\" in manifest_path:
        raise ValueError(f"Unsafe manifest path: {manifest_path!r}")
    root = data_root.expanduser().resolve()
    resolved = root.joinpath(*relative.parts).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Manifest path escapes data root: {manifest_path!r}") from exc
    return resolved


class SensitivityManifestDataset(Sequence[dict[str, Any]]):
    """A lightweight manifest dataset that can optionally decode page images."""

    def __init__(
        self,
        manifest_path: Path,
        data_root: Path,
        *,
        load_images: bool = True,
        transform: Callable[[Any], Any] | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.data_root = data_root
        self.records = load_manifest(manifest_path)
        self.load_images = load_images
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = resolve_data_path(self.data_root, record.image_path)
        ocr_path = resolve_data_path(self.data_root, record.ocr_path)
        item: dict[str, Any] = {
            "page_id": record.page_id,
            "doc_id": record.doc_id,
            "image_path": image_path,
            "ocr_path": ocr_path,
            "label": float(record.is_sensitive),
            "record": record,
        }
        if self.load_images:
            from PIL import Image

            with Image.open(image_path) as image:
                decoded = image.convert("RGB")
            item["image"] = self.transform(decoded) if self.transform else decoded
        return item
