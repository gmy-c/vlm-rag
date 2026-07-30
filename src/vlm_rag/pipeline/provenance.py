from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def fingerprint_files(paths: Iterable[Path], *, root: Path | None = None) -> str:
    digest = hashlib.sha256()
    resolved_root = root.resolve() if root is not None else None
    existing = sorted(
        (path.resolve() for path in paths if path.is_file()),
        key=lambda path: path.as_posix(),
    )
    if not existing:
        raise FileNotFoundError("No files were available for fingerprinting")
    for path in existing:
        label = (
            path.relative_to(resolved_root).as_posix()
            if resolved_root is not None and path.is_relative_to(resolved_root)
            else path.name
        )
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def fingerprint_adapter(adapter_dir: Path) -> str:
    root = adapter_dir.expanduser().resolve()
    candidates = [
        root / "retrieval_config.json",
        root / "retrieval_heads.pt",
        root / "language_lora" / "adapter_config.json",
        root / "language_lora" / "adapter_model.safetensors",
        root / "language_lora" / "adapter_model.bin",
    ]
    return fingerprint_files(candidates, root=root)


def fingerprint_base_model_metadata(model_dir: Path) -> str:
    root = model_dir.expanduser().resolve()
    candidates = [
        root / "config.json",
        root / "preprocessor_config.json",
        root / "processor_config.json",
        root / "tokenizer_config.json",
        root / "model.safetensors.index.json",
    ]
    return fingerprint_files(candidates, root=root)
