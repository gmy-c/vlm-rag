from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

from packaging.version import InvalidVersion, Version


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


EXPECTED_PACKAGES = {
    "torch": "2.7.1",
    "torchvision": "0.22.1",
    "transformers": "5.14.1",
    "accelerate": "1.14.0",
    "peft": "0.19.1",
    "huggingface_hub": "1.25.1",
    "safetensors": "0.8.0",
    "colpali-engine": "0.3.17",
    "bitsandbytes": "0.50.0",
    "Pillow": "12.2.0",
    "requests": "2.34.2",
    "numpy": "2.4.4",
    "PyYAML": "6.0.3",
    "packaging": "26.2",
    "sentencepiece": "0.2.2",
    "einops": "0.8.2",
    "tqdm": "4.70.0",
}

MINIMUM_PACKAGES = {
    **EXPECTED_PACKAGES,
    "packaging": "24.2",
}

EXPECTED_PAGE_COUNTS = {
    "full_pages": 12_767,
    "positive_pages": 3_538,
    "negative_pages": 9_229,
}


@dataclass(slots=True)
class CheckResult:
    name: str
    status: str
    elapsed_seconds: float
    details: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate data, manifests, checkpoint, dependencies, CUDA, and "
            "both sensitivity/retrieval forward paths."
        )
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "project_validation.json",
    )
    parser.add_argument(
        "--skip-gpu-smoke",
        action="store_true",
        help="Run structural checks only; mark both real model forwards as skipped.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dependency-policy",
        choices=("compatible", "exact"),
        default="compatible",
        help=(
            "compatible accepts newer installed versions; exact reproduces "
            "the locally verified lock versions."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = (
        args.data_root.expanduser().resolve()
        if args.data_root is not None
        else _resolve_data_root()
    )
    model_root = (
        args.model.expanduser().resolve()
        if args.model is not None
        else _resolve_model_root()
    )
    output_path = args.output.expanduser().resolve()
    results: list[CheckResult] = []

    _run_check(
        results,
        "critical_dependencies",
        lambda: _check_dependencies(args.dependency_policy),
    )
    _run_check(results, "cuda_and_bf16", lambda: _check_cuda(args.device))
    _run_check(results, "dataset_directories", lambda: _check_data(data_root))
    _run_check(results, "sensitivity_manifest", lambda: _check_manifest(data_root))
    _run_check(
        results,
        "retrieval_manifest",
        lambda: _check_retrieval_manifest(data_root),
    )
    _run_check(results, "base_checkpoint", lambda: _check_checkpoint(model_root))

    if args.skip_gpu_smoke:
        results.append(CheckResult("sensitivity_forward", "skipped", 0.0, "flag"))
        results.append(CheckResult("retrieval_forward", "skipped", 0.0, "flag"))
    else:
        _run_check(
            results,
            "sensitivity_forward",
            lambda: _run_subprocess(
                [
                    args.python,
                    str(PROJECT_ROOT / "scripts" / "smoke_sensitivity.py"),
                    "--data-root",
                    str(data_root),
                    "--model",
                    str(model_root),
                    "--batch-size",
                    "2",
                    "--samples-per-class",
                    "1",
                    "--forward-only",
                    "--device",
                    args.device,
                    "--output-dir",
                    str(PROJECT_ROOT / "outputs" / "validation_sensitivity"),
                ],
                timeout=180,
            ),
        )
        _run_check(
            results,
            "retrieval_forward",
            lambda: _run_subprocess(
                [
                    args.python,
                    str(PROJECT_ROOT / "scripts" / "smoke_train.py"),
                    "--data-root",
                    str(data_root),
                    "--model",
                    str(model_root),
                    "--batch-size",
                    "1",
                    "--index-pages",
                    "2",
                    "--index-batch-size",
                    "1",
                    "--forward-only",
                    "--device",
                    args.device,
                ],
                timeout=240,
            ),
        )

    failed = [result for result in results if result.status == "failed"]
    payload = {
        "schema_version": 1,
        "project_root": str(PROJECT_ROOT),
        "data_root": str(data_root),
        "model_root": str(model_root),
        "python": args.python,
        "overall_status": "failed" if failed else "passed",
        "checks": [asdict(result) for result in results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("=" * 72)
    for result in results:
        print(
            f"{result.status.upper():7s} {result.name:24s} "
            f"{result.elapsed_seconds:7.2f}s"
        )
    print("=" * 72)
    print(f"overall_status: {payload['overall_status']}")
    print(f"report: {output_path}")
    return 1 if failed else 0


def _run_check(
    results: list[CheckResult],
    name: str,
    function: Callable[[], Any],
) -> None:
    start = time.perf_counter()
    try:
        details = function()
    except Exception as exc:
        result = CheckResult(
            name=name,
            status="failed",
            elapsed_seconds=time.perf_counter() - start,
            details={"error_type": type(exc).__name__, "error": str(exc)},
        )
    else:
        result = CheckResult(
            name=name,
            status="passed",
            elapsed_seconds=time.perf_counter() - start,
            details=details,
        )
    results.append(result)
    message = f"[{result.status.upper()}] {name}: {result.details}"
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe_message = message.encode(encoding, errors="backslashreplace").decode(encoding)
    print(safe_message)


def _check_dependencies(policy: str = "compatible") -> dict[str, str]:
    installed: dict[str, str] = {}
    mismatches: dict[str, dict[str, str]] = {}
    for package, expected in EXPECTED_PACKAGES.items():
        try:
            actual = version(package)
        except PackageNotFoundError as exc:
            raise RuntimeError(f"Required package is missing: {package}") from exc
        installed[package] = actual
        if policy == "exact":
            accepted = actual == expected or actual.startswith(expected + "+")
            requirement = expected
        else:
            minimum = MINIMUM_PACKAGES[package]
            try:
                accepted = Version(actual.split("+", 1)[0]) >= Version(minimum)
            except InvalidVersion:
                accepted = False
            requirement = f">={minimum}"
        if not accepted:
            mismatches[package] = {
                "policy": policy,
                "expected": requirement,
                "actual": actual,
            }
    if mismatches:
        raise RuntimeError(f"Critical dependency version mismatch: {mismatches}")
    return {"policy": policy, "installed": installed}


def _check_cuda(device: str) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    if not device.startswith("cuda"):
        raise ValueError(f"Full validation expects a CUDA device, got {device!r}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device does not support BF16")
    return {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(torch.device(device)),
        "total_memory_gb": torch.cuda.get_device_properties(
            torch.device(device)
        ).total_memory
        / 1024**3,
        "bf16_supported": True,
    }


def _check_data(data_root: Path) -> dict[str, int]:
    required_dirs = (
        "docvqa_extracted",
        "docvqa_images",
        "ocr",
        "desensitized/docvqa_extracted",
        "desensitized/docvqa_images",
        "desensitized/ocr",
    )
    missing = [
        relative
        for relative in required_dirs
        if not (data_root / relative).is_dir()
    ]
    if missing:
        raise FileNotFoundError(f"Missing dataset directories: {missing}")
    required_qa = (
        "train_v1.0_withQT.json",
        "val_v1.0_withQT.json",
        "test_v1.0.json",
    )
    missing_qa = [
        filename
        for filename in required_qa
        if not (data_root / "docvqa_extracted" / filename).is_file()
    ]
    if missing_qa:
        raise FileNotFoundError(f"Missing full DocVQA QA files: {missing_qa}")
    full_images = {
        path.stem for path in (data_root / "docvqa_images").glob("*.png")
    }
    full_ocr = {path.stem for path in (data_root / "ocr").glob("*.json")}
    positive_images = {
        path.stem
        for path in (data_root / "desensitized" / "docvqa_images").glob("*.png")
    }
    positive_ocr = {
        path.stem
        for path in (data_root / "desensitized" / "ocr").glob("*.json")
    }
    if full_images != full_ocr:
        raise RuntimeError("Full image and OCR page IDs are not identical")
    if positive_images != positive_ocr:
        raise RuntimeError("Positive image and OCR page IDs are not identical")
    if not positive_images <= full_images:
        raise RuntimeError("desensitized positives are not a subset of full pages")
    counts = {
        "full_pages": len(full_images),
        "positive_pages": len(positive_images),
        "negative_pages": len(full_images - positive_images),
    }
    if counts != EXPECTED_PAGE_COUNTS:
        raise RuntimeError(
            f"Unexpected dataset counts: expected {EXPECTED_PAGE_COUNTS}, got {counts}"
        )
    return counts


def _check_manifest(data_root: Path) -> dict[str, Any]:
    from vlm_rag.sensitivity.schema import load_manifest

    manifest_dir = data_root / "manifests" / "sensitivity"
    paths = {
        split: manifest_dir / f"{split}.jsonl"
        for split in ("train", "val", "test")
    }
    all_path = manifest_dir / "all.jsonl"
    summary_path = manifest_dir / "summary.json"
    missing = [
        str(path)
        for path in (all_path, summary_path, *paths.values())
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing sensitivity manifest files: {missing}")
    all_records = load_manifest(all_path)
    split_records = {split: load_manifest(path) for split, path in paths.items()}
    all_page_ids = {record.page_id for record in all_records}
    union_page_ids = {
        record.page_id
        for records in split_records.values()
        for record in records
    }
    if len(all_records) != len(all_page_ids):
        raise RuntimeError("all.jsonl contains duplicate page IDs")
    if union_page_ids != all_page_ids:
        raise RuntimeError("train/val/test union does not match all.jsonl")
    if sum(len(records) for records in split_records.values()) != len(all_records):
        raise RuntimeError("train/val/test contain duplicate pages")
    docs = {
        split: {record.doc_id for record in records}
        for split, records in split_records.items()
    }
    overlaps = {
        "train_val": len(docs["train"] & docs["val"]),
        "train_test": len(docs["train"] & docs["test"]),
        "val_test": len(docs["val"] & docs["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Document leakage detected: {overlaps}")
    positives = sum(record.is_sensitive for record in all_records)
    counts = {
        "full_pages": len(all_records),
        "positive_pages": positives,
        "negative_pages": len(all_records) - positives,
    }
    if counts != EXPECTED_PAGE_COUNTS:
        raise RuntimeError(
            f"Manifest counts do not match expected data: "
            f"expected {EXPECTED_PAGE_COUNTS}, got {counts}"
        )
    return {
        "pages": len(all_records),
        "positive": positives,
        "negative": len(all_records) - positives,
        "documents": len({record.doc_id for record in all_records}),
        "document_overlap": overlaps,
    }


def _check_checkpoint(model_root: Path) -> dict[str, Any]:
    config_path = model_root / "config.json"
    index_path = model_root / "model.safetensors.index.json"
    single_path = model_root / "model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError("Base checkpoint must contain config.json")
    if not index_path.is_file() and not single_path.is_file():
        raise FileNotFoundError(
            "Base checkpoint must contain either model.safetensors or "
            "model.safetensors.index.json plus every referenced shard"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Checkpoint weight_map is missing or empty")
        tensor_names = list(weight_map)
        shard_names = sorted(set(str(value) for value in weight_map.values()))
        missing_shards = [
            shard for shard in shard_names if not (model_root / shard).is_file()
        ]
        if missing_shards:
            raise FileNotFoundError(f"Missing checkpoint shards: {missing_shards}")
    else:
        from safetensors import safe_open

        with safe_open(single_path, framework="pt", device="cpu") as handle:
            tensor_names = list(handle.keys())
        shard_names = [single_path.name]
    vision_tensors = sum(
        key.startswith("model.vision_tower.vision_model.")
        for key in tensor_names
    )
    if vision_tensors == 0:
        raise RuntimeError("Checkpoint contains no recognised SigLIP vision weights")
    return {
        "architecture": config.get("architectures"),
        "model_type": config.get("model_type"),
        "weight_tensors": len(tensor_names),
        "vision_tensors": vision_tensors,
        "shards": shard_names,
    }


def _check_retrieval_manifest(data_root: Path) -> dict[str, Any]:
    from vlm_rag.retrieval.schema import load_retrieval_manifest

    manifest_dir = data_root / "manifests" / "retrieval"
    paths = {
        split: manifest_dir / f"{split}.jsonl"
        for split in ("train", "val", "test")
    }
    all_path = manifest_dir / "all.jsonl"
    missing = [
        str(path)
        for path in (all_path, manifest_dir / "summary.json", *paths.values())
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing retrieval manifest files: {missing}")
    all_records = load_retrieval_manifest(all_path)
    split_records = {
        split: load_retrieval_manifest(path)
        for split, path in paths.items()
    }
    query_ids = [record.query_id for record in all_records]
    if len(query_ids) != len(set(query_ids)):
        raise RuntimeError("Retrieval all.jsonl contains duplicate query IDs")
    split_query_ids = [
        record.query_id
        for records in split_records.values()
        for record in records
    ]
    if set(split_query_ids) != set(query_ids):
        raise RuntimeError("Retrieval split union does not match all.jsonl")
    if len(split_query_ids) != len(set(split_query_ids)):
        raise RuntimeError("Retrieval queries occur in more than one split")
    docs = {
        split: {record.doc_id for record in records}
        for split, records in split_records.items()
    }
    overlaps = {
        "train_val": len(docs["train"] & docs["val"]),
        "train_test": len(docs["train"] & docs["test"]),
        "val_test": len(docs["val"] & docs["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Retrieval document leakage: {overlaps}")
    pages = {record.positive_page_id for record in all_records}
    if len(all_records) != 50_000 or len(pages) != 12_767:
        raise RuntimeError(
            "Unexpected retrieval counts: "
            f"queries={len(all_records)}, pages={len(pages)}"
        )
    return {
        "queries": len(all_records),
        "pages": len(pages),
        "documents": len({record.doc_id for record in all_records}),
        "document_overlap": overlaps,
        "splits": {
            split: {
                "queries": len(records),
                "pages": len(
                    {record.positive_page_id for record in records}
                ),
                "documents": len(docs[split]),
            }
            for split, records in split_records.items()
        },
    }


def _run_subprocess(command: list[str], *, timeout: int) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    output_lines = (completed.stdout + "\n" + completed.stderr).splitlines()
    tail = output_lines[-30:]
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: "
            f"{' '.join(command)}\n" + "\n".join(tail)
        )
    return {
        "command": command,
        "returncode": completed.returncode,
        "output_tail": tail,
    }


def _resolve_data_root() -> Path:
    environment_value = os.environ.get("DOCVQA_DATA_ROOT")
    candidates = [
        Path(environment_value) if environment_value else None,
        PROJECT_ROOT / "data",
        PROJECT_ROOT.parent / "data",
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "docvqa_images").is_dir():
            return candidate.expanduser().resolve()
    raise FileNotFoundError(
        "Could not find full DocVQA data. Set DOCVQA_DATA_ROOT or pass --data-root."
    )


def _resolve_model_root() -> Path:
    environment_value = os.environ.get("COLPALI_MODEL_PATH")
    candidates = [
        Path(environment_value) if environment_value else None,
        PROJECT_ROOT / "checkpoint",
        PROJECT_ROOT.parent / "checkpoint",
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "config.json").is_file():
            return candidate.expanduser().resolve()
    raise FileNotFoundError(
        "Could not find local ColPali checkpoint. "
        "Set COLPALI_MODEL_PATH or pass --model."
    )


if __name__ == "__main__":
    raise SystemExit(main())
