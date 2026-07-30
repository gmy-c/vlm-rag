from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import json
from pathlib import Path
import sys
from typing import TextIO

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vlm_rag.sensitivity.inference import (
    InferenceError,
    PredictionResult,
    item_from_image,
    items_from_directory,
    items_from_manifest,
    run_batched_inference,
)
from vlm_rag.sensitivity.model import load_sensitivity_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify single pages, directories, or sensitivity manifests."
    )
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--image", type=Path)
    sources.add_argument("--input-dir", type=Path)
    sources.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--format",
        choices=("csv", "jsonl", "both"),
        default="both",
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="Return exit code 0 even when corrupt/unreadable images were audited.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    base_model = _resolve_model(args.model)
    threshold, threshold_source = _resolve_threshold(
        args.threshold,
        args.calibration,
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else PROJECT_ROOT / "outputs" / "sensitivity_inference"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.image is not None:
        items = iter([item_from_image(args.image)])
        input_mode = "image"
    elif args.input_dir is not None:
        items = items_from_directory(args.input_dir, recursive=args.recursive)
        input_mode = "directory"
    else:
        if args.data_root is None:
            raise ValueError("--data-root is required with --manifest")
        items = items_from_manifest(
            args.manifest.expanduser().resolve(),
            args.data_root.expanduser().resolve(),
        )
        input_mode = "manifest"

    if torch.device(args.device).type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model, _ = load_sensitivity_checkpoint(
        checkpoint,
        device=args.device,
        checkpoint_path_override=str(base_model),
    )
    baseline_allocated = _cuda_gb(torch.cuda.memory_allocated)

    prediction_count = 0
    sensitive_count = 0
    error_count = 0
    with ExitStack() as stack:
        jsonl_handle: TextIO | None = None
        csv_handle: TextIO | None = None
        csv_writer: csv.DictWriter | None = None
        if args.format in {"jsonl", "both"}:
            jsonl_handle = stack.enter_context(
                (output_dir / "predictions.jsonl").open(
                    "w",
                    encoding="utf-8",
                    newline="\n",
                )
            )
        if args.format in {"csv", "both"}:
            csv_handle = stack.enter_context(
                (output_dir / "predictions.csv").open(
                    "w",
                    encoding="utf-8-sig",
                    newline="",
                )
            )
            csv_writer = csv.DictWriter(
                csv_handle,
                fieldnames=[
                    "page_id",
                    "probability",
                    "threshold",
                    "predicted_label",
                    "input_path",
                    "true_label",
                ],
            )
            csv_writer.writeheader()
        error_handle = stack.enter_context(
            (output_dir / "errors.jsonl").open(
                "w",
                encoding="utf-8",
                newline="\n",
            )
        )

        for event in run_batched_inference(
            model,
            items,
            batch_size=args.batch_size,
            threshold=threshold,
        ):
            if isinstance(event, PredictionResult):
                row = event.to_dict()
                if jsonl_handle is not None:
                    jsonl_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                if csv_writer is not None:
                    csv_writer.writerow(row)
                prediction_count += 1
                sensitive_count += event.predicted_label
            else:
                error_handle.write(
                    json.dumps(event.to_dict(), ensure_ascii=False) + "\n"
                )
                error_count += 1

    summary = {
        "schema_version": 1,
        "input_mode": input_mode,
        "checkpoint": str(checkpoint),
        "base_model": str(base_model),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "batch_size": args.batch_size,
        "predictions": prediction_count,
        "predicted_sensitive": sensitive_count,
        "predicted_non_sensitive": prediction_count - sensitive_count,
        "errors": error_count,
        "baseline_gpu_allocated_gb": baseline_allocated,
        "peak_gpu_allocated_gb": _cuda_gb(torch.cuda.max_memory_allocated),
        "peak_gpu_reserved_gb": _cuda_gb(torch.cuda.max_memory_reserved),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if error_count:
        print(
            f"Completed with {error_count} audited image errors; "
            f"see {output_dir / 'errors.jsonl'}",
            file=sys.stderr,
        )
        return 0 if args.allow_errors else 2
    return 0


def _resolve_threshold(
    explicit_threshold: float | None,
    calibration_path: Path | None,
) -> tuple[float, str]:
    if explicit_threshold is not None:
        if not 0.0 <= explicit_threshold <= 1.0:
            raise ValueError("--threshold must be in [0, 1]")
        return explicit_threshold, "command_line"
    if calibration_path is not None:
        payload = json.loads(
            calibration_path.expanduser().resolve().read_text(encoding="utf-8")
        )
        threshold = float(payload["selected_threshold"])
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("calibration selected_threshold must be in [0, 1]")
        return threshold, f"calibration:{calibration_path}"
    return 0.5, "default"


def _resolve_model(value: str) -> Path:
    path = Path(value).expanduser()
    for candidate in (path, PROJECT_ROOT / path, PROJECT_ROOT.parent / path):
        if (candidate / "config.json").is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve local base model: {value}")


def _cuda_gb(getter) -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(getter() / 1024**3)


if __name__ == "__main__":
    raise SystemExit(main())
