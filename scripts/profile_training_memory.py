from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Profile isolated real train/validation steps and select the "
            "largest batch below a hard reserved-memory ceiling."
        )
    )
    parser.add_argument(
        "--task",
        choices=("global", "late", "sensitivity-head", "sensitivity-unfreeze4"),
        required=True,
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=int, nargs="+", default=None)
    parser.add_argument("--max-reserved-gb", type=float, default=28.0)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    defaults = {
        "global": (
            Path("configs/retrieval_global_5090.yaml"),
            [64, 96, 128],
        ),
        "late": (
            Path("configs/retrieval_late_5090.yaml"),
            [2, 4, 6, 8],
        ),
        "sensitivity-head": (
            Path("configs/sensitivity_head_5090.yaml"),
            [16, 24, 32, 40],
        ),
        "sensitivity-unfreeze4": (
            Path("configs/sensitivity_unfreeze4_5090.yaml"),
            [8, 12, 16, 24],
        ),
    }
    config, default_candidates = defaults[args.task]
    config = (args.config or config).resolve()
    candidates = args.candidates or default_candidates
    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "task": args.task,
        "config": str(config),
        "hard_ceiling_reserved_gb": args.max_reserved_gb,
        "results": [],
        "recommended_batch_size": None,
    }

    for batch_size in candidates:
        run_dir = work_dir / f"{args.task}-{batch_size}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        command, summary_path = _command(
            args.task,
            args.data_root,
            args.model,
            config,
            run_dir,
            batch_size,
            args.num_workers,
            raw,
        )
        environment = os.environ.copy()
        environment.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            "expandable_segments:True",
        )
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        result: dict[str, Any] = {
            "batch_size": batch_size,
            "return_code": completed.returncode,
        }
        if completed.returncode == 0 and summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            reserved = float(summary.get("peak_reserved_gb", 0.0))
            result.update(
                {
                    "status": (
                        "accepted"
                        if reserved <= args.max_reserved_gb
                        else "over_ceiling"
                    ),
                    "peak_allocated_gb": summary.get("peak_allocated_gb"),
                    "peak_reserved_gb": reserved,
                }
            )
            if reserved <= args.max_reserved_gb:
                report["recommended_batch_size"] = batch_size
        else:
            lowered = completed.stdout.lower()
            result["status"] = (
                "oom" if "out of memory" in lowered else "failed"
            )
            result["output_tail"] = completed.stdout[-3000:]
        report["results"].append(result)
        print(json.dumps(result, ensure_ascii=False))
        shutil.rmtree(run_dir, ignore_errors=True)

    report_path = work_dir / f"{args.task}-memory-profile.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Profile report: {report_path}")
    if report["recommended_batch_size"] is None:
        print("No candidate stayed below the configured ceiling.")
        return 1
    return 0


def _command(
    task: str,
    data_root: Path,
    model: str,
    config: Path,
    output: Path,
    batch_size: int,
    workers: int,
    raw: dict[str, Any],
) -> tuple[list[str], Path]:
    common = [sys.executable]
    if task == "global":
        command = common + [
            "scripts/train_retrieval_global.py",
            "--config", str(config),
            "--data-root", str(data_root),
            "--model", model,
            "--output-dir", str(output),
            "--batch-size", str(batch_size),
            "--epochs", "1",
            "--num-workers", str(workers),
            "--max-train-samples", str(batch_size * 2),
        ]
        return command, output / "global_training_state.json"
    if task == "late":
        command = common + [
            "scripts/train_retrieval_late.py",
            "--config", str(config),
            "--data-root", str(data_root),
            "--model", model,
            "--output-dir", str(output),
            "--micro-batch-size", str(batch_size),
            "--gradient-accumulation-steps", "1",
            "--epochs", "1",
            "--num-workers", str(workers),
            "--max-train-samples", str(batch_size * 2),
            "--max-val-samples", str(batch_size),
        ]
        return command, output / "training_summary.json"
    command = common + [
        "scripts/train_sensitivity.py",
        "--config", str(config),
        "--data-root", str(data_root),
        "--model", model,
        "--output-dir", str(output),
        "--batch-size", str(batch_size),
        "--gradient-accumulation-steps", "1",
        "--epochs", "1",
        "--max-train-samples", str(max(batch_size * 2, 4)),
            "--max-val-samples", str(max(batch_size, 4)),
        "--num-workers", str(workers),
    ]
    return command, output / "training_summary.json"


if __name__ == "__main__":
    raise SystemExit(main())
