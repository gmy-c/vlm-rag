from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vlm_rag.sensitivity import ManifestBuildError, build_sensitivity_manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build page-level sensitivity manifests. Source images/OCR are "
            "validated but never moved, copied, renamed, or modified."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=(
            "Dataset root containing docvqa_images/, ocr/, docvqa_extracted/, "
            "and desensitized/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <data-root>/manifests/sensitivity).",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--verify-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Pillow to verify every full and positive-subset PNG (default: enabled).",
    )
    parser.add_argument(
        "--validate-ocr-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Parse every full and positive-subset OCR JSON file (default: enabled).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = build_sensitivity_manifests(
            data_root=args.data_root,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            verify_images=args.verify_images,
            validate_ocr_json=args.validate_ocr_json,
        )
    except (ManifestBuildError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Sensitivity manifests built successfully.")
    print(f"output_dir: {result.output_dir}")
    print(json.dumps(result.summary["total"], ensure_ascii=False, sort_keys=True))
    for split, stats in result.summary["splits"].items():
        print(f"{split}: {json.dumps(stats, ensure_ascii=False, sort_keys=True)}")
    overlap = result.summary["validation"]["generated_document_overlap"]
    print(f"document_overlap: {json.dumps(overlap, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
