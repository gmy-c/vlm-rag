from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vlm_rag.sensitivity.catalog import build_catalog, write_catalog


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Merge sensitivity predictions with the full page manifest into "
            "a strict runtime safety catalog."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--errors", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-image-hashes",
        action="store_true",
        help="Faster but weaker provenance; full production catalogs should hash images.",
    )
    args = parser.parse_args()

    entries, metadata = build_catalog(
        manifest_path=args.manifest.expanduser().resolve(),
        predictions_path=args.predictions.expanduser().resolve(),
        errors_path=(
            args.errors.expanduser().resolve()
            if args.errors is not None
            else None
        ),
        data_root=args.data_root.expanduser().resolve(),
        checkpoint_path=args.checkpoint.expanduser().resolve(),
        calibration_path=args.calibration.expanduser().resolve(),
        hash_images=not args.skip_image_hashes,
    )
    output = args.output.expanduser().resolve()
    write_catalog(output, entries, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Sensitivity catalog: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
