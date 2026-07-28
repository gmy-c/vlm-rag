from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vlm_rag.pipeline import run_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight VLM-RAG demo.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    data_dir = _project_path(args.data_dir)
    output_dir = _project_path(args.output_dir)

    metrics = run_demo(data_dir, output_dir, top_k=args.top_k, project_root=PROJECT_ROOT)
    print("VLM-RAG demo finished.")
    for name, value in metrics.items():
        print(f"{name}: {value}")
    print(f"results: {_relative_to_project(output_dir / 'retrieval_results.json')}")


def _project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _relative_to_project(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    main()
