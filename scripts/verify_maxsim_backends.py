from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vlm_rag.retrieval.maxsim import (
    late_interaction_kernel_available,
    maxsim_score_matrix,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify LIK and chunked MaxSim value/gradient equivalence."
    )
    parser.add_argument("--atol", type=float, default=0.03)
    parser.add_argument("--rtol", type=float, default=0.03)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not late_interaction_kernel_available():
        raise RuntimeError("late-interaction-kernels is not available")
    torch.manual_seed(42)
    device = torch.device("cuda")
    queries = torch.nn.functional.normalize(
        torch.randn(2, 32, 128, device=device, dtype=torch.bfloat16),
        dim=-1,
    )
    documents = torch.nn.functional.normalize(
        torch.randn(3, 1024, 128, device=device, dtype=torch.bfloat16),
        dim=-1,
    )
    queries[:, -8:] = 0
    values: dict[str, dict[str, float]] = {}
    for normalization in ("mean", "sum"):
        chunked = maxsim_score_matrix(
            queries,
            documents,
            backend="chunked",
            normalization=normalization,
        )
        lik = maxsim_score_matrix(
            queries,
            documents,
            backend="lik",
            normalization=normalization,
            query_batch_chunk=1,
            document_batch_chunk=1,
        )
        torch.testing.assert_close(
            lik.float(),
            chunked.float(),
            atol=args.atol,
            rtol=args.rtol,
        )
        difference = (lik.float() - chunked.float()).abs()
        chunked_queries = queries.detach().clone().requires_grad_(True)
        chunked_documents = documents.detach().clone().requires_grad_(True)
        lik_queries = queries.detach().clone().requires_grad_(True)
        lik_documents = documents.detach().clone().requires_grad_(True)
        chunked_loss = maxsim_score_matrix(
            chunked_queries,
            chunked_documents,
            backend="chunked",
            normalization=normalization,
        ).sum()
        lik_loss = maxsim_score_matrix(
            lik_queries,
            lik_documents,
            backend="lik",
            normalization=normalization,
            query_batch_chunk=1,
            document_batch_chunk=1,
        ).sum()
        chunked_grads = torch.autograd.grad(
            chunked_loss, (chunked_queries, chunked_documents)
        )
        lik_grads = torch.autograd.grad(
            lik_loss, (lik_queries, lik_documents)
        )
        gradient_differences = []
        for lik_grad, chunked_grad in zip(lik_grads, chunked_grads):
            torch.testing.assert_close(
                lik_grad.float(),
                chunked_grad.float(),
                atol=args.atol,
                rtol=args.rtol,
            )
            gradient_differences.append(
                float((lik_grad.float() - chunked_grad.float()).abs().max())
            )
        values[normalization] = {
            "max_abs_difference": float(difference.max()),
            "mean_abs_difference": float(difference.mean()),
            "max_gradient_difference": max(gradient_differences),
        }
    report = {
        "status": "passed",
        "device": torch.cuda.get_device_name(0),
        "dtype": "bfloat16",
        "atol": args.atol,
        "rtol": args.rtol,
        "results": values,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
