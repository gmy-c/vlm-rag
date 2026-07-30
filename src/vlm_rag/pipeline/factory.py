from __future__ import annotations

from pathlib import Path

from .audit import JsonlAuditWriter
from .engine import SecureRAGEngine
from .policy import (
    SensitivityPolicy,
    SensitivityPolicyConfig,
    load_redaction_manifest,
)
from .provenance import (
    fingerprint_adapter,
    fingerprint_base_model_metadata,
)
from ..generation import DoubaoClientConfig, DoubaoVisionClient
from ..retrieval.index import MultiVectorIndex
from ..retrieval.model import LateInteractionRetriever
from ..sensitivity.catalog import SensitivityCatalog


def build_secure_rag_engine(
    *,
    data_root: Path,
    base_model: Path,
    retrieval_adapter: Path,
    index_dir: Path,
    sensitivity_catalog: Path,
    api_key: str,
    redaction_manifest: Path | None = None,
    doubao_config: DoubaoClientConfig | None = None,
    coarse_top_k: int = 128,
    rerank_top_k: int = 20,
    answer_page_limit: int = 3,
    maxsim_backend: str = "auto",
    audit_path: Path | None = None,
    include_query_in_audit: bool = False,
    device: str = "cuda",
) -> SecureRAGEngine:
    base_model = base_model.expanduser().resolve()
    retrieval_adapter = retrieval_adapter.expanduser().resolve()
    retriever, _ = LateInteractionRetriever.from_adapter(
        retrieval_adapter,
        checkpoint_path_override=str(base_model),
        device=device,
    )
    retriever.eval()
    index = MultiVectorIndex(
        index_dir,
        expected_adapter_sha256=fingerprint_adapter(retrieval_adapter),
        expected_base_model_metadata_sha256=(
            fingerprint_base_model_metadata(base_model)
        ),
    )
    catalog = SensitivityCatalog.load(sensitivity_catalog)
    policy = SensitivityPolicy(
        catalog=catalog,
        data_root=data_root,
        redactions=load_redaction_manifest(redaction_manifest),
        config=SensitivityPolicyConfig(answer_page_limit=answer_page_limit),
    )
    generator = DoubaoVisionClient(api_key, doubao_config)
    audit_writer = (
        JsonlAuditWriter(
            audit_path,
            include_query=include_query_in_audit,
        )
        if audit_path is not None
        else None
    )
    return SecureRAGEngine(
        retriever=retriever,
        index=index,
        policy=policy,
        generator=generator,
        coarse_top_k=coarse_top_k,
        rerank_top_k=rerank_top_k,
        maxsim_backend=maxsim_backend,
        audit_writer=audit_writer,
    )
