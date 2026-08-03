"""Document-disjoint global and multi-vector retrieval components."""

from .dataset import (
    DocumentUniqueBatchSampler,
    PageGroupedBatchSampler,
    PageGroupedRetrievalBatch,
    RetrievalBatch,
    RetrievalManifestDataset,
    load_hard_negative_map,
    resolve_retrieval_path,
    retrieval_collate,
    page_grouped_retrieval_collate,
)
from .manifest import (
    RetrievalManifestBuildResult,
    build_retrieval_manifests,
)
from .index import MultiVectorIndex, build_multivector_index, unique_pages
from .losses import (
    HybridLossConfig,
    MultiVectorMemoryQueue,
    hybrid_retrieval_loss,
    multi_positive_symmetric_cross_entropy,
    symmetric_global_info_nce,
)
from .maxsim import (
    late_interaction_kernel_available,
    maxsim_score_matrix,
)
from .model import LateInteractionModelConfig, LateInteractionRetriever
from .schema import (
    RetrievalRecord,
    load_retrieval_manifest,
    write_retrieval_manifest,
)

__all__ = [
    "DocumentUniqueBatchSampler",
    "HybridLossConfig",
    "LateInteractionModelConfig",
    "LateInteractionRetriever",
    "MultiVectorMemoryQueue",
    "PageGroupedBatchSampler",
    "PageGroupedRetrievalBatch",
    "MultiVectorIndex",
    "RetrievalBatch",
    "RetrievalManifestBuildResult",
    "RetrievalManifestDataset",
    "RetrievalRecord",
    "build_retrieval_manifests",
    "build_multivector_index",
    "hybrid_retrieval_loss",
    "multi_positive_symmetric_cross_entropy",
    "late_interaction_kernel_available",
    "load_retrieval_manifest",
    "load_hard_negative_map",
    "resolve_retrieval_path",
    "retrieval_collate",
    "page_grouped_retrieval_collate",
    "maxsim_score_matrix",
    "symmetric_global_info_nce",
    "write_retrieval_manifest",
    "unique_pages",
]
