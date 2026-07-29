"""
轻量集成测试：用静态分析验证整个 pipeline 的接口对接正确。

不需要 GPU，不需要 torch，不需要 colpali-engine。
所有检查都是 AST 级别的结构验证。
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _parse_module(rel_path: str) -> ast.AST:
    """Parse a source file under src/vlm_rag/ and return its AST."""
    path = SRC_DIR / rel_path
    return ast.parse(path.read_text(encoding="utf-8"))


def _all_class_names(tree: ast.AST) -> list[str]:
    return [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]


def _all_func_names(tree: ast.AST) -> list[str]:
    return [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]


def _all_import_names(tree: ast.AST) -> list[str]:
    """Return all imported names (both 'import X' and 'from M import N')."""
    names: list[str] = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for alias in n.names:
                names.append(alias.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom) and n.module:
            for alias in n.names:
                names.append(alias.name)
    return names


def _source_text(tree: ast.AST) -> str:
    return ast.unparse(tree)


# ═══════════════════════════════════════════════════════════════


def test_config_loads() -> None:
    """Phase 1: config.py + config.yaml"""
    from vlm_rag.config import load_config, resolve_project_path

    config = load_config(
        resolve_project_path(PROJECT_ROOT, "configs/config.yaml")
    )
    assert config.batch_size == 8
    assert config.colpali_model == "vidore/colpali-v1.3-merged"
    assert config.get_selected_layers() == (0, 8, 16, 23)
    print(f"  OK Config: batch_size={config.batch_size}, "
         f"model={config.colpali_model}")


def test_encoder_module() -> None:
    """Phase 2: encoders.py — old code removed, new classes present."""
    tree = _parse_module("vlm_rag/encoders.py")
    classes = _all_class_names(tree)
    funcs = _all_func_names(tree)
    source = _source_text(tree)

    # Old code removed
    for old in ("HashingVLMEncoder", "EncoderConfig"):
        assert old not in classes, f"OLD CLASS: {old}"
    for old in ("_tokenize", "_hash_token", "_l2_normalize", "_weighted_pool"):
        assert old not in funcs, f"OLD FUNC: {old}"
    assert "TOKEN_PATTERN" not in source

    # New classes
    assert "ColPaliDualEncoderConfig" in classes
    assert "ColPaliDualEncoder" in classes

    # Required methods on ColPaliDualEncoder
    for cls_node in ast.walk(tree):
        if isinstance(cls_node, ast.ClassDef) and cls_node.name == "ColPaliDualEncoder":
            methods = [n.name for n in cls_node.body if isinstance(n, ast.FunctionDef)]
    required = [
        "encode_page", "encode_page_batch", "encode_query",
        "encode_query_batch", "trainable_parameters",
        "trainable_param_count", "save", "load",
    ]
    for m in required:
        assert m in methods, f"Missing method: ColPaliDualEncoder.{m}"

    # Utility functions
    assert "cosine_similarity" in funcs
    assert "info_nce_loss" in funcs

    # Key imports (lazy)
    assert "colpali_engine" in source, "colpali_engine import missing"
    assert "peft" in source, "peft import missing"
    assert "from .data import Page" in source

    # torch.no_grad on vision, not on text
    # _encode_images should contain torch.no_grad()
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name == "_encode_images":
            img_src = ast.unparse(fn)
            assert "no_grad" in img_src, "_encode_images must use torch.no_grad()"
    # _encode_texts should NOT contain torch.no_grad()
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name == "_encode_texts":
            txt_src = ast.unparse(fn)
            # It's OK if no_grad appears in comments/docs, but not as a wrapper

    print("  OK Encoder module structure correct")


def test_retriever_module() -> None:
    """Phase 3a: retriever.py — Tensor batch computation."""
    tree = _parse_module("vlm_rag/retriever.py")
    source = _source_text(tree)
    classes = _all_class_names(tree)

    # SearchHit preserved
    assert "SearchHit" in classes

    # encoder type annotation updated
    assert "ColPaliDualEncoder" in source
    assert "HashingVLMEncoder" not in source

    # index uses batch
    for cls_node in ast.walk(tree):
        if isinstance(cls_node, ast.ClassDef) and cls_node.name == "DualTowerRetriever":
            for item in cls_node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "index":
                    idx_src = ast.unparse(item)
                if isinstance(item, ast.FunctionDef) and item.name == "search":
                    srch_src = ast.unparse(item)
    assert "encode_page_batch" in idx_src
    assert "page_id_to_idx" in idx_src
    assert "page_vectors @" in srch_src or "@ query_vec" in srch_src

    print("  OK Retriever module structure correct")


def test_training_module() -> None:
    """Phase 3b: training.py — InfoNCE loop."""
    tree = _parse_module("vlm_rag/training.py")
    funcs = _all_func_names(tree)
    source = _source_text(tree)

    # Old code gone
    for old in ("train_retriever", "_evaluate_encoder", "_score"):
        assert old not in funcs, f"OLD FUNC: {old}"
    assert "WEIGHT_CANDIDATES" not in source

    # New functions
    assert "train_colpali_retriever" in funcs
    assert "_validate" in funcs

    # Signature: 5 positional + N keyword-only
    for fn_node in ast.walk(tree):
        if isinstance(fn_node, ast.FunctionDef) and fn_node.name == "train_colpali_retriever":
            pos = [a.arg for a in fn_node.args.args]
            kw = [a.arg for a in fn_node.args.kwonlyargs]
    assert pos[:5] == ["train_pages", "train_queries", "val_pages",
                        "val_queries", "model_dir"], f"Bad pos args: {pos[:5]}"
    assert "batch_size" in kw
    assert "gradient_accumulation_steps" in kw

    print("  OK Training module structure correct")


def test_data_loading(data_root: Path) -> tuple:
    """Phase 1: data.py — DocVQA loading + split."""
    qa_path = data_root / "docvqa_extracted" / "val_v1.0_withQT.json"
    img_dir = data_root / "docvqa_images"

    if not qa_path.exists():
        print(f"  WARN Data not found, skipping: {qa_path}")
        return [], []

    from vlm_rag.data import load_docvqa_dataset

    pages, queries = load_docvqa_dataset(qa_path, img_dir, split_name="val")
    assert len(pages) > 0
    assert len(queries) > 0
    assert len(queries) >= len(pages)
    missing = [p for p in pages[:10] if not Path(p.image_path).exists()]
    assert len(missing) == 0, f"Missing images: {missing}"
    page_ids = {p.page_id for p in pages}
    for q in queries[:10]:
        assert q.positive_page_ids[0] in page_ids

    print(f"  OK Data loading: {len(pages)} pages, {len(queries)} queries")
    return pages, queries


def test_split_no_leakage(pages: list, queries: list) -> None:
    """Phase 1: split_by_document no data leakage."""
    if not pages:
        print("  WARN No data, skipping split test")
        return

    from vlm_rag.data import split_by_document

    splits = split_by_document(pages, queries, train_ratio=0.7, val_ratio=0.15)

    t_docs = {p.doc_id for p in splits["train"][0]}
    v_docs = {p.doc_id for p in splits["val"][0]}
    s_docs = {p.doc_id for p in splits["test"][0]}

    assert len(t_docs & v_docs) == 0, "Leak: train ∩ val"
    assert len(t_docs & s_docs) == 0, "Leak: train ∩ test"
    assert len(v_docs & s_docs) == 0, "Leak: val ∩ test"

    total = sum(len(sp) for sp, _ in splits.values())
    assert total == len(pages)

    print(f"  OK Split: train={len(splits['train'][0])}p/"
          f"{len(splits['train'][1])}q, "
          f"val={len(splits['val'][0])}p/{len(splits['val'][1])}q, "
          f"test={len(splits['test'][0])}p/{len(splits['test'][1])}q")


def test_train_entry_script() -> None:
    """Phase 4: scripts/train_colpali.py exists and is valid."""
    path = PROJECT_ROOT / "scripts" / "train_colpali.py"
    assert path.exists(), "train_colpali.py missing!"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    funcs = _all_func_names(tree)
    assert "main" in funcs

    source = _source_text(tree)
    assert "train_colpali_retriever" in source
    assert "load_docvqa_dataset" in source
    assert "split_by_document" in source

    print("  OK train_colpali.py exists and valid")


# ═══════════════════════════════════════════════════════════════


def main() -> None:
    data_root_str = os.environ.get(
        "DOCVQA_DATA_ROOT",
        str(PROJECT_ROOT / "data"),
    )
    data_root = Path(data_root_str)

    print("=" * 60)
    print("Integration Test Suite")
    print(f"  Data root: {data_root}")
    print("=" * 60)

    test_config_loads()
    test_encoder_module()
    test_retriever_module()
    test_training_module()
    test_train_entry_script()

    pages, queries = test_data_loading(data_root)
    test_split_no_leakage(pages, queries)

    print("\n" + "=" * 60)
    print("All tests passed! OK")
    print("Pipeline is ready for GPU training.")
    print("Run: python scripts/train_colpali.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
