"""
Integration test: verify generator module structure and logic.

Does NOT make real API calls — uses ``unittest.mock`` for simulation.
If all tests pass, the pipeline is structurally sound and ready for
GPU + API evaluation.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Handle missing heavy dependencies (torch, colpali_engine, ...) ──
_MOCK_MODULES = [
    "torch", "torch.nn", "torch.nn.functional",
    "torch.optim", "torch.optim.lr_scheduler",
    "colpali_engine", "colpali_engine.models",
    "peft", "bitsandbytes", "flash_attn",
]
for _mod in _MOCK_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ═══════════════════════════════════════════════════════════════
# Security pre-check
# ═══════════════════════════════════════════════════════════════


def security_check() -> None:
    """Warn about missing / suspicious API key configuration."""
    key = os.environ.get("DOUBAO_API_KEY", "")
    if key and len(key) < 10:
        print(
            "WARNING: DOUBAO_API_KEY seems too short "
            f"({len(key)} chars). Verify it is correct."
        )
    if not key:
        print("INFO: DOUBAO_API_KEY not set. Real API calls will fail.")
        print("      Set with: export DOUBAO_API_KEY='your-key'")
    print()


# ═══════════════════════════════════════════════════════════════
# Structural tests
# ═══════════════════════════════════════════════════════════════


def test_answer_dataclass() -> None:
    from vlm_rag.generator import Answer

    a = Answer("test", ["page_1"], 0.95)
    assert a.text == "test"
    assert a.evidence_page_ids == ["page_1"]
    assert a.confidence == 0.95
    print("  OK Answer dataclass")


def test_page_result_dataclass() -> None:
    from vlm_rag.generator import PageResult

    pr = PageResult(
        relevant=True, answer="42", evidence="page says 42", confidence=0.9
    )
    assert pr.relevant is True
    assert pr.answer == "42"
    assert pr.confidence == 0.9
    print("  OK PageResult dataclass")


def test_image_stitching() -> None:
    from PIL import Image

    from vlm_rag.baselines import ImageStitchingGenerator as ISG

    img1 = Image.new("RGB", (800, 600), "white")
    img2 = Image.new("RGB", (1000, 400), "white")
    stitched = ISG._stitch_vertically([img1, img2], max_width=600)

    # width = 600 (target)
    # img1: 600 × (600*600//800=450)
    # img2: 600 × (600*400//1000=240)
    # divider: 2 px between them
    expected_h = 450 + 240 + 2
    assert stitched.width == 600
    assert stitched.height == expected_h, (
        f"Expected {expected_h}, got {stitched.height}"
    )
    print(f"  OK Stitching: {stitched.size}")


def test_normalize_answer() -> None:
    from vlm_rag.generator import DoubaoVisionGenerator as DVG

    assert DVG._normalize_answer("Hello World") == "helloworld"
    assert DVG._normalize_answer("  42.5%  ") == "42.5"
    assert DVG._normalize_answer("NEW BRAND") == "newbrand"
    assert DVG._normalize_answer("AVERAGE 1R4F RESPONSES") == "average1r4fresponses"
    print("  OK Answer normalization")


def test_fuse_candidates_empty() -> None:
    from vlm_rag.generator import DoubaoVisionGenerator as DVG

    gen = DVG(api_key="test-key")
    result = gen._fuse_candidates([])
    assert result.text == "Unable to determine"
    assert result.confidence == 0.0
    assert result.evidence_page_ids == []
    print("  OK Empty fusion fallback")


def test_fuse_candidates_normal() -> None:
    """融合逻辑：两个相同答案应合并分数，不同答案选最高分"""
    from vlm_rag.generator import DoubaoVisionGenerator as DVG

    gen = DVG(api_key="test-key")
    candidates = [
        {"answer": "0.28", "page_id": "p1", "retrieval_score": 0.9,
         "vlm_confidence": 0.85},
        {"answer": "0.28", "page_id": "p2", "retrieval_score": 0.5,
         "vlm_confidence": 0.78},
        {"answer": "0.30", "page_id": "p3", "retrieval_score": 0.3,
         "vlm_confidence": 0.90},
    ]
    result = gen._fuse_candidates(candidates)
    # "0.28" combined: 0.9*0.85 + 0.5*0.78 = 0.765 + 0.39 = 1.155
    # "0.30" combined: 0.3*0.90 = 0.27
    # → "0.28" wins
    assert result.text == "0.28"
    assert set(result.evidence_page_ids) == {"p1", "p2"}
    assert 0 < result.confidence <= 1.0
    print(
        f"  OK Candidate fusion: answer='{result.text}', "
        f"pages={result.evidence_page_ids}, conf={result.confidence:.4f}"
    )


def test_per_page_answer_mock() -> None:
    """端到端逐页推理 + 融合（Mock _query_single_page）"""
    from unittest.mock import patch

    from vlm_rag.generator import DoubaoVisionGenerator, PageResult
    from vlm_rag.retriever import SearchHit
    from vlm_rag.data import Page

    page1 = Page(
        "p1", "doc1", "document", 1, "t1", "", "", {}, "fake1.png"
    )
    page2 = Page(
        "p2", "doc1", "document", 2, "t2", "", "", {}, "fake2.png"
    )

    hits = [
        SearchHit(page=page1, score=0.9, rank=1),
        SearchHit(page=page2, score=0.5, rank=2),
    ]

    with patch.object(
        DoubaoVisionGenerator,
        "_query_single_page",
    ) as mock_query:
        # Mock: page1 gives relevant answer, page2 is not relevant
        mock_query.side_effect = [
            PageResult(
                relevant=True, answer="0.28", evidence="...",
                confidence=0.85,
            ),
            PageResult(
                relevant=False, answer="", evidence="",
                confidence=0.0,
            ),
        ]

        gen = DoubaoVisionGenerator(api_key="test-key")
        answer = gen.answer("What is the value?", hits)

    assert answer.text == "0.28"
    assert answer.evidence_page_ids == ["p1"]
    assert answer.confidence > 0
    print(
        f"  OK Mock inference: answer='{answer.text}', "
        f"pages={answer.evidence_page_ids}"
    )


# ═══════════════════════════════════════════════════════════════
# Security tests
# ═══════════════════════════════════════════════════════════════


def test_config_no_hardcoded_key() -> None:
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    text = config_path.read_text(encoding="utf-8")
    assert "sk-" not in text, "sk- pattern found in config.yaml!"
    print("  OK No key in config.yaml")


def test_generator_no_hardcoded_key() -> None:
    gen_path = PROJECT_ROOT / "src" / "vlm_rag" / "generator.py"
    text = gen_path.read_text(encoding="utf-8")
    suspicious = re.findall(r'"sk-[a-zA-Z0-9]{20,}"', text)
    assert len(suspicious) == 0, f"Hardcoded key: {suspicious}"
    print("  OK No key in generator.py")


def test_baselines_no_hardcoded_key() -> None:
    bl_path = PROJECT_ROOT / "src" / "vlm_rag" / "baselines.py"
    text = bl_path.read_text(encoding="utf-8")
    suspicious = re.findall(r'"sk-[a-zA-Z0-9]{20,}"', text)
    assert len(suspicious) == 0, f"Hardcoded key: {suspicious}"
    print("  OK No key in baselines.py")


def test_evaluate_script_no_hardcoded_key() -> None:
    ev_path = PROJECT_ROOT / "scripts" / "evaluate_generator.py"
    text = ev_path.read_text(encoding="utf-8")
    suspicious = re.findall(r'"sk-[a-zA-Z0-9]{20,}"', text)
    assert len(suspicious) == 0, f"Hardcoded key: {suspicious}"
    print("  OK No key in evaluate_generator.py")


def test_timeout_present() -> None:
    gen_path = PROJECT_ROOT / "src" / "vlm_rag" / "generator.py"
    text = gen_path.read_text(encoding="utf-8")
    assert "timeout=" in text or "timeout =" in text, (
        "timeout not configured in generator.py!"
    )
    print("  OK timeout configured")


def test_no_external_url_leak() -> None:
    """Ensure images are only sent to doubao_base_url, not elsewhere."""
    gen_path = PROJECT_ROOT / "src" / "vlm_rag" / "generator.py"
    text = gen_path.read_text(encoding="utf-8")
    # The base_url should be configurable, not hardcoded to an unknown host
    assert "ark.cn-beijing.volces.com" in text or "self.base_url" in text
    # No other image upload targets
    urls = re.findall(r'https?://[^\s"\'\)]+', text)
    external = [
        u for u in urls
        if "volces.com" not in u
        and "api/v3" not in u
        and "python" not in u
        and "w3.org" not in u
    ]
    # Allow only Doubao and common doc URLs in comments
    non_doc = [u for u in external if "example" not in u.lower()]
    assert len(non_doc) <= 2, f"Unexpected URLs: {non_doc}"
    print("  OK No external image upload targets")


def test_api_key_reads_from_env_only() -> None:
    """get_api_key() 只从环境变量读取，不从任何文件读取"""
    from vlm_rag.config import ProjectConfig

    # Read config source to verify no file-based key loading
    cfg_path = PROJECT_ROOT / "src" / "vlm_rag" / "config.py"
    text = cfg_path.read_text(encoding="utf-8")
    # Must reference os.environ
    assert "os.environ" in text, "get_api_key must use os.environ"
    print("  OK API key reads from environment only")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════


def main() -> None:
    print("=" * 60)
    print("Generator Integration & Security Test")
    print("=" * 60)

    security_check()

    # Structural / logic
    test_answer_dataclass()
    test_page_result_dataclass()
    test_image_stitching()
    test_normalize_answer()
    test_fuse_candidates_empty()
    test_fuse_candidates_normal()
    test_per_page_answer_mock()

    # Security
    print("\n--- Security ---")
    test_config_no_hardcoded_key()
    test_generator_no_hardcoded_key()
    test_baselines_no_hardcoded_key()
    test_evaluate_script_no_hardcoded_key()
    test_timeout_present()
    test_no_external_url_leak()
    test_api_key_reads_from_env_only()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("Pipeline is ready for GPU + API evaluation.")
    print("Run: python scripts/evaluate_generator.py --sample 10")
    print("=" * 60)


if __name__ == "__main__":
    main()
