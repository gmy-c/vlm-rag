"""Baseline generation methods for multi-page visual QA.

Implements:
    - ImageStitchingGenerator: vertical page stitching baseline.
    - evaluate_generation_methods: unified comparison framework.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image

from .data import Page, Query
from .generator import Answer, DoubaoVisionGenerator
from .metrics import accuracy
from .retriever import SearchHit


# ═══════════════════════════════════════════════════════════════
# Prompt
# ═══════════════════════════════════════════════════════════════

STITCHED_PROMPT = """\
You are looking at {k} document pages combined vertically into one image
(separated by thin gray lines). Page 1 is at the top, Page {k} at the bottom.

Answer the following question based on the combined image.

Question: {query}

Instructions:
1. Scan ALL {k} pages for relevant information.
2. If you find the answer, state it clearly and list which page number(s) \
(1-{k}) contain the evidence.
3. If none of the pages contain the answer, set answer to empty string.

Return ONLY a JSON object:
{{"answer": "...", "evidence_page_numbers": [1, 3], "confidence": 0.0-1.0}}\
"""


# ═══════════════════════════════════════════════════════════════
# Image stitching baseline
# ═══════════════════════════════════════════════════════════════


class ImageStitchingGenerator:
    """Vertical page-stitching baseline for multi-page visual QA.

    Concatenates Top-K pages into a single tall image and sends it to
    the Doubao Vision API in one call.  Used to verify that the main
    per-page-inference + fusion approach outperforms naive stitching.

    Composition pattern: holds a *DoubaoVisionGenerator* instance to
    reuse its ``_call_doubao_api`` method (same retry logic, same
    auth, same endpoint).
    """

    # Per-page divider: 2 px in neutral gray
    _DIVIDER_HEIGHT = 2
    _DIVIDER_COLOR = (128, 128, 128)

    def __init__(
        self,
        generator: DoubaoVisionGenerator,
        *,
        max_width: int = 1200,
        max_file_mb: float = 20.0,
        fallback_width: int = 800,
    ) -> None:
        """Initialise the stitching generator.

        Args:
            generator: A configured ``DoubaoVisionGenerator`` whose
                ``_call_doubao_api`` method will be reused.
            max_width: Target width (px) for the stitched canvas.
            max_file_mb: Maximum PNG file size (MB) before auto-downscaling.
            fallback_width: Width to use when downscaling is triggered.
        """
        self._generator = generator
        self._max_width = max_width
        self._max_file_mb = max_file_mb
        self._fallback_width = fallback_width

    # ═══════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════

    def answer(self, query: str, hits: list[SearchHit]) -> Answer:
        """Generate an answer from stitched Top-K pages."""
        # Guard
        if not hits:
            return Answer("Unable to determine", [], 0.0)

        # ── 1. Load images ──
        images: list[Image.Image] = []
        valid_pages: list[Page] = []

        for hit in hits:
            try:
                img = Image.open(hit.page.image_path).convert("RGB")
                images.append(img)
                valid_pages.append(hit.page)
            except (FileNotFoundError, OSError):
                continue

        if not images:
            return Answer("Unable to determine", [], 0.0)

        # ── 2. Vertical stitch ──
        try:
            stitched = self._stitch_vertically(images, self._max_width)
        except Exception:
            return Answer("Image stitching failed", [], 0.0)

        # ── 3. File size guard ──
        buf = io.BytesIO()
        stitched.save(buf, format="PNG")
        size_mb = buf.tell() / (1024 * 1024)

        if size_mb > self._max_file_mb:
            # Downscale: re-stitch at fallback width
            buf = io.BytesIO()
            stitched = self._stitch_vertically(images, self._fallback_width)
            stitched.save(buf, format="PNG")

        # ── 4. Base64 encode ──
        b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")

        # ── 5. Call Doubao API (reuse generator's internal method) ──
        prompt = STITCHED_PROMPT.format(k=len(images), query=query)
        try:
            response = self._generator._call_doubao_api(b64_data, prompt)
        except Exception:
            return Answer("API error", [], 0.0)

        # ── 6. Parse response ──
        answer_text = str(response.get("answer", ""))
        evidence_nums: list[int] = response.get("evidence_page_numbers", [])

        evidence_pages = [
            valid_pages[i - 1].page_id
            for i in evidence_nums
            if 1 <= i <= len(valid_pages)
        ]
        confidence = float(response.get("confidence", 0.0))

        if not answer_text:
            return Answer("Unable to determine", [], 0.0)

        return Answer(answer_text, evidence_pages, round(confidence, 4))

    # ═══════════════════════════════════════════════════════════
    # Stitching logic
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _stitch_vertically(
        images: list[Image.Image],
        max_width: int = 1200,
    ) -> Image.Image:
        """Vertically concatenate images into a single canvas.

        Steps:
            1. Resize every image to *max_width* (preserving aspect ratio).
            2. Compute total height = sum of resized heights +
               divider heights between pages.
            3. Create a white canvas and paste images top-to-bottom.
            4. Draw a 2 px gray line between consecutive pages.
        """
        # Resize
        resized: list[Image.Image] = []
        for img in images:
            ratio = max_width / img.width
            new_h = int(img.height * ratio)
            resized.append(img.resize((max_width, new_h), Image.LANCZOS))

        # Total height
        n = len(resized)
        total_height = (
            sum(img.height for img in resized)
            + ImageStitchingGenerator._DIVIDER_HEIGHT * max(0, n - 1)
        )

        # Canvas
        canvas = Image.new("RGB", (max_width, total_height), "white")

        # Paste
        y_offset = 0
        for idx, img in enumerate(resized):
            canvas.paste(img, (0, y_offset))
            y_offset += img.height

            # Draw divider (except after the last page)
            if idx < n - 1:
                for px in range(ImageStitchingGenerator._DIVIDER_HEIGHT):
                    for x in range(max_width):
                        canvas.putpixel(
                            (x, y_offset + px),
                            ImageStitchingGenerator._DIVIDER_COLOR,
                        )
                y_offset += ImageStitchingGenerator._DIVIDER_HEIGHT

        return canvas


# ═══════════════════════════════════════════════════════════════
# Unified evaluation
# ═══════════════════════════════════════════════════════════════


def evaluate_generation_methods(
    retriever: object,       # DualTowerRetriever
    pages: list[Page],
    queries: list[Query],
    api_key: str,
    top_k: int = 3,
) -> dict[str, dict[str, float]]:
    """Evaluate per-page vs stitched generation on the same retrieval results.

    Args:
        retriever: A ``DualTowerRetriever`` instance with an already-built index.
        pages: All pages in the evaluation set.
        queries: All queries in the evaluation set.
        api_key: Doubao API key.
        top_k: Number of pages to retrieve per query.

    Returns:
        Dict mapping method name → {"accuracy": float, "em": float}.
    """
    results: dict[str, dict[str, float]] = {}

    # ── Build per-page generator ──
    per_page_gen = DoubaoVisionGenerator(api_key=api_key)

    # ── Build stitched generator (composition) ──
    stitched_gen = ImageStitchingGenerator(generator=per_page_gen)

    methods: list[tuple[str, object]] = [
        ("per_page", per_page_gen),
        ("stitched", stitched_gen),
    ]

    for method_name, gen in methods:
        predictions: dict[str, str] = {}

        for q in queries:
            hits = retriever.search(q.text, top_k=top_k)
            answer = gen.answer(q.text, hits)
            predictions[q.query_id] = answer.text

        acc = accuracy(predictions, queries)
        results[method_name] = {
            "accuracy": round(acc, 4),
            "em": round(acc, 4),
        }

    return results
