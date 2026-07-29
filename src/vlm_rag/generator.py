"""Multi-page visual generation via Doubao Vision API."""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from PIL import Image

from .retriever import SearchHit


# ═══════════════════════════════════════════════════════════════
# Prompt 模板
# ═══════════════════════════════════════════════════════════════

PER_PAGE_PROMPT = """\
You are analyzing a single page from a scanned document.
Answer the following question based ONLY on what you can see in this page image.

Question: {query}

Instructions:
1. Look carefully at ALL content — text, tables, charts, handwritten notes, headers, footers.
2. If the page CONTAINS information relevant to the question:
   - Set "relevant" to true
   - Provide your best answer using the EXACT text/numbers as they appear
   - Quote the supporting evidence from the page
3. If the page DOES NOT contain relevant information:
   - Set "relevant" to false
   - Leave "answer" and "evidence_quote" as empty strings
4. Confidence guidelines:
   - 0.9-1.0: clearly visible, unambiguous text/numbers
   - 0.7-0.9: handwriting or requires interpretation
   - 0.5-0.7: unclear, noisy, or inferred
   - below 0.5: set relevant=false instead

Return ONLY a JSON object (no markdown, no extra text):
{{"relevant": true/false, "answer": "...", "evidence_quote": "...", "confidence": 0.0-1.0}}\
"""


# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Answer:
    """Final answer from multi-page visual generation.

    Attributes:
        text: The answer string.
        evidence_page_ids: page_ids that contributed evidence.
        confidence: Overall confidence in [0, 1].
    """

    text: str
    evidence_page_ids: list[str]
    confidence: float


@dataclass
class PageResult:
    """Structured output from a single-page VQA call.

    Attributes:
        relevant: Whether the page contains information relevant to the query.
        answer: The answer extracted from this page.
        evidence: Quoted supporting text from the page.
        confidence: Model self-assessed confidence in [0, 1].
    """

    relevant: bool
    answer: str
    evidence: str
    confidence: float


# ═══════════════════════════════════════════════════════════════
# Doubao Vision Generator
# ═══════════════════════════════════════════════════════════════


class DoubaoVisionGenerator:
    """Multi-page visual generator backed by Doubao Vision API.

    Architecture:
        1. Per-page inference: each page image → Base64 → Doubao API → JSON.
        2. Score-weighted fusion: combined = retrieval_score × vlm_confidence.
        3. Fallback: if no page is relevant, return "Unable to determine".

    The Doubao API is OpenAI-compatible.  Images are passed as Base64
    Data URLs; the API key is supplied by the caller at construction time.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
        timeout: int = 30,
    ) -> None:
        """Initialise the generator.

        Args:
            api_key: Volcano Engine Ark API key (from DOUBAO_API_KEY env var).
            model: Model ID.  Defaults to ``doubao-seed-1-6-vision-250815``.
            base_url: API base URL.  Defaults to the Volcano Engine Ark endpoint.
            max_tokens: Maximum output tokens per call.
            temperature: Sampling temperature (lower = more deterministic).
            timeout: HTTP request timeout in seconds.
        """
        self.api_key = api_key
        self.model = model or "doubao-seed-1-6-vision-250815"
        self.base_url = base_url or "https://ark.cn-beijing.volces.com/api/v3"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

    # ═══════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════

    def answer(self, query: str, hits: list[SearchHit]) -> Answer:
        """Generate an answer from the Top-K retrieved pages.

        Phase 1: Query each page independently via the Doubao Vision API.
        Phase 2: Fuse candidate answers weighted by retrieval scores.
        """
        # Guard: empty hits
        if not hits:
            return Answer("Unable to determine", [], 0.0)

        # ── Phase 1: per-page inference ──
        candidates: list[dict] = []

        for hit in hits:
            result = self._query_single_page(hit.page, query)
            if result is not None and result.relevant:
                candidates.append(
                    {
                        "answer": result.answer,
                        "page_id": hit.page.page_id,
                        "retrieval_score": hit.score,
                        "vlm_confidence": result.confidence,
                    }
                )

        # ── Phase 2: score-weighted fusion ──
        if not candidates:
            return Answer("Unable to determine", [], 0.0)

        return self._fuse_candidates(candidates)

    # ═══════════════════════════════════════════════════════════
    # Per-page inference
    # ═══════════════════════════════════════════════════════════

    def _query_single_page(self, page, query: str) -> PageResult | None:
        """Send a single page image + query to the Doubao API.

        Returns ``None`` when the image is missing or the API call fails
        after all retries, so the caller can silently skip this page.
        """
        # ── Encode image ──
        try:
            image_b64 = self._encode_image_base64(page.image_path)
        except (FileNotFoundError, OSError):
            return None

        # ── Call API ──
        prompt = PER_PAGE_PROMPT.format(query=query)
        try:
            response = self._call_doubao_api(image_b64, prompt)
        except Exception:
            return None

        # ── Build result ──
        return PageResult(
            relevant=bool(response.get("relevant", False)),
            answer=str(response.get("answer", "")),
            evidence=str(response.get("evidence_quote", "")),
            confidence=float(response.get("confidence", 0.0)),
        )

    # ═══════════════════════════════════════════════════════════
    # API call with retry
    # ═══════════════════════════════════════════════════════════

    def _call_doubao_api(self, image_base64: str, prompt: str) -> dict:
        """Call the Doubao Vision chat completions endpoint.

        Returns the parsed JSON body of the assistant's response.
        Retries up to 2 extra times (3 total) with exponential back-off.
        """
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }

        last_error: Exception | None = None

        for attempt in range(3):  # 1 original + 2 retries
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                body = resp.json()
                content = body["choices"][0]["message"]["content"]
                return json.loads(content)
            except Exception as exc:
                last_error = exc
                if attempt < 2:  # wait before retry
                    time.sleep(1 + attempt * 2)  # 1 s, then 3 s

        # All attempts exhausted
        raise last_error  # type: ignore[misc]

    # ═══════════════════════════════════════════════════════════
    # Candidate fusion
    # ═══════════════════════════════════════════════════════════

    def _fuse_candidates(self, candidates: list[dict]) -> Answer:
        """Fuse per-page candidate answers via retrieval-score weighting.

        Formula:
            combined_score = retrieval_score × vlm_confidence

        Answers that normalise to the same key have their scores summed
        and evidence pages merged.  The highest-scoring key wins.
        """
        # Group by normalised answer
        merged: dict[str, dict] = {}

        for c in candidates:
            key = self._normalize_answer(c["answer"])
            combined = c["retrieval_score"] * c["vlm_confidence"]

            if key not in merged:
                merged[key] = {
                    "raw_answer": c["answer"],
                    "score": 0.0,
                    "pages": [],
                }
            merged[key]["score"] += combined
            merged[key]["pages"].append(c["page_id"])

        # Guard: degenerate case (shouldn't happen if callers check first)
        total_score = sum(m["score"] for m in merged.values())
        if total_score == 0.0:
            return Answer("Unable to determine", [], 0.0)

        # Pick best
        best_key = max(merged, key=lambda k: merged[k]["score"])
        best = merged[best_key]

        final_conf = best["score"] / max(total_score, 0.001)

        return Answer(
            text=best["raw_answer"],
            evidence_page_ids=best["pages"],
            confidence=round(min(final_conf, 1.0), 4),
        )

    # ═══════════════════════════════════════════════════════════
    # Static utilities
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _encode_image_base64(image_path: str) -> str:
        """Read an image file and return its Base64-encoded string.

        Raises:
            FileNotFoundError: If *image_path* does not exist.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with path.open("rb") as fh:
            return base64.b64encode(fh.read()).decode("utf-8")

    @staticmethod
    def _normalize_answer(text: str) -> str:
        """Normalise answer text for case-insensitive, whitespace-insensitive
        comparison.

        Behaviour is intentionally aligned with ``metrics._normalize`` so
        that the fusion grouping matches the evaluation metric.
        """
        return "".join(text.lower().split()).replace("%", "")
