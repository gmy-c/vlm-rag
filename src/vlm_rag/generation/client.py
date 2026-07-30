from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import mimetypes
from pathlib import Path
import random
import time
from typing import Any

import requests

from .cache import JsonResponseCache
from .fusion import fuse_page_results
from .schema import (
    GenerationAnswer,
    GenerationError,
    GenerationPage,
    PageGenerationResult,
)
from ..pipeline.provenance import sha256_file, sha256_text


PROMPT_VERSION = "secure-page-vqa-v1"
PER_PAGE_PROMPT = """\
You are answering a question from one document page image.
Use only information visibly present on this page.

Question: {query}

Return one JSON object with exactly these fields:
{{"relevant": true, "answer": "...", "evidence_quote": "...", "confidence": 0.0}}

Rules:
- If the page does not contain enough evidence, set relevant=false and use empty strings.
- Preserve exact numbers, currency symbols, dates, and units from the page.
- Never infer an answer from outside knowledge.
- confidence must be between 0 and 1.
"""


@dataclass(frozen=True, slots=True)
class DoubaoClientConfig:
    model: str = "doubao-seed-2-1-pro"
    base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    max_tokens: int = 512
    temperature: float = 0.1
    timeout_seconds: float = 60.0
    max_attempts: int = 3
    minimum_interval_seconds: float = 0.0
    cache_dir: str | None = "outputs/doubao_cache"


class DoubaoVisionClient:
    """Fail-explicit Ark Chat API client for page-level visual QA."""

    def __init__(
        self,
        api_key: str,
        config: DoubaoClientConfig | None = None,
        *,
        session: requests.Session | Any | None = None,
        sleep_fn=time.sleep,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Ark API key is empty")
        self.config = config or DoubaoClientConfig()
        self._api_key = api_key
        self._session = session or requests.Session()
        self._sleep = sleep_fn
        self._last_call_at = 0.0
        self._cache = (
            JsonResponseCache(Path(self.config.cache_dir))
            if self.config.cache_dir
            else None
        )

    def answer(
        self,
        query: str,
        pages: list[GenerationPage],
    ) -> GenerationAnswer:
        results: list[PageGenerationResult] = []
        errors: list[GenerationError] = []
        api_calls = 0
        cache_hits = 0
        for page in pages:
            outcome, calls = self._query_page(query, page)
            api_calls += calls
            if isinstance(outcome, PageGenerationResult):
                results.append(outcome)
                cache_hits += int(outcome.cached)
            else:
                errors.append(outcome)
        text, evidence, confidence = fuse_page_results(results)
        return GenerationAnswer(
            text=text,
            evidence_page_ids=evidence,
            confidence=round(confidence, 4),
            page_results=tuple(results),
            errors=tuple(errors),
            model=self.config.model,
            api_calls=api_calls,
            cache_hits=cache_hits,
        )

    def _query_page(
        self,
        query: str,
        page: GenerationPage,
    ) -> tuple[PageGenerationResult | GenerationError, int]:
        path = Path(page.image_path).expanduser().resolve()
        try:
            image_hash = sha256_file(path)
            image_data_url = self._image_data_url(path)
        except (FileNotFoundError, OSError) as exc:
            return (
                GenerationError(
                    page_id=page.page_id,
                    error_type="image_error",
                    message=str(exc),
                    retryable=False,
                ),
                0,
            )
        prompt = PER_PAGE_PROMPT.format(query=query)
        cache_key = sha256_text(
            "\n".join(
                (
                    PROMPT_VERSION,
                    self.config.model,
                    sha256_text(query),
                    image_hash,
                )
            )
        )
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                try:
                    return self._parse_result(
                        page, cached, cached=True
                    ), 0
                except (KeyError, TypeError, ValueError):
                    pass
        response, error, api_calls = self._request(image_data_url, prompt, page)
        if error is not None:
            return error, api_calls
        assert response is not None
        if self._cache is not None:
            self._cache.put(cache_key, response)
        try:
            return self._parse_result(page, response, cached=False), api_calls
        except (KeyError, TypeError, ValueError) as exc:
            return (
                GenerationError(
                    page_id=page.page_id,
                    error_type="invalid_response",
                    message=str(exc),
                    retryable=False,
                    request_id=_request_id(response),
                ),
                api_calls,
            )

    def _request(
        self,
        image_data_url: str,
        prompt: str,
        page: GenerationPage,
    ) -> tuple[dict[str, Any] | None, GenerationError | None, int]:
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url,
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        calls = 0
        for attempt in range(self.config.max_attempts):
            self._respect_rate_limit()
            calls += 1
            try:
                response = self._session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout_seconds,
                )
            except requests.RequestException as exc:
                if attempt + 1 < self.config.max_attempts:
                    self._sleep(self._backoff(attempt, None))
                    continue
                return None, GenerationError(
                    page_id=page.page_id,
                    error_type="network_error",
                    message=str(exc),
                    retryable=True,
                ), calls
            request_id = response.headers.get("x-request-id")
            if response.status_code >= 400:
                retryable = (
                    response.status_code in {408, 409, 429}
                    or response.status_code >= 500
                )
                if retryable and attempt + 1 < self.config.max_attempts:
                    self._sleep(
                        self._backoff(
                            attempt,
                            response.headers.get("Retry-After"),
                        )
                    )
                    continue
                return None, GenerationError(
                    page_id=page.page_id,
                    error_type=_http_error_type(response.status_code),
                    message=_safe_http_message(response),
                    retryable=retryable,
                    status_code=response.status_code,
                    request_id=request_id,
                ), calls
            try:
                body = response.json()
            except ValueError as exc:
                return None, GenerationError(
                    page_id=page.page_id,
                    error_type="invalid_json",
                    message=str(exc),
                    retryable=False,
                    status_code=response.status_code,
                    request_id=request_id,
                ), calls
            if not isinstance(body, dict):
                return None, GenerationError(
                    page_id=page.page_id,
                    error_type="invalid_response",
                    message="Ark response body is not a JSON object",
                    retryable=False,
                    status_code=response.status_code,
                    request_id=request_id,
                ), calls
            body["_http_request_id"] = request_id
            return body, None, calls
        raise AssertionError("unreachable retry loop")

    def _parse_result(
        self,
        page: GenerationPage,
        body: dict[str, Any],
        *,
        cached: bool,
    ) -> PageGenerationResult:
        content = body["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("choices[0].message.content must be a string")
        value = json.loads(content)
        if not isinstance(value, dict):
            raise TypeError("model output must be a JSON object")
        relevant = value.get("relevant")
        if not isinstance(relevant, bool):
            raise TypeError("model output field 'relevant' must be boolean")
        answer = value.get("answer", "")
        evidence = value.get("evidence_quote", "")
        confidence = float(value.get("confidence", 0.0))
        if not isinstance(answer, str) or not isinstance(evidence, str):
            raise TypeError("answer and evidence_quote must be strings")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        return PageGenerationResult(
            page_id=page.page_id,
            relevant=relevant,
            answer=answer,
            evidence=evidence,
            confidence=confidence,
            retrieval_score=page.retrieval_score,
            source=page.source,
            cached=cached,
            request_id=_request_id(body),
            usage=dict(body.get("usage", {})),
        )

    def _respect_rate_limit(self) -> None:
        interval = self.config.minimum_interval_seconds
        elapsed = time.monotonic() - self._last_call_at
        if interval > 0 and elapsed < interval:
            self._sleep(interval - elapsed)
        self._last_call_at = time.monotonic()

    @staticmethod
    def _backoff(attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(30.0, max(0.0, float(retry_after)))
            except ValueError:
                pass
        return min(30.0, (2**attempt) + random.random())

    @staticmethod
    def _image_data_url(path: Path) -> str:
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"


def _request_id(body: dict[str, Any]) -> str | None:
    value = body.get("_http_request_id") or body.get("id")
    return str(value) if value else None


def _http_error_type(status_code: int) -> str:
    if status_code in {401, 403}:
        return "authentication_error"
    if status_code == 429:
        return "rate_limit_error"
    if status_code >= 500:
        return "server_error"
    return "request_error"


def _safe_http_message(response: Any) -> str:
    try:
        value = response.json()
        if isinstance(value, dict):
            error = value.get("error", value)
            if isinstance(error, dict):
                return str(error.get("message", error.get("code", "Ark API error")))
    except ValueError:
        pass
    return f"Ark API returned HTTP {response.status_code}"
