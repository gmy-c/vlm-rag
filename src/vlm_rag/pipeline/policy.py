from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
from typing import Any

from .contracts import (
    PageAccessDecision,
    RetrievalHit,
    SensitivityDecision,
)
from .provenance import sha256_file
from ..sensitivity.catalog import SensitivityCatalog


@dataclass(frozen=True, slots=True)
class RedactionRecord:
    page_id: str
    redacted_path: str
    approved: bool
    source_sha256: str | None = None
    redacted_sha256: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RedactionRecord":
        return cls(
            page_id=str(value["page_id"]),
            redacted_path=str(value["redacted_path"]),
            approved=bool(value.get("approved", False)),
            source_sha256=(
                str(value["source_sha256"])
                if value.get("source_sha256")
                else None
            ),
            redacted_sha256=(
                str(value["redacted_sha256"])
                if value.get("redacted_sha256")
                else None
            ),
        )


def load_redaction_manifest(path: Path | None) -> dict[str, RedactionRecord]:
    if path is None:
        return {}
    result: dict[str, RedactionRecord] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            record = RedactionRecord.from_dict(value)
            if record.page_id in result:
                raise ValueError(
                    f"{path}:{line_number}: duplicate page_id={record.page_id}"
                )
            result[record.page_id] = record
    return result


@dataclass(frozen=True, slots=True)
class SensitivityPolicyConfig:
    answer_page_limit: int = 3
    verify_catalog_image_hash: bool = False
    verify_redaction_hashes: bool = True


class SensitivityPolicy:
    """Fail-closed policy: only explicitly safe bytes may leave the process."""

    def __init__(
        self,
        *,
        catalog: SensitivityCatalog,
        data_root: Path,
        redactions: dict[str, RedactionRecord] | None = None,
        config: SensitivityPolicyConfig | None = None,
    ) -> None:
        self.catalog = catalog
        self.data_root = data_root.expanduser().resolve()
        self.redactions = redactions or {}
        self.config = config or SensitivityPolicyConfig()
        if self.config.answer_page_limit < 1:
            raise ValueError("answer_page_limit must be positive")

    def apply(
        self,
        hits: list[RetrievalHit],
    ) -> tuple[list[PageAccessDecision], list[PageAccessDecision]]:
        decisions = [self.decide(hit) for hit in hits]
        selected = [
            decision
            for decision in decisions
            if decision.action in {"allow_original", "allow_redacted"}
        ][: self.config.answer_page_limit]
        return decisions, selected

    def decide(self, hit: RetrievalHit) -> PageAccessDecision:
        entry = self.catalog.get(hit.page_id)
        if entry is None:
            return self._blocked(
                hit,
                "missing",
                "Page is absent from the sensitivity catalog",
            )
        if entry.doc_id != hit.doc_id or entry.image_path != hit.image_path:
            return self._blocked(
                hit,
                "mismatch",
                "Retrieval metadata does not match the sensitivity catalog",
            )
        sensitivity = SensitivityDecision(
            page_id=hit.page_id,
            probability=entry.probability,
            threshold=entry.threshold,
            is_sensitive=entry.is_sensitive,
            catalog_status=entry.status,
            reason=entry.error or entry.status,
        )
        if entry.status != "ok" or entry.is_sensitive is None:
            return PageAccessDecision(
                hit=hit,
                sensitivity=sensitivity,
                action="error",
                selected_image_path=None,
                reason="Sensitivity inference is unavailable; fail closed",
            )
        source_path = self._resolve(entry.image_path)
        if not source_path.is_file():
            return PageAccessDecision(
                hit=hit,
                sensitivity=sensitivity,
                action="error",
                selected_image_path=None,
                reason="Original page image is missing; fail closed",
            )
        if (
            self.config.verify_catalog_image_hash
            and entry.image_sha256
            and sha256_file(source_path) != entry.image_sha256
        ):
            return PageAccessDecision(
                hit=hit,
                sensitivity=sensitivity,
                action="error",
                selected_image_path=None,
                reason="Original page hash differs from the classified bytes",
            )
        if not entry.is_sensitive:
            return PageAccessDecision(
                hit=hit,
                sensitivity=sensitivity,
                action="allow_original",
                selected_image_path=str(source_path),
                reason="Catalog classifies the original page as non-sensitive",
            )
        redaction = self.redactions.get(hit.page_id)
        if redaction is None or not redaction.approved:
            return PageAccessDecision(
                hit=hit,
                sensitivity=sensitivity,
                action="block",
                selected_image_path=None,
                reason="Sensitive page has no approved redacted derivative",
            )
        redacted_path = self._resolve(redaction.redacted_path)
        if not redacted_path.is_file():
            return PageAccessDecision(
                hit=hit,
                sensitivity=sensitivity,
                action="error",
                selected_image_path=None,
                reason="Approved redacted derivative is missing",
            )
        if self.config.verify_redaction_hashes:
            if (
                redaction.source_sha256
                and sha256_file(source_path) != redaction.source_sha256
            ):
                return PageAccessDecision(
                    hit=hit,
                    sensitivity=sensitivity,
                    action="error",
                    selected_image_path=None,
                    reason="Redaction source hash mismatch",
                )
            if (
                redaction.redacted_sha256
                and sha256_file(redacted_path) != redaction.redacted_sha256
            ):
                return PageAccessDecision(
                    hit=hit,
                    sensitivity=sensitivity,
                    action="error",
                    selected_image_path=None,
                    reason="Redacted derivative hash mismatch",
                )
        return PageAccessDecision(
            hit=hit,
            sensitivity=sensitivity,
            action="allow_redacted",
            selected_image_path=str(redacted_path),
            reason="Sensitive original replaced by an approved derivative",
        )

    def _blocked(
        self,
        hit: RetrievalHit,
        status: str,
        reason: str,
    ) -> PageAccessDecision:
        return PageAccessDecision(
            hit=hit,
            sensitivity=SensitivityDecision(
                page_id=hit.page_id,
                probability=None,
                threshold=None,
                is_sensitive=None,
                catalog_status=status,
                reason=reason,
            ),
            action="block",
            selected_image_path=None,
            reason=reason,
        )

    def _resolve(self, relative_path: str) -> Path:
        pure = PurePosixPath(relative_path.replace("\\", "/"))
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"Unsafe page path: {relative_path!r}")
        resolved = self.data_root.joinpath(*pure.parts).resolve()
        try:
            resolved.relative_to(self.data_root)
        except ValueError as exc:
            raise ValueError(
                f"Page path escapes data root: {relative_path!r}"
            ) from exc
        return resolved
