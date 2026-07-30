from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable

import torch


@dataclass(frozen=True, slots=True)
class MemoryLimits:
    target_reserved_gb: float = 26.0
    max_reserved_gb: float = 28.0
    max_device_used_gb: float = 29.0


@dataclass(frozen=True, slots=True)
class MemoryProfile:
    batch_size: int
    status: str
    max_allocated_gb: float
    max_reserved_gb: float
    details: dict[str, Any]


def profile_candidate(
    batch_size: int,
    step: Callable[[int], dict[str, Any]],
) -> MemoryProfile:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for memory profiling")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        details = step(batch_size)
        torch.cuda.synchronize()
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        return MemoryProfile(batch_size, "oom", 0.0, 0.0, {"error": str(exc)})
    return MemoryProfile(
        batch_size=batch_size,
        status="passed",
        max_allocated_gb=torch.cuda.max_memory_allocated() / 2**30,
        max_reserved_gb=torch.cuda.max_memory_reserved() / 2**30,
        details=details,
    )


def choose_largest_safe_profile(
    profiles: list[MemoryProfile],
    limits: MemoryLimits,
) -> MemoryProfile:
    safe = [
        profile
        for profile in profiles
        if profile.status == "passed"
        and profile.max_reserved_gb <= limits.max_reserved_gb
    ]
    if not safe:
        raise RuntimeError("No candidate batch fits the configured memory limit")
    return max(safe, key=lambda profile: profile.batch_size)


def write_memory_profiles(
    path: Path,
    profiles: list[MemoryProfile],
    limits: MemoryLimits,
    selected: MemoryProfile | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "limits": asdict(limits),
                "profiles": [asdict(profile) for profile in profiles],
                "selected_batch_size": (
                    selected.batch_size if selected is not None else None
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
