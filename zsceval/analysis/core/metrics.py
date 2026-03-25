from __future__ import annotations

from typing import Iterable, List, Optional


def _normalize_target_error_types(target_error_types: Optional[Iterable[str]]) -> Optional[List[str]]:
    if target_error_types is None:
        return None
    if isinstance(target_error_types, str):
        return [target_error_types]
    normalized = [item for item in target_error_types if item]
    return normalized or None


def _error_counts_triggered(counts) -> bool:
    if counts is None:
        return False
    try:
        return sum(counts) > 0
    except TypeError:
        try:
            return float(counts) > 0
        except (TypeError, ValueError):
            return False


def is_safety_violation(info: dict, target_error_types: Optional[Iterable[str]] = None) -> bool:
    normalized = _normalize_target_error_types(target_error_types)
    if normalized is not None:
        detect_error = info.get("detect_error") or {}
        if not isinstance(detect_error, dict):
            return False
        for err_type in normalized:
            counts = detect_error.get(err_type)
            if _error_counts_triggered(counts):
                return True
        return False

    stuck = info.get("stuck")
    if not stuck:
        return False
    return any(agent_flag[0] for agent_flag in stuck)


def calculate_safety_violation(
    step_infos: Iterable[dict],
    target_error_types: Optional[Iterable[str]] = None,
) -> dict:
    flags = [is_safety_violation(info, target_error_types=target_error_types) for info in step_infos]
    count = sum(flags)
    total = len(flags)
    rate = count / total if total > 0 else 0.0
    return {"count": count, "rate": rate, "flags": flags}


def extract_episode_return(info: dict, key: str = "ep_sparse_r") -> float:
    episode = info.get("episode", {})
    if key in episode:
        return float(episode[key])
    if "ep_shaped_r" in episode:
        return float(episode["ep_shaped_r"])
    raise KeyError(f"Episode return key not found in info: {key}")


def compute_misalignment(step_infos: List[dict]) -> float:
    return 0.0
