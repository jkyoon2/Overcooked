from __future__ import annotations

from typing import Mapping, Sequence


SUBTASK_ID = {
    "none": 0,
    "grab_onion": 1,
    "grab_tomato": 2,
    "pot_onion": 3,
    "pot_tomato": 4,
    "put_onion_on_counter": 5,
    "put_tomato_on_counter": 6,
    "grab_dish": 7,
    "put_dish_on_counter": 8,
    "pickup_soup": 9,
    "deliver_soup": 10,
}


def infer_step_subtask_id(step_record: Mapping[str, object]) -> int:
    shaped_info = step_record.get("shaped_info", {})
    if not isinstance(shaped_info, Mapping):
        shaped_info = {}

    event_flags = step_record.get("event_flags", {})
    if not isinstance(event_flags, Mapping):
        event_flags = {}

    if _hit(shaped_info, "delivery") or bool(event_flags.get("soup_delivery", False)):
        return SUBTASK_ID["deliver_soup"]
    if _hit(shaped_info, "SOUP_PICKUP") or bool(event_flags.get("soup_pickup", False)):
        return SUBTASK_ID["pickup_soup"]
    if _hit(shaped_info, "potting_onion") or bool(event_flags.get("potting_onion", False)):
        return SUBTASK_ID["pot_onion"]
    if _hit(shaped_info, "potting_tomato") or bool(event_flags.get("potting_tomato", False)):
        return SUBTASK_ID["pot_tomato"]
    if _hit(shaped_info, "pickup_dish_from_D") or _hit(shaped_info, "pickup_dish_from_X"):
        return SUBTASK_ID["grab_dish"]
    if _hit(shaped_info, "put_dish_on_X"):
        return SUBTASK_ID["put_dish_on_counter"]
    if _hit(shaped_info, "put_onion_on_X"):
        return SUBTASK_ID["put_onion_on_counter"]
    if _hit(shaped_info, "put_tomato_on_X"):
        return SUBTASK_ID["put_tomato_on_counter"]
    if _hit(shaped_info, "pickup_onion_from_O") or _hit(shaped_info, "pickup_onion_from_X"):
        return SUBTASK_ID["grab_onion"]
    if _hit(shaped_info, "pickup_tomato_from_T") or _hit(shaped_info, "pickup_tomato_from_X"):
        return SUBTASK_ID["grab_tomato"]
    return SUBTASK_ID["none"]


def backfill_subtasks(step_records: Sequence[Mapping[str, object]], backfill_horizon: int = 40) -> list[int]:
    """
    Backfill rule:
    - For each event step z!=none, fill preceding contiguous `none` labels with that z.
    - Stop when encountering an already non-none label.
    - Maximum backfill length is `backfill_horizon` steps.
    """
    instant = [infer_step_subtask_id(step) for step in step_records]
    labeled = list(instant)
    if backfill_horizon <= 0:
        return labeled

    for idx, z_id in enumerate(instant):
        if z_id == SUBTASK_ID["none"]:
            continue
        filled = 0
        for prev in range(idx - 1, -1, -1):
            if filled >= backfill_horizon:
                break
            if labeled[prev] != SUBTASK_ID["none"]:
                break
            labeled[prev] = z_id
            filled += 1
    return labeled


def _hit(shaped_info: Mapping[str, object], key: str) -> bool:
    return float(shaped_info.get(key, 0)) > 0
