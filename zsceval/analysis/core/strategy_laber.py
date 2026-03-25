from __future__ import annotations

from typing import Mapping, Sequence, Tuple


RECIPE_BASE_ID = {
    "TTT": 0,
    "TTO": 3,
    "TOO": 6,
    "OOO": 9,
}

ROLE_OFFSET = {
    "supplier": 0,
    "cook": 1,
    "individual": 2,
}

SUPPLIER_KEYS = (
    "pickup_onion_from_O",
    "pickup_tomato_from_T",
    "pickup_onion_from_X",
    "pickup_tomato_from_X",
    "potting_onion",
    "potting_tomato",
    "PLACEMENT_IN_POT",
)

COOK_KEYS = (
    "pickup_dish_from_D",
    "SOUP_PICKUP",
    "delivery",
    "USEFUL_DISH_PICKUP",
    "USEFUL_SOUP_PICKUP",
)


def normalize_recipe_code(recipe_code: str) -> str:
    code = recipe_code.strip().upper()
    if code not in RECIPE_BASE_ID:
        raise ValueError(f"Unknown recipe code '{recipe_code}'. Expected one of {sorted(RECIPE_BASE_ID)}")
    return code


def normalize_role(role: str) -> str:
    role_name = role.strip().lower()
    if role_name not in ROLE_OFFSET:
        raise ValueError(f"Unknown role '{role}'. Expected one of {sorted(ROLE_OFFSET)}")
    return role_name


def encode_w_id(recipe_code: str, role: str) -> int:
    code = normalize_recipe_code(recipe_code)
    role_name = normalize_role(role)
    return RECIPE_BASE_ID[code] + ROLE_OFFSET[role_name]


def decode_w_id(w_id: int) -> Tuple[str, str]:
    if w_id < 0 or w_id > 11:
        raise ValueError(f"w_id must be in [0, 11], got {w_id}")
    recipe_idx, role_idx = divmod(int(w_id), 3)
    recipe = ("TTT", "TTO", "TOO", "OOO")[recipe_idx]
    role = ("supplier", "cook", "individual")[role_idx]
    return recipe, role


def infer_recipe_code_from_layout(layout_name: str) -> str:
    compact = "".join(ch for ch in layout_name.upper() if ch in {"T", "O"})
    if compact in RECIPE_BASE_ID:
        return compact
    raise ValueError(
        f"Could not infer recipe code from layout '{layout_name}'. "
        f"Use explicit recipe code in {sorted(RECIPE_BASE_ID)}."
    )


def infer_role_from_episode(
    step_records: Sequence[Mapping[str, object]],
    tie_margin: float = 1.25,
) -> str:
    supplier_score = 0.0
    cook_score = 0.0
    for step in step_records:
        shaped_info = step.get("shaped_info", {})
        if not isinstance(shaped_info, Mapping):
            continue
        supplier_score += sum(float(shaped_info.get(k, 0)) for k in SUPPLIER_KEYS)
        cook_score += sum(float(shaped_info.get(k, 0)) for k in COOK_KEYS)

    if supplier_score <= 0 and cook_score <= 0:
        return "individual"
    if supplier_score >= cook_score * tie_margin:
        return "supplier"
    if cook_score >= supplier_score * tie_margin:
        return "cook"
    return "individual"

