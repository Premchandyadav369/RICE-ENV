from __future__ import annotations

from typing import Any, Dict

from .crops import CROP_PARAMS


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_soil_health(soil_moisture: float, fertility_level: float, crop: str) -> float:
    """
    A smooth proxy for “resource alignment” (used for shaping).
    """
    ideal_m = CROP_PARAMS[crop].ideal_moisture
    ideal_f = CROP_PARAMS[crop].ideal_fertility
    moisture_score = 1.0 - min(abs(soil_moisture - ideal_m) / max(1.0, ideal_m), 1.0)
    fertility_score = 1.0 - min(abs(fertility_level - ideal_f) / max(1.0, ideal_f), 1.0)
    return float(clamp01(0.6 * moisture_score + 0.4 * fertility_score))


def shaped_reward(
    *,
    task: str,
    crop: str | None,
    prev_growth_progress: float,
    new_growth_progress: float,
    prev_soil_health: float,
    new_soil_health: float,
    overwater_amount: float,
    overfert_amount: float,
    planted_wrong_crop: bool,
    sold: bool,
) -> tuple[float, Dict[str, Any]]:
    """
    Dense reward in roughly [0, 1]. We still clamp hard to avoid instability.
    """

    info: Dict[str, Any] = {}

    growth_delta = max(0.0, new_growth_progress - prev_growth_progress)
    growth_reward = 0.65 * growth_delta  # up to ~0.65 per step

    soil_health_delta = new_soil_health - prev_soil_health
    soil_reward = 0.25 * soil_health_delta

    water_penalty = 0.0
    fertilizer_penalty = 0.0
    if overwater_amount > 0:
        water_penalty = min(0.4 * overwater_amount, 0.4)
    if overfert_amount > 0:
        fertilizer_penalty = min(0.45 * overfert_amount, 0.45)

    wrong_crop_penalty = 0.0
    if planted_wrong_crop:
        wrong_crop_penalty = 0.22

    # Encourage not wasting steps after selling; if sold early we keep reward small.
    sold_bonus = 0.0
    if sold:
        sold_bonus = 0.05

    reward = growth_reward + soil_reward - water_penalty - fertilizer_penalty - wrong_crop_penalty + sold_bonus
    reward = float(clamp01(reward))

    info.update(
        {
            "growth_reward": float(growth_reward),
            "soil_reward": float(soil_reward),
            "water_penalty": float(water_penalty),
            "fertilizer_penalty": float(fertilizer_penalty),
            "wrong_crop_penalty": float(wrong_crop_penalty),
            "sold_bonus": float(sold_bonus),
        }
    )
    return reward, info

