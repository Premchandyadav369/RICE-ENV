from __future__ import annotations

from typing import Optional

from .crops import CROP_BASE_PRICE, CROP_PARAMS, CropParams


def get_optimal_crop(
    *,
    soil_type: str,
    rainfall_forecast: float,
    temperature: float,
    scenario: str,
) -> str:
    """
    Deterministic “ground truth” used for grading the crop-selection task.
    The goal is not agriculture-perfect accuracy, but consistent evaluation signal.
    """

    # Scenario shifts effective rainfall / stress risk.
    if scenario == "drought":
        effective_rain = rainfall_forecast * 0.75
        stress_bias = 1.15
    elif scenario == "flood":
        effective_rain = rainfall_forecast * 1.25
        stress_bias = 1.05
    else:
        effective_rain = rainfall_forecast
        stress_bias = 1.0

    # Soil modulates water retention.
    if soil_type == "sandy":
        retention = 0.85
    elif soil_type == "clay":
        retention = 1.15
    else:
        retention = 1.0

    effective_rain *= retention

    # Temperature matters mainly for growth velocity.
    too_cold = temperature < 26
    very_hot = temperature > 36

    # Decision rules:
    # - Rice likes higher moisture + warm temps.
    # - Wheat likes moderate moisture + tolerates moderate heat.
    # - Maize prefers good fertility and moderate heat.
    # - Mustard is drought tolerant and likes cooler weather.
    # - Sugarcane is water intensive and likes heat.

    if scenario == "heatwave":
        if effective_rain > 90:
            return "sugarcane"
        return "wheat"

    if too_cold:
        return "mustard"

    if very_hot:
        if effective_rain > 80:
            return "sugarcane"
        return "wheat"

    if effective_rain >= 110:
        if scenario == "flood" or scenario == "monsoon":
            return "rice"
        return "sugarcane"

    if effective_rain >= 80:
        return "rice"

    if effective_rain >= 50:
        return "maize"

    return "mustard"


def _optimal_resources(
    *,
    crop: str,
    soil_type: str,
    rainfall_forecast: float,
    temperature: float,
    scenario: str,
) -> tuple[float, float, float]:
    """
    Estimated resource budgets for achieving “near-ideal” yield.
    Used to compute efficiency_score (0–1).
    """

    params: CropParams = CROP_PARAMS[crop]

    # Water demand is higher when forecast is low or scenario is drought.
    if scenario == "drought":
        rain_factor = 0.65
    elif scenario == "flood":
        rain_factor = 1.15
    elif scenario == "monsoon":
        rain_factor = 1.4
    elif scenario == "heatwave":
        rain_factor = 0.6
    else:
        rain_factor = 1.0

    rain_deficit = max(0.0, (85.0 - rainfall_forecast * rain_factor) / 85.0)

    # Soil retention changes the irrigation required.
    if soil_type == "sandy":
        retention = 0.8
    elif soil_type == "clay":
        retention = 1.1
    else:
        retention = 1.0

    optimal_water = (25.0 + 55.0 * rain_deficit) * (params.ideal_moisture / 70.0) / retention

    # Fertility demand increases slightly with temperature extremes.
    temp_stress = max(0.0, abs(temperature - 30.0) / 15.0)
    optimal_fertilizer = (18.0 + 35.0 * temp_stress) * (params.ideal_fertility / 70.0)

    # Pesticide demand
    optimal_pesticide = 5.0
    if scenario == "pest_outbreak":
        optimal_pesticide = 15.0
    elif scenario == "monsoon":
        optimal_pesticide = 10.0

    return float(optimal_water), float(optimal_fertilizer), float(optimal_pesticide)


def compute_episode_optima(
    *,
    soil_type: str,
    rainfall_forecast: float,
    temperature: float,
    scenario: str,
    episode_horizon_days: int,
) -> dict[str, float | str | None]:
    """
    Compute optimal_crop and corresponding yield/profit/resource targets.
    These are “for scoring”, not for controlling the simulator.
    """

    optimal_crop = get_optimal_crop(
        soil_type=soil_type,
        rainfall_forecast=rainfall_forecast,
        temperature=temperature,
        scenario=scenario,
    )
    params = CROP_PARAMS[optimal_crop]

    # Yield scales with ideal curves and a mild scenario modifier.
    scenario_yield_mult = 1.0
    if scenario == "flood":
        scenario_yield_mult = 0.98
    elif scenario == "drought":
        scenario_yield_mult = 0.96

    optimal_yield = params.base_yield * scenario_yield_mult

    # Estimate an “optimal” market price around mid-episode.
    mid_day = max(1, episode_horizon_days // 2)
    base_price = CROP_BASE_PRICE[optimal_crop]
    optimal_market_price = base_price + 0.12 * mid_day

    optimal_water, optimal_fertilizer, optimal_pesticide = _optimal_resources(
        crop=optimal_crop,
        soil_type=soil_type,
        rainfall_forecast=rainfall_forecast,
        temperature=temperature,
        scenario=scenario,
    )

    irrigation_cost_per_unit = 0.05
    fertilizer_cost_per_unit = 0.11
    pesticide_cost_per_unit = 0.25
    optimal_profit = (
        optimal_yield * optimal_market_price
        - irrigation_cost_per_unit * optimal_water
        - fertilizer_cost_per_unit * optimal_fertilizer
        - pesticide_cost_per_unit * optimal_pesticide
    )

    return {
        "optimal_crop": optimal_crop,
        "optimal_yield": float(optimal_yield),
        "optimal_profit": float(max(0.0, optimal_profit)),
        "optimal_water": float(max(1.0, optimal_water)),
        "optimal_fertilizer": float(max(1.0, optimal_fertilizer)),
        "optimal_pesticide": float(max(1.0, optimal_pesticide)),
    }


def compute_scores(
    *,
    yield_amount: float,
    profit: float,
    water_used: float,
    fertilizer_used: float,
    pesticide_used: float,
    carbon_footprint: float,
    soil_health: float,
    optimal_yield: float,
    optimal_profit: float,
    optimal_water: float,
    optimal_fertilizer: float,
    optimal_pesticide: float,
) -> dict[str, float]:
    """
    Return continuous scores in [0, 1].
    """

    yield_score = 0.0 if optimal_yield <= 0 else min(max(yield_amount / optimal_yield, 0.0), 1.0)
    profit_score = (
        0.0 if optimal_profit <= 0 else min(max(profit / optimal_profit, 0.0), 1.0)
    )

    # Efficiency: penalize wasted resources relative to “ideal budgets”.
    water_ratio = water_used / max(1.0, optimal_water)
    fert_ratio = fertilizer_used / max(1.0, optimal_fertilizer)
    pest_ratio = pesticide_used / max(1.0, optimal_pesticide)
    waste = (water_ratio + fert_ratio + pest_ratio) / 3.0
    efficiency_score = min(max(1.0 - waste, 0.0), 1.0)

    # Sustainability score: based on carbon footprint and soil health
    # Normalize carbon footprint: assume 500 is a high value
    carbon_penalty = min(carbon_footprint / 500.0, 1.0)
    sustainability_score = 0.5 * soil_health + 0.5 * (1.0 - carbon_penalty)

    return {
        "yield_score": float(yield_score),
        "profit_score": float(profit_score),
        "efficiency_score": float(efficiency_score),
        "sustainability_score": float(sustainability_score),
    }


def compute_final_score(*, task: str, scores: dict[str, float]) -> float:
    """
    Task-specific weighting (easy/medium/hard).
    """
    yield_score = scores["yield_score"]
    profit_score = scores["profit_score"]
    efficiency_score = scores["efficiency_score"]
    sustainability_score = scores["sustainability_score"]

    if task == "easy":
        # Crop selection is “easy” so focus on yield quality and ignore profit.
        return min(max(0.8 * yield_score + 0.2 * sustainability_score, 0.0), 1.0)
    if task == "medium":
        # Yield optimization: emphasize yield + efficiency + sustainability.
        return min(max(0.6 * yield_score + 0.2 * efficiency_score + 0.2 * sustainability_score, 0.0), 1.0)
    # Hard:
    return min(
        max(0.4 * profit_score + 0.2 * yield_score + 0.2 * efficiency_score + 0.2 * sustainability_score, 0.0),
        1.0,
    )

