from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

from models import FarmAction, FarmState
from .crops import CROP_PARAMS, CROP_BASE_PRICE
from .graders import compute_episode_optima, compute_scores, compute_final_score
from .rewards import compute_soil_health, shaped_reward


@dataclass(frozen=True)
class EpisodeConfig:
    max_steps: int = 8
    episode_horizon_days: int = 8
    irrigation_cost_per_unit: float = 0.05
    fertilizer_cost_per_unit: float = 0.11

    # Dynamics coefficients
    evaporation_per_deg: float = 0.35
    rainfall_to_moisture: float = 0.22
    irrigation_to_moisture: float = 0.55

    # Noise levels
    rainfall_noise_sigma: float = 18.0
    temp_noise_sigma: float = 1.6


def _scenario_temperature_multiplier(scenario: str) -> float:
    if scenario == "drought":
        return 1.05
    if scenario == "flood":
        return 0.98
    return 1.0


def _scenario_rain_multiplier(scenario: str) -> float:
    if scenario == "drought":
        return 0.78
    if scenario == "flood":
        return 1.28
    return 1.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def make_initial_state(
    *,
    seed: Optional[int],
    task: str,
    scenario: str,
    soil_type: str,
    rainfall_forecast: float,
    temperature: float,
    episode_config: EpisodeConfig,
) -> tuple[FarmState, random.Random]:
    rng = random.Random(seed if seed is not None else 42)

    market_price = {c: CROP_BASE_PRICE[c] for c in CROP_BASE_PRICE.keys()}
    for c in market_price:
        # Apply a gentle scenario drift to base price
        if scenario == "drought":
            market_price[c] *= 1.03 if c == "wheat" else 0.99
        elif scenario == "flood":
            market_price[c] *= 0.99 if c == "rice" else 1.01

    state = FarmState(
        task=task,  # type: ignore[arg-type]
        scenario=scenario,  # type: ignore[arg-type]
        day=0,
        crop_planted=None,
        crop_stage="none",
        growth_progress=0.0,
        sold=False,
        sold_quantity=0.0,
        soil_type=soil_type,  # type: ignore[arg-type]
        soil_moisture=45.0,
        fertility_level=70.0,
        water_level=60.0,
        fertilizer_stock=80.0,
        rainfall_forecast=float(rainfall_forecast),
        temperature=float(temperature),
        market_price=market_price,
        water_used=0.0,
        fertilizer_used=0.0,
        cumulative_overwater=0.0,
        cumulative_overfertilizer=0.0,
        yield_amount=0.0,
        profit=0.0,
        optimal_crop=None,
        optimal_yield=0.0,
        optimal_profit=0.0,
        optimal_water=0.0,
        optimal_fertilizer=0.0,
        explainability_last={},
    )

    optima = compute_episode_optima(
        soil_type=state.soil_type,
        rainfall_forecast=state.rainfall_forecast,
        temperature=state.temperature,
        scenario=state.scenario,
        episode_horizon_days=episode_config.episode_horizon_days,
    )
    state.optimal_crop = optima["optimal_crop"]  # type: ignore[assignment]
    state.optimal_yield = optima["optimal_yield"]  # type: ignore[assignment]
    state.optimal_profit = optima["optimal_profit"]  # type: ignore[assignment]
    state.optimal_water = optima["optimal_water"]  # type: ignore[assignment]
    state.optimal_fertilizer = optima["optimal_fertilizer"]  # type: ignore[assignment]

    return state, rng


def step_simulation(
    *,
    state: FarmState,
    rng: random.Random,
    action: FarmAction,
    episode_config: EpisodeConfig,
) -> tuple[FarmState, float]:
    """
    Mutates a copy of `state` and returns (next_state, reward).
    """
    # Shallow copy is enough because we reassign dict-like market_price.
    ns = state.model_copy(deep=True)

    prev_growth = ns.growth_progress
    prev_soil_health = (
        compute_soil_health(ns.soil_moisture, ns.fertility_level, ns.crop_planted)
        if ns.crop_planted is not None
        else 0.5
    )

    # Parse action inputs into “desired” resource changes for this day.
    irrigation_amount = 0.0
    fertilizer_qty = 0.0
    fertilizer_type = action.fertilizer_type or "NPK"

    planted_wrong_crop = False

    # 1) Validate and apply discrete actions (resource intents)
    if action.action_type == "plant":
        if ns.crop_planted is None and ns.crop_stage == "none" and action.crop is not None:
            ns.crop_planted = action.crop
            ns.crop_stage = "sowing"
            ns.growth_progress = 0.08
            ns.explainability_last = {"reason": "Crop planted."}
        else:
            # Invalid plant: penalize via explainability (reward computed later).
            ns.explainability_last = {"reason": "Plant action ignored (already planted or invalid)."}

    elif action.action_type == "irrigate":
        if ns.crop_planted is not None and ns.water_level > 0 and action.amount is not None:
            irrigation_amount = float(action.amount)
            irrigation_amount = _clamp(irrigation_amount, 0.0, 40.0)
            irrigation_amount = min(irrigation_amount, ns.water_level)
            ns.explainability_last = {"reason": f"Applying irrigation={irrigation_amount:.1f}."}
        else:
            ns.explainability_last = {"reason": "Irrigate action ignored (no crop or no water)."}

    elif action.action_type == "fertilize":
        if ns.crop_planted is not None and ns.fertilizer_stock > 0 and action.fertilizer_qty is not None:
            fertilizer_qty = float(action.fertilizer_qty)
            fertilizer_qty = _clamp(fertilizer_qty, 0.0, 60.0)
            fertilizer_qty = min(fertilizer_qty, ns.fertilizer_stock)
            ns.explainability_last = {"reason": f"Applying fertilizer {fertilizer_type} qty={fertilizer_qty:.1f}."}
        else:
            ns.explainability_last = {"reason": "Fertilize action ignored (no crop or no fertilizer)."}

    elif action.action_type == "wait":
        # Wait is “do nothing”; days can be >0 but we keep per-step dynamics simple.
        ns.explainability_last = {"reason": "Waiting (no explicit resource action)."}

    elif action.action_type == "sell":
        # Selling is allowed only if crop is in harvest-ready stage.
        pass
    else:
        ns.explainability_last = {"reason": "Unknown action ignored."}

    # Determine if planted crop is wrong for scoring signals.
    if ns.crop_planted is not None and ns.optimal_crop is not None:
        planted_wrong_crop = ns.crop_planted != ns.optimal_crop

    # 2) Advance environment by one day (weather + growth + market)
    ns.day += 1

    # Weather (forecast + randomness)
    rain_noise = rng.gauss(0.0, episode_config.rainfall_noise_sigma)
    temp_noise = rng.gauss(0.0, episode_config.temp_noise_sigma)

    rain_multiplier = _scenario_rain_multiplier(ns.scenario)
    actual_rain = ns.rainfall_forecast * rain_multiplier + rain_noise
    actual_rain = max(0.0, actual_rain)

    temp_multiplier = _scenario_temperature_multiplier(ns.scenario)
    ns.temperature = float(ns.temperature * 0.98 * temp_multiplier + temp_noise)
    ns.temperature = _clamp(ns.temperature, -5.0, 55.0)

    # Soil moisture update
    prev_moisture = ns.soil_moisture
    ns.soil_moisture += actual_rain * episode_config.rainfall_to_moisture
    ns.soil_moisture += irrigation_amount * episode_config.irrigation_to_moisture

    evaporation = episode_config.evaporation_per_deg * max(0.0, ns.temperature) * 0.02
    ns.soil_moisture -= evaporation

    ns.soil_moisture = _clamp(ns.soil_moisture, 0.0, 100.0)

    # Update resources used and remaining
    if irrigation_amount > 0:
        ns.water_level = max(0.0, ns.water_level - irrigation_amount)
        ns.water_used += irrigation_amount
    if fertilizer_qty > 0:
        ns.fertilizer_stock = max(0.0, ns.fertilizer_stock - fertilizer_qty)
        ns.fertilizer_used += fertilizer_qty
        ns.fertility_level = _clamp(ns.fertility_level + fertilizer_qty * 0.55, 0.0, 100.0)

    # Crop growth progression
    prev_stage = ns.crop_stage
    overwater_amount = 0.0
    overfert_amount = 0.0

    if ns.crop_planted is not None and not ns.sold:
        params = CROP_PARAMS[ns.crop_planted]
        # Moisture alignment: ideal moisture yields higher growth.
        moisture_alignment = 1.0 - min(abs(ns.soil_moisture - params.ideal_moisture) / max(1.0, params.ideal_moisture), 1.0)
        fertility_alignment = 1.0 - min(abs(ns.fertility_level - params.ideal_fertility) / max(1.0, params.ideal_fertility), 1.0)

        # Temperature effect: mild Gaussian penalty away from 30C.
        temp_penalty = min(abs(ns.temperature - 30.0) / 18.0, 1.0)
        temperature_factor = 1.0 - 0.55 * temp_penalty

        # Overuse penalties accumulate:
        if ns.soil_moisture > params.ideal_moisture + 12:
            overwater_amount = (ns.soil_moisture - (params.ideal_moisture + 12)) / 100.0
        if ns.fertility_level > params.ideal_fertility + 18:
            overfert_amount = (ns.fertility_level - (params.ideal_fertility + 18)) / 120.0
        ns.cumulative_overwater += overwater_amount
        ns.cumulative_overfertilizer += overfert_amount

        growth_rate = params.base_growth_rate * (0.25 + 0.75 * moisture_alignment) * (0.3 + 0.7 * fertility_alignment) * temperature_factor
        ns.growth_progress = _clamp(ns.growth_progress + growth_rate, 0.0, 1.0)

        # Stage logic
        if ns.day <= 1:
            ns.crop_stage = "sowing"
        else:
            planted_day = ns.day - 1  # approximation
            # Use growth_progress to decide stage.
            if ns.growth_progress < 0.25:
                ns.crop_stage = "sowing"
            elif ns.growth_progress < 0.60:
                ns.crop_stage = "vegetative"
            elif ns.growth_progress < 0.92:
                ns.crop_stage = "mature"
            else:
                ns.crop_stage = "harvest"

    # 3) Selling (economics)
    if action.action_type == "sell":
        if ns.crop_planted is None:
            ns.explainability_last = {"reason": "Sell ignored (nothing planted)."}
        elif ns.crop_stage != "harvest":
            ns.explainability_last = {"reason": f"Sell ignored (crop_stage={ns.crop_stage})."}
        elif not ns.sold:
            # Compute yield and profit at harvest.
            params = CROP_PARAMS[ns.crop_planted]
            soil_health = compute_soil_health(ns.soil_moisture, ns.fertility_level, ns.crop_planted)

            stress_factor = 0.55 + 0.45 * soil_health
            ns.yield_amount = params.base_yield * ns.growth_progress * stress_factor

            # Quantity multiplier: 1.0 sells full yield.
            qty_mult = float(action.sell_quantity) if action.sell_quantity is not None else 1.0
            qty_mult = _clamp(qty_mult, 0.0, 1.2)
            ns.yield_amount *= qty_mult
            ns.sold_quantity = ns.yield_amount

            # Market price evolves; use today's price for selling.
            price_today = ns.market_price.get(ns.crop_planted, CROP_BASE_PRICE[ns.crop_planted])
            ns.profit = (
                ns.yield_amount * price_today
                - episode_config.irrigation_cost_per_unit * ns.water_used
                - episode_config.fertilizer_cost_per_unit * ns.fertilizer_used
            )

            ns.sold = True
            ns.explainability_last = {
                "reason": "Harvest sold.",
                "sold_yield": float(ns.yield_amount),
                "price_today": float(price_today),
                "profit": float(ns.profit),
            }
        else:
            ns.explainability_last = {"reason": "Sell ignored (already sold)."}

    # 4) Update market prices for the day
    # A gentle uptrend with scenario volatility.
    volatility = 0.6 if ns.scenario == "normal" else 0.85
    trend = 0.12 if ns.scenario != "flood" else 0.10
    ns.market_price = {
        crop: (
            max(1.0, base + trend * ns.day + rng.gauss(0.0, volatility))
        )
        for crop, base in CROP_BASE_PRICE.items()
    }

    # 5) Compute shaped reward (dense)
    soil_health_now = (
        compute_soil_health(ns.soil_moisture, ns.fertility_level, ns.crop_planted)
        if ns.crop_planted is not None
        else 0.5
    )

    shaped, reward_info = shaped_reward(
        task=ns.task,
        crop=ns.crop_planted,
        prev_growth_progress=prev_growth,
        new_growth_progress=ns.growth_progress,
        prev_soil_health=prev_soil_health,
        new_soil_health=soil_health_now,
        overwater_amount=overwater_amount,
        overfert_amount=overfert_amount,
        planted_wrong_crop=planted_wrong_crop,
        sold=ns.sold,
    )

    # 6) Terminal scoring (continuous 0..1)
    done = False  # server will set done; we compute final score anyway for info.
    scores = compute_scores(
        yield_amount=float(ns.yield_amount),
        profit=float(max(0.0, ns.profit)),
        water_used=float(ns.water_used),
        fertilizer_used=float(ns.fertilizer_used),
        optimal_yield=float(ns.optimal_yield),
        optimal_profit=float(ns.optimal_profit),
        optimal_water=float(ns.optimal_water),
        optimal_fertilizer=float(ns.optimal_fertilizer),
    )

    final_score = compute_final_score(task=ns.task, scores=scores)

    # If sold, shaped reward tends to be low; give terminal score immediately on sold day.
    reward = shaped
    if ns.sold:
        reward = float(final_score)

    # 7) Build explainability dict
    explain: Dict[str, Any] = {
        "action_reason": ns.explainability_last.get("reason"),
        "soil_moisture_prev": float(prev_moisture),
        "soil_moisture_now": float(ns.soil_moisture),
        "growth_progress_prev": float(prev_growth),
        "growth_progress_now": float(ns.growth_progress),
        "crop_stage_prev": prev_stage,
        "crop_stage_now": ns.crop_stage,
        "water_used": float(ns.water_used),
        "fertilizer_used": float(ns.fertilizer_used),
        "scores": {
            "yield_score": float(scores["yield_score"]),
            "profit_score": float(scores["profit_score"]),
            "efficiency_score": float(scores["efficiency_score"]),
            "final_score": float(final_score),
        },
        "reward_components": reward_info,
        "risks": {},
    }

    # Provide a short risk explanation for judges/UI.
    if ns.crop_planted is not None:
        params = CROP_PARAMS[ns.crop_planted]
        if ns.soil_moisture > params.ideal_moisture + 10:
            explain["risks"]["overwatering_risk"] = "Moisture significantly above ideal."
        if ns.fertility_level > params.ideal_fertility + 15:
            explain["risks"]["overfertilization_risk"] = "Fertility above ideal range."
        if planted_wrong_crop:
            explain["risks"]["wrong_crop_risk"] = "Planted crop differs from optimal crop for this scenario."

    ns.explainability_last = explain

    return ns, float(reward)


def estimate_yield_amount(state: FarmState) -> float:
    """
    Compute a yield estimate purely from growth state (used for grading even if the agent
    does not explicitly sell before episode termination).
    """
    if state.crop_planted is None:
        return 0.0
    params = CROP_PARAMS[state.crop_planted]
    soil_health = compute_soil_health(state.soil_moisture, state.fertility_level, state.crop_planted)
    stress_factor = 0.55 + 0.45 * soil_health
    return float(params.base_yield * state.growth_progress * stress_factor)


def estimate_profit_amount(*, state: FarmState, yield_amount: float, episode_config: EpisodeConfig) -> float:
    """
    Profit estimate uses market price today and subtracts resource costs.
    If the agent didn't sell, callers can decide to set profit_score=0.
    """
    if state.crop_planted is None:
        return 0.0
    price_today = state.market_price.get(state.crop_planted, CROP_BASE_PRICE[state.crop_planted])
    return float(
        yield_amount * price_today
        - episode_config.irrigation_cost_per_unit * state.water_used
        - episode_config.fertilizer_cost_per_unit * state.fertilizer_used
    )

