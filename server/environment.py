from __future__ import annotations

import random
from typing import Any, Dict, Optional

from openenv.core.env_server.interfaces import Environment

from models import FarmAction, FarmObservation, FarmState
from rice_env.simulator import (
    EpisodeConfig,
    estimate_profit_amount,
    estimate_yield_amount,
    make_initial_state,
    step_simulation,
)
from rice_env.graders import compute_final_score, compute_scores


class RiceEnvironment(Environment[FarmAction, FarmObservation, FarmState]):
    """
    RICE-ENV simulation environment.
    Designed for deterministic grading + dense reward shaping.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, episode_config: EpisodeConfig | None = None):
        super().__init__()
        self._episode_config = episode_config or EpisodeConfig()
        self._state: Optional[FarmState] = None
        self._rng: Optional[random.Random] = None

    @property
    def state(self) -> FarmState:  # type: ignore[override]
        if self._state is None:
            # OpenEnv will not call state() before reset, but keep it safe.
            self._state = FarmState()
        return self._state

    def _choose_scenario_params(self, *, rng: random.Random, scenario: str) -> Dict[str, Any]:
        soil_type = rng.choice(["loamy", "sandy", "clay"])

        base_rain = 85.0
        if scenario == "drought":
            base_rain = 62.0
        elif scenario == "flood":
            base_rain = 110.0

        rainfall_forecast = float(max(20.0, base_rain + rng.uniform(-20.0, 35.0)))

        base_temp = 30.0
        if scenario == "drought":
            base_temp = 33.0
        elif scenario == "flood":
            base_temp = 28.5

        temperature = float(max(18.0, min(42.0, base_temp + rng.uniform(-3.0, 3.5))))

        return {
            "soil_type": soil_type,
            "rainfall_forecast": rainfall_forecast,
            "temperature": temperature,
        }

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> FarmObservation:
        # OpenEnv passes seed/episode_id; extra kwargs can include task/scenario.
        task = str(kwargs.get("task", "hard"))
        scenario = str(kwargs.get("scenario", "normal"))
        # Clamp to allowed strings; keep validators happy.
        if task not in ("easy", "medium", "hard"):
            task = "hard"
        if scenario not in ("normal", "drought", "flood"):
            scenario = "normal"

        rng_for_params = random.Random(seed if seed is not None else 42)
        params = self._choose_scenario_params(rng=rng_for_params, scenario=scenario)

        state, rng = make_initial_state(
            seed=seed,
            task=task,
            scenario=scenario,
            soil_type=params["soil_type"],
            rainfall_forecast=params["rainfall_forecast"],
            temperature=params["temperature"],
            episode_config=self._episode_config,
        )

        state.episode_id = episode_id
        state.step_count = 0
        self._state = state
        self._rng = rng

        # Initial explainability includes optimal crop for grading reference (useful for UI).
        explain = {
            "reason": "Episode reset.",
            "optimal_crop": state.optimal_crop,
            "episode_params": {
                "soil_type": state.soil_type,
                "rainfall_forecast": state.rainfall_forecast,
                "temperature": state.temperature,
                "scenario": state.scenario,
                "task": state.task,
            },
        }
        self._state.explainability_last = explain

        obs = FarmObservation(
            done=False,
            reward=0.0,
            metadata={},
            day=state.day,
            soil_type=state.soil_type,
            soil_moisture=state.soil_moisture,
            fertility_level=state.fertility_level,
            water_level=state.water_level,
            rainfall_forecast=state.rainfall_forecast,
            temperature=state.temperature,
            market_price=state.market_price,
            crop_planted=state.crop_planted,
            crop_stage=state.crop_stage,
            growth_progress=state.growth_progress,
            profit_so_far=state.profit,
            yield_estimate=0.0,
            yield_score=None,
            profit_score=None,
            efficiency_score=None,
            final_score=None,
            explainability=state.explainability_last,
        )
        return obs

    def step(
        self,
        action: FarmAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> FarmObservation:
        if self._state is None or self._rng is None:
            raise RuntimeError("Environment must be reset before step().")

        state, shaped_reward = step_simulation(
            state=self._state,
            rng=self._rng,
            action=action,
            episode_config=self._episode_config,
        )

        state.step_count += 1
        self._state = state

        done = bool(state.sold or state.day >= self._episode_config.episode_horizon_days)

        # Default reward from shaping; override at termination with task-aware final score.
        reward = float(shaped_reward)

        yield_est = estimate_yield_amount(state)
        profit_est = estimate_profit_amount(
            state=state,
            yield_amount=(state.yield_amount if state.sold else yield_est),
            episode_config=self._episode_config,
        )

        if done:
            # For “hard”: if not sold, profit_score should be 0 (selling decision matters).
            yield_amount_scoring = float(state.yield_amount if state.sold else yield_est)
            profit_scoring = float(state.profit if state.sold else 0.0)

            scores = compute_scores(
                yield_amount=yield_amount_scoring,
                profit=profit_scoring,
                water_used=float(state.water_used),
                fertilizer_used=float(state.fertilizer_used),
                optimal_yield=float(state.optimal_yield),
                optimal_profit=float(state.optimal_profit),
                optimal_water=float(state.optimal_water),
                optimal_fertilizer=float(state.optimal_fertilizer),
            )
            final_score = compute_final_score(task=state.task, scores=scores)
            reward = float(final_score)

            state.yield_amount = yield_amount_scoring
            state.profit = profit_scoring

            state.explainability_last = {
                **(state.explainability_last or {}),
                "termination": True,
                "yield_estimate": float(yield_est),
                "profit_estimate": float(profit_est),
                "scores": {
                    "yield_score": float(scores["yield_score"]),
                    "profit_score": float(scores["profit_score"]),
                    "efficiency_score": float(scores["efficiency_score"]),
                    "final_score": float(final_score),
                },
                "task": state.task,
                "sold": state.sold,
            }

        obs = FarmObservation(
            done=done,
            reward=reward,
            metadata={},
            day=state.day,
            soil_type=state.soil_type,
            soil_moisture=state.soil_moisture,
            fertility_level=state.fertility_level,
            water_level=state.water_level,
            rainfall_forecast=state.rainfall_forecast,
            temperature=state.temperature,
            market_price=state.market_price,
            crop_planted=state.crop_planted,
            crop_stage=state.crop_stage,
            growth_progress=state.growth_progress,
            profit_so_far=float(state.profit),
            yield_estimate=float(yield_est),
            yield_score=state.explainability_last.get("scores", {}).get("yield_score"),
            profit_score=state.explainability_last.get("scores", {}).get("profit_score"),
            efficiency_score=state.explainability_last.get("scores", {}).get("efficiency_score"),
            final_score=state.explainability_last.get("scores", {}).get("final_score"),
            explainability=state.explainability_last,
        )
        return obs

    def get_metadata(self) -> Dict[str, Any]:  # type: ignore[override]
        return {
            "env_name": "rice-env",
            "task_types": ["easy", "medium", "hard"],
            "scenarios": ["normal", "drought", "flood"],
        }

