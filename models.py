from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


CropType = Literal["rice", "wheat", "maize", "mustard", "sugarcane"]
SoilType = Literal["loamy", "sandy", "clay"]
CropStage = Literal["none", "sowing", "vegetative", "mature", "harvest"]
TaskType = Literal["easy", "medium", "hard"]
ScenarioType = Literal["normal", "drought", "flood", "monsoon", "heatwave", "pest_outbreak"]


class FarmAction(Action):
    """
    Single structured action. OpenEnv uses this model for type-safe validation.
    """

    action_type: Literal["plant", "irrigate", "fertilize", "wait", "sell", "pesticide"] = Field(
        ...,
        description="Action category.",
    )

    # plant
    crop: Optional[CropType] = Field(None, description="Crop to plant (plant action).")

    # irrigate
    amount: Optional[float] = Field(
        None, ge=0.0, description="Irrigation amount (irrigate action)."
    )

    # fertilize
    fertilizer_type: Optional[str] = Field(
        None, description="Fertilizer type label (fertilize action)."
    )
    fertilizer_qty: Optional[float] = Field(
        None, ge=0.0, description="Fertilizer quantity (fertilize action)."
    )

    # wait
    days: Optional[int] = Field(
        None, ge=0, le=3, description="How many days to wait (wait action)."
    )

    # pesticide
    pesticide_qty: Optional[float] = Field(
        None, ge=0.0, description="Pesticide quantity (pesticide action)."
    )

    # sell
    sell_quantity: Optional[float] = Field(
        None,
        ge=0.0,
        description="How much produce to sell (sell action). Interpreted as a multiplier.",
    )

    # Optional metadata for explainability or debugging
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form parameters for advanced agents.",
    )


class FarmObservation(Observation):
    """
    Public observation returned to agents (OpenEnv encodes it into JSON).
    """

    # Environment & weather
    day: int = Field(..., ge=0)
    soil_type: SoilType
    soil_moisture: float = Field(..., ge=0.0, le=100.0)
    fertility_level: float = Field(..., ge=0.0, le=100.0)
    water_level: float = Field(..., ge=0.0)
    rainfall_forecast: float = Field(..., ge=0.0)
    temperature: float = Field(..., ge=-10.0, le=60.0)
    market_price: Dict[str, float] = Field(..., description="Price per crop for today.")
    pest_level: float = Field(..., ge=0.0, le=100.0)
    soil_ph: float = Field(..., ge=0.0, le=14.0)

    # Crop state
    crop_planted: Optional[CropType] = Field(None, description="Currently planted crop.")
    crop_stage: CropStage
    growth_progress: float = Field(..., ge=0.0, le=1.0)

    # Economic state
    profit_so_far: float
    yield_estimate: float
    carbon_footprint: float

    # Per-episode scores (computed on termination)
    yield_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    profit_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    efficiency_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    sustainability_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    final_score: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Explainability: what the environment believes just happened.
    explainability: Dict[str, Any] = Field(default_factory=dict)


class FarmState(State):
    """
    Internal episode state (separate from observations).
    """

    task: TaskType = "hard"
    scenario: ScenarioType = "normal"

    # Episode progress
    day: int = 0

    # Planting / selling
    crop_planted: Optional[CropType] = None
    crop_stage: CropStage = "none"
    growth_progress: float = 0.0
    sold: bool = False
    sold_quantity: float = 0.0

    # Environment
    soil_type: SoilType = "loamy"
    soil_moisture: float = 45.0
    fertility_level: float = 70.0
    water_level: float = 60.0
    fertilizer_stock: float = 80.0
    pesticide_stock: float = 50.0
    rainfall_forecast: float = 80.0
    temperature: float = 30.0
    pest_level: float = 5.0
    soil_ph: float = 6.5
    market_price: Dict[str, float] = Field(default_factory=dict)

    # Resources used
    water_used: float = 0.0
    fertilizer_used: float = 0.0
    pesticide_used: float = 0.0
    carbon_footprint: float = 0.0

    # Stress accounting (for shaping + explanation)
    cumulative_overwater: float = 0.0
    cumulative_overfertilizer: float = 0.0
    cumulative_pest_damage: float = 0.0

    # Rewards / scoring inputs
    yield_amount: float = 0.0
    profit: float = 0.0

    # Optimal targets for scoring
    optimal_crop: Optional[CropType] = None
    optimal_yield: float = 0.0
    optimal_profit: float = 0.0
    optimal_water: float = 0.0
    optimal_fertilizer: float = 0.0
    optimal_pesticide: float = 5.0

    explainability_last: Dict[str, Any] = Field(default_factory=dict)

