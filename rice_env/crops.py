from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CropParams:
    base_yield: float
    maturity_days: int
    base_growth_rate: float
    ideal_moisture: float
    ideal_fertility: float
    ideal_water_temperature_sensitivity: float


# These are intentionally simple but “real-feeling” curves.
# You can tweak constants later without breaking OpenEnv contracts.
CROP_PARAMS: dict[str, CropParams] = {
    "rice": CropParams(
        base_yield=950.0,
        maturity_days=4,
        base_growth_rate=0.22,
        ideal_moisture=75.0,
        ideal_fertility=75.0,
        ideal_water_temperature_sensitivity=0.10,
    ),
    "wheat": CropParams(
        base_yield=820.0,
        maturity_days=3,
        base_growth_rate=0.24,
        ideal_moisture=55.0,
        ideal_fertility=65.0,
        ideal_water_temperature_sensitivity=0.07,
    ),
    "maize": CropParams(
        base_yield=900.0,
        maturity_days=4,
        base_growth_rate=0.21,
        ideal_moisture=60.0,
        ideal_fertility=70.0,
        ideal_water_temperature_sensitivity=0.09,
    ),
    "mustard": CropParams(
        base_yield=750.0,
        maturity_days=3,
        base_growth_rate=0.26,
        ideal_moisture=45.0,
        ideal_fertility=60.0,
        ideal_water_temperature_sensitivity=0.06,
    ),
    "sugarcane": CropParams(
        base_yield=1500.0,
        maturity_days=6,
        base_growth_rate=0.15,
        ideal_moisture=80.0,
        ideal_fertility=85.0,
        ideal_water_temperature_sensitivity=0.12,
    ),
}

# Pricing per crop (base). Final daily prices are base + trend + noise.
CROP_BASE_PRICE: dict[str, float] = {
    "rice": 20.0,
    "wheat": 18.0,
    "maize": 19.0,
    "mustard": 25.0,
    "sugarcane": 12.0,
}

