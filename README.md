# 🌾 RICE-ENV (OpenEnv)
Agricultural Decision Intelligence Environment with K2 Think V2 agent integration.

## What it simulates
- Crop selection based on soil + rainfall forecast
- Irrigation timing + water balance
- Fertilizer usage + soil alignment
- Harvest selling decision tied to market price dynamics
- Climate scenarios (normal / drought / flood)

## OpenEnv endpoints
The OpenEnv server exposes (under the hood via `openenv-core`):
- `GET /health`
- `WS /ws`
- `POST /reset`
- `POST /step`
- `GET /state`
- UI at `/web` (Gradio)

## Environment tasks (reward shaping)
- `easy`: crop selection → yield-alignment score
- `medium`: yield optimization → yield + efficiency score
- `hard`: profit maximization under uncertainty → weighted profit/yield/efficiency

Final score (hard):
`0.5 * profit + 0.3 * yield + 0.2 * efficiency`, all in `[0, 1]`

## K2 Think V2 integration
Your server/UI and `inference.py` call K2 Think V2 via:
- `K2_API_KEY` (environment variable)

Do not commit API keys.

## Local run (Docker)
```bash
docker build -t rice-env ./rice-env
docker run -p 8000:8000 -e K2_API_KEY=YOUR_KEY rice-env
```

Open UI:
- http://localhost:8000/web

## Baseline agent
Run:
```bash
docker run -e K2_API_KEY=YOUR_KEY rice-env python inference.py --task hard --scenario normal --seed 42
```

It prints:
- `[START] ...`
- `[STEP] ...`
- `[END] ...`

