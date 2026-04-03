# 🌾 RICE-ENV (OpenEnv)
### Agricultural Decision Intelligence with K2-Think V2 Reasoning Agent

[![OpenEnv](https://img.shields.io/badge/Framework-OpenEnv-green.svg)](https://github.com/openenv/openenv)
[![Model](https://img.shields.io/badge/Model-K2--Think--V2-blue.svg)](https://api.k2think.ai)
[![Deployment](https://img.shields.io/badge/Deploy-Hugging%20Face-orange.svg)](https://huggingface.co/spaces)

RICE-ENV is a high-fidelity agricultural simulation environment built on the **OpenEnv** framework. It challenges AI agents to manage a virtual farm under shifting climate conditions, market volatility, and resource constraints. By integrating **K2-Think V2**, RICE-ENV showcases the future of **Autonomous Decision Intelligence** in sustainable agriculture.

---

## 🚀 Why RICE-ENV? (The Future of Farming)
Global food security is one of the most pressing challenges of the 21st century. RICE-ENV simulates the complex trade-offs farmers face daily:
- **Climate Resilience**: Survival in `Drought` or `Flood` scenarios via data-driven irrigation.
- **Sustainable Intensification**: Maximizing yield while minimizing resource waste (Efficiency).
- **Economic Viability**: Timing the market to ensure profitability for small-scale farmers.

RICE-ENV provides a sandbox for training agents that can assist real-world farmers in optimizing resource usage and adapting to a changing climate, shaping a more food-secure future.

---

## ✨ Key Features
- **Dynamic Physics Engine**: Soil moisture, fertility levels, and temperature dynamics influenced by stochastic weather and resource application.
- **Explainable Observations**: Every step includes an `explainability` block detailing risks (e.g., overwatering) and the environment's internal "reasoning."
- **Scenario Diversity**: Pre-configured `Normal`, `Drought`, and `Flood` modes to test agent robustness.
- **Interactive Dashboard**: Custom Gradio UI with real-time Plotly charts for human-AI collaboration.
- **Reward Shaping**: Dense rewards that provide continuous feedback on growth progress and soil health.

---

## 💎 Hackathon Uniqueness
What sets RICE-ENV apart for the **OpenEnv Round 1 Hackathon**:
1. **Deep Reasoning Integration**: Native support for **K2-Think V2**, allowing agents to "think" (Chain-of-Thought) through agricultural risks before executing actions.
2. **Explainability-First Design**: Unlike "black box" environments, RICE-ENV tells you *why* an action succeeded or failed, making it ideal for debugging frontier models.
3. **Multi-Objective Optimization**: Agents must balance Yield, Profit, and Efficiency—mirroring real-world sustainability goals.

---

## 🛠 Environment Specification

### 📥 Observation Space
The agent receives a structured state via the `FarmObservation` model:
- **Weather & Soil**: Day, Soil Type, Moisture (0-100), Fertility (0-100), Temperature, Rainfall Forecast.
- **Crop Progress**: Planted Crop (Rice/Wheat/Maize), Stage (Sowing → Harvest), Growth Progress (0.0-1.0).
- **Economics**: Live Market Prices, Current Profit, Estimated Yield.
- **Scores**: Real-time feedback on Yield, Profit, and Efficiency (0.0-1.0).

### 📤 Action Space
- `plant(crop)`: Sow one of the supported crops.
- `irrigate(amount)`: Apply water (0-40 units).
- `fertilize(type, qty)`: Boost soil fertility levels.
- `wait(days)`: Skip 1-3 days to observe growth.
- `sell(quantity_mult)`: Harvest and sell produce (only at Harvest stage).

---

## 🏆 Tasks & Grading
RICE-ENV implements a tiered difficulty system:

| Task | Objective | Scoring Metric (0.0 - 1.0) |
| :--- | :--- | :--- |
| **Easy** | Crop Selection | `100% Yield Score` |
| **Medium** | Yield Optimization | `70% Yield + 30% Efficiency` |
| **Hard** | Profit Maximization | `50% Profit + 30% Yield + 20% Efficiency` |

*Efficiency is calculated by penalizing resource waste relative to ideal consumption.*

---

## 💻 Technologies Used
- **Backend**: Python 3.10+, FastAPI (OpenEnv Core)
- **AI Engine**: K2-Think V2 (Deep Reasoning LLM)
- **Frontend**: Gradio (Interactive App)
- **Visualization**: Plotly (Live Metrics)
- **Data Integrity**: Pydantic v2 (Strict Typing)
- **Infrastructure**: Docker, Hugging Face Spaces

---

## 📦 Installation & Local Setup

### Using Docker (Recommended)
```bash
# Build the image
docker build -t rice-env .

# Run the environment
docker run -p 8000:8000 -e K2_API_KEY=YOUR_KEY rice-env
```
Access the dashboard at `http://localhost:8000/web`.

### Manual Setup
1. `pip install -r requirements.txt`
2. `export K2_API_KEY="your_api_key"`
3. `uvicorn server.app:app --host 0.0.0.0 --port 8000`

---

## ☁️ End-to-End Deployment Guide

### Hugging Face Spaces (Production)
1. **New Space**: Create a Docker Space on Hugging Face.
2. **Secrets**: Add `K2_API_KEY` in the Space settings.
3. **Deployment**:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USER/YOUR_SPACE
   git push space main
   ```
4. **Verification**: Run `openenv validate --url YOUR_SPACE_URL`.

### Streamlit Cloud / Vercel
While RICE-ENV is optimized for Docker/FastAPI, it can be deployed to any platform supporting containerized Python applications. For Streamlit-like experience, the Gradio UI is automatically served at the `/web` endpoint.

---

## 📊 Baseline Results
To reproduce the baseline scores using K2-Think V2:
```bash
python inference.py --task hard --scenario normal --seed 42
```
*Target Score (Hard Task): 0.82+*

---
*Developed by **Team RED-DRAGON** for the OpenEnv Round 1 Hackathon.*
