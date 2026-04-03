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
- **Enhanced Physics Engine**: Advanced modeling of **Soil pH**, **Carbon Footprint**, and **Pest Dynamics**.
- **Indian Agricultural Context**: Native support for Indian crops like **Mustard** and **Sugarcane**, alongside regional scenarios like **Monsoon** and **Heatwaves**.
- **Explainable Observations**: Every step includes an `explainability` block detailing risks (e.g., pest outbreaks, pH imbalance) and the environment's internal "reasoning."
- **Scenario Diversity**: Pre-configured `Normal`, `Drought`, `Flood`, `Monsoon`, `Heatwave`, and `Pest Outbreak` modes.
- **Advanced Interactive Dashboard**: Revamped Gradio UI with multi-metric Plotly charts, an Episode History table, and real-time Sustainability tracking.
- **Sustainability Scoring**: Rewards agents for minimizing carbon footprint and maintaining soil health.

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
- **Weather & Soil**: Day, Soil Type, Moisture, Fertility, Temperature, Rainfall Forecast, **Soil pH**, **Pest Level**.
- **Crop Progress**: Planted Crop (**Rice/Wheat/Maize/Mustard/Sugarcane**), Stage, Growth Progress.
- **Sustainability**: **Carbon Footprint** (CO2e) based on resource usage.
- **Economics**: Live Market Prices, Current Profit, Estimated Yield.
- **Scores**: Real-time feedback on Yield, Profit, Efficiency, and **Sustainability**.

### 📤 Action Space
- `plant(crop)`: Sow one of the supported crops.
- `irrigate(amount)`: Apply water (0-40 units).
- `fertilize(type, qty)`: Boost soil fertility (affects pH).
- `pesticide(qty)`: Manage pest levels (affects carbon footprint).
- `wait(days)`: Observe growth dynamics.
- `sell(quantity_mult)`: Harvest and sell produce.

---

## 🏆 Tasks & Grading
RICE-ENV implements a tiered difficulty system focused on **Sustainable Intensification**:

| Task | Objective | Scoring Metric |
| :--- | :--- | :--- |
| **Easy** | Crop Selection | `80% Yield + 20% Sustainability` |
| **Medium** | Yield Optimization | `60% Yield + 20% Efficiency + 20% Sustainability` |
| **Hard** | Profit Maximization | `40% Profit + 20% Yield + 20% Efficiency + 20% Sustainability` |

*Sustainability is calculated by balancing Soil Health (pH/Fertility) and Carbon Footprint (Resource usage).*

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

### 🥇 Recommended: Hugging Face Spaces
Hugging Face Spaces is the ideal platform for RICE-ENV due to its native support for Docker and Gradio.

1. **Create Space**: Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. **Setup**:
   - **Space Name**: `rice-env`
   - **SDK**: `Docker`
   - **Template**: `Blank`
3. **Configure Secrets**:
   - Navigate to **Settings** > **Variables and secrets**.
   - Add `K2_API_KEY` with your API key from [api.k2think.ai](https://api.k2think.ai).
4. **Deploy**:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/rice-env
   git push space main
   ```
5. **Verify**: Once the build is complete, your dashboard will be live! Verify with:
   ```bash
   openenv validate --url https://huggingface.co/spaces/YOUR_USERNAME/rice-env
   ```

### 🥈 Alternative: Render / Railway
For standard cloud hosting:
1. Connect your GitHub repository to [Render](https://render.com) or [Railway](https://railway.app).
2. Set the **Build Command** to `pip install -r requirements.txt`.
3. Set the **Start Command** to `uvicorn server.app:app --host 0.0.0.0 --port $PORT`.
4. Add the `K2_API_KEY` environment variable.

---

## 📊 Baseline Results
To reproduce the baseline scores using K2-Think V2:
```bash
python inference.py --task hard --scenario normal --seed 42
```
*Target Score (Hard Task): 0.82+*

---
*Developed by **Team RED-DRAGON** for the OpenEnv Round 1 Hackathon.*
