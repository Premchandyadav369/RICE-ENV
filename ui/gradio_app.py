from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import requests


K2_MODEL = "MBZUAI-IFM/K2-Think-v2"
K2_URL = "https://api.k2think.ai/v1/chat/completions"


def _safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    """
    Tries to extract a single JSON object from model output.
    """
    if not text:
        return None

    text = text.strip()

    # Remove markdown fences if present.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    # Fast path: pure JSON.
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass

    # Otherwise, find the first {...} block.
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _parse_action_from_text(text: str) -> Dict[str, Any]:
    """
    Expected model response formats:
    1) JSON: {"action_type":"irrigate","amount":20}
    2) Function-like: irrigate(20) / plant(rice) / wait(1) / sell(1)
    """
    data = _safe_json_extract(text)
    if data and isinstance(data, dict) and "action_type" in data:
        if "crop" in data and isinstance(data["crop"], str):
            data["crop"] = data["crop"].strip().lower()
        data["action_type"] = str(data["action_type"]).strip()
        data.setdefault("parameters", {})
        return data

    t = text.strip().lower()

    def _call(name: str) -> Optional[List[str]]:
        m = re.search(rf"{name}\((.*?)\)", t)
        if not m:
            return None
        inner = m.group(1).strip()
        return [x.strip() for x in inner.split(",")] if inner else []

    # Function-like parsing
    if "plant" in t:
        args = _call("plant")
        if args:
            return {"action_type": "plant", "crop": args[0].strip().lower(), "parameters": {}}
    if "irrigate" in t:
        args = _call("irrigate")
        if args:
            try:
                return {"action_type": "irrigate", "amount": float(args[0]), "parameters": {}}
            except Exception:
                pass
    if "fertil" in t:
        # fertilize(type, qty) or fertilize(qty)
        args = _call("fertilize")
        if args:
            if len(args) == 1:
                return {"action_type": "fertilize", "fertilizer_type": "NPK", "fertilizer_qty": float(args[0]), "parameters": {}}
            if len(args) >= 2:
                return {"action_type": "fertilize", "fertilizer_type": args[0], "fertilizer_qty": float(args[1]), "parameters": {}}
    if "wait" in t:
        args = _call("wait")
        if args:
            try:
                return {"action_type": "wait", "days": int(float(args[0])), "parameters": {}}
            except Exception:
                pass
    if "sell" in t:
        args = _call("sell")
        if args:
            try:
                return {"action_type": "sell", "sell_quantity": float(args[0]), "parameters": {}}
            except Exception:
                pass
    if "pesticide" in t:
        args = _call("pesticide")
        if args:
            try:
                return {"action_type": "pesticide", "pesticide_qty": float(args[0]), "parameters": {}}
            except Exception:
                pass

    # Fallback: wait 1
    return {"action_type": "wait", "days": 1, "parameters": {}}


async def k2_choose_action(*, observation: Dict[str, Any], task: str) -> Dict[str, Any]:
    """
    Calls K2 Think V2 to produce exactly one next action.
    Uses `K2_API_KEY` from environment (never hardcode keys).
    """
    api_key = os.getenv("K2_API_KEY")
    if not api_key:
        return {"action_type": "wait", "days": 1, "parameters": {}}

    system_prompt = (
        "You are K2 Think V2, an expert agricultural decision agent. "
        "Return ONLY a JSON object for the next action. No extra text.\n\n"
        "Action JSON schema:\n"
        "{"  # keep braces visible in string
        "\"action_type\": \"plant|irrigate|fertilize|wait|sell|pesticide\","
        "\"crop\": \"rice|wheat|maize|mustard|sugarcane\" (only for plant),"
        "\"amount\": number (only for irrigate),"
        "\"fertilizer_type\": string, \"fertilizer_qty\": number (only for fertilize),"
        "\"days\": integer (only for wait),"
        "\"sell_quantity\": number (only for sell),"
        "\"pesticide_qty\": number (only for pesticide),"
        "\"parameters\": {} \n"
        "}\n\n"
        "Rules:\n"
        "1) Only sell when crop_stage is 'harvest'.\n"
        "2) Plant only if crop_planted is null.\n"
        "3) Use irrigation/fertilizer to move soil_moisture/fertility_level toward ideal for the planted crop.\n"
        "4) Optimize according to task: easy=crop selection, medium=yield, hard=profit & efficiency under uncertainty.\n"
    )

    user_prompt = (
        "Current environment state (JSON):\n"
        f"{json.dumps(observation, ensure_ascii=False)}\n\n"
        f"Task: {task}\n"
        "Return ONLY the next action JSON."
    )

    payload = {
        "model": K2_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 180,
        "stream": False,
    }

    def _call() -> Dict[str, Any]:
        resp = requests.post(
            K2_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=45,
        )
        resp.raise_for_status()
        return resp.json()

    data = await asyncio.to_thread(_call)
    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )

    action = _parse_action_from_text(content)
    # Ensure required discriminator exists.
    if "action_type" not in action:
        action = {"action_type": "wait", "days": 1, "parameters": {}}
    action.setdefault("parameters", {})
    return action


def _plot_metrics(history: List[Dict[str, Any]]):
    # history elements are per-step observations
    xs = list(range(len(history)))
    profits = [float(h.get("profit_so_far", 0.0) or 0.0) for h in history]
    pests = [float(h.get("pest_level", 0.0) or 0.0) for h in history]
    moisture = [float(h.get("soil_moisture", 0.0) or 0.0) for h in history]

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=2, cols=1, subplot_titles=("Profit over time", "Pests & Moisture"))

        fig.add_trace(go.Scatter(x=xs, y=profits, mode="lines+markers", name="Profit"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=pests, mode="lines+markers", name="Pests"), row=2, col=1)
        fig.add_trace(go.Scatter(x=xs, y=moisture, mode="lines+markers", name="Moisture"), row=2, col=1)

        fig.update_layout(
            height=600,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        return fig
    except Exception:
        # Fallback: show raw points.
        return {"x": xs, "y": profits}

def _build_history_table(history: List[Dict[str, Any]]):
    if not history:
        return []

    rows = []
    for i, obs in enumerate(history):
        rows.append([
            i,
            obs.get("day", i),
            obs.get("crop_planted") or "None",
            obs.get("crop_stage", "none"),
            f"{obs.get('growth_progress', 0):.2f}",
            f"{obs.get('soil_moisture', 0):.1f}",
            f"{obs.get('pest_level', 0):.1f}",
            f"{obs.get('profit_so_far', 0):.1f}"
        ])
    return rows


def build_gradio(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md) -> gr.Blocks:
    with gr.Blocks(title="RICE-ENV", analytics_enabled=False) as demo:
        gr.Markdown("# 🌾 RICE-ENV\nAgricultural Decision Intelligence (OpenEnv) with K2 Think V2")

        with gr.Row():
            with gr.Column(scale=1):
                task = gr.Dropdown(["easy", "medium", "hard"], value="hard", label="Task")
                scenario = gr.Dropdown(["normal", "drought", "flood", "monsoon", "heatwave", "pest_outbreak"], value="normal", label="Scenario")
                seed = gr.Number(value=42, precision=0, label="Seed (optional)")

                mode = gr.Radio(["AI Mode (K2 Think)", "Human Mode"], value="AI Mode (K2 Think)", label="Mode")

                explain_mode = gr.Checkbox(value=True, label="Explain Mode (show risks/reasoning)")

                reset_btn = gr.Button("Reset Episode", variant="primary")
                step_btn = gr.Button("Run Step")
                auto_btn = gr.Button("Auto Run (8 steps)")

                # Human action controls (always visible; ignored in AI mode)
                action_type = gr.Dropdown(
                    ["plant", "irrigate", "fertilize", "wait", "sell", "pesticide"], value="wait", label="Action type"
                )
                human_crop = gr.Dropdown(["rice", "wheat", "maize", "mustard", "sugarcane"], value="rice", label="Crop (plant)")
                human_amount = gr.Slider(0, 40, value=10, step=1, label="Amount (irrigate)")
                human_fert_type = gr.Textbox(value="NPK", label="Fertilizer type (fertilize)")
                human_fert_qty = gr.Slider(0, 60, value=20, step=1, label="Fertilizer qty (fertilize)")
                human_pesticide_qty = gr.Slider(0, 20, value=5, step=1, label="Pesticide qty (pesticide)")
                human_days = gr.Slider(0, 3, value=1, step=1, label="Days (wait)")
                human_sell_qty = gr.Slider(0, 1.2, value=1.0, step=0.1, label="Sell qty multiplier (sell)")

            with gr.Column(scale=2):
                state_json = gr.JSON(label="Farm State (Observation)")
                ai_json = gr.JSON(label="AI Brain / Explainability")

                logs = gr.Textbox(label="Action Logs", lines=12)
                done_box = gr.Markdown("Status: running")

            with gr.Column(scale=2):
                main_plot = gr.Plot(label="Environmental Dynamics")
                metrics_json = gr.JSON(label="Scores (Current vs Optimal)")
                history_table = gr.Dataframe(
                    headers=["Step", "Day", "Crop", "Stage", "Growth", "Moisture", "Pests", "Profit"],
                    label="Episode History"
                )

        episode_history = gr.State([])  # list of observation dicts

        async def do_reset(task_value, scenario_value, seed_value, explain_toggle):
            reset_kwargs: Dict[str, Any] = {"task": task_value, "scenario": scenario_value}
            if seed_value is not None:
                reset_kwargs["seed"] = int(seed_value)

            res = await web_manager.reset_environment(reset_kwargs)
            obs = res.get("observation") or res.get("current_observation") or {}
            # Logs
            episode_history = [obs]
            metrics = {
                "yield_score": obs.get("yield_score"),
                "profit_score": obs.get("profit_score"),
                "efficiency_score": obs.get("efficiency_score"),
                "sustainability_score": obs.get("sustainability_score"),
                "final_score": obs.get("final_score"),
            }
            logs_text = f"[RESET] task={task_value} scenario={scenario_value}\n"
            explain = obs.get("explainability", {}) if obs else {}

            status = "Status: running"
            return (
                obs,
                explain if explain_toggle else {},
                logs_text,
                status,
                episode_history,
                _plot_metrics(episode_history),
                metrics,
                _build_history_table(episode_history),
            )

        def build_human_action() -> Dict[str, Any]:
            at = action_type.value if hasattr(action_type, "value") else "wait"
            # gradio components don't expose .value in callbacks reliably; we pass values via bindings instead.
            return {"action_type": "wait", "days": 1}

        async def do_step(
            mode_value: str,
            explain_toggle: bool,
            task_value: str,
            scenario_value: str,
            human_action_type: str,
            human_crop_value: str,
            human_amount_value: float,
            human_fert_type_value: str,
            human_fert_qty_value: float,
            human_pesticide_qty_value: float,
            human_days_value: int,
            human_sell_qty_value: float,
            current_history: List[Dict[str, Any]],
            current_logs: str,
        ):
            if not current_history:
                # Auto-reset if user forgot.
                res0 = await web_manager.reset_environment({"task": task_value, "scenario": scenario_value, "seed": 42})
                current_history = [res0.get("observation") or {}]

            last_obs = current_history[-1] if current_history else {}
            done = bool(last_obs.get("done", False))
            if done:
                return (
                    last_obs,
                    (last_obs.get("explainability", {}) if explain_toggle else {}),
                    current_logs + "\n[SKIP] Episode already done.",
                    "Status: done",
                    current_history,
                    _plot_metrics(current_history),
                    {
                        "yield_score": last_obs.get("yield_score"),
                        "profit_score": last_obs.get("profit_score"),
                        "efficiency_score": last_obs.get("efficiency_score"),
                        "sustainability_score": last_obs.get("sustainability_score"),
                        "final_score": last_obs.get("final_score"),
                    },
                    _build_history_table(current_history),
                )

            if mode_value.startswith("AI"):
                action_data = await k2_choose_action(
                    observation=last_obs,
                    task=task_value,
                )
            else:
                # Human mode: use UI fields to create a typed FarmAction payload.
                if human_action_type == "plant":
                    action_data = {"action_type": "plant", "crop": human_crop_value, "parameters": {}}
                elif human_action_type == "irrigate":
                    action_data = {"action_type": "irrigate", "amount": float(human_amount_value), "parameters": {}}
                elif human_action_type == "fertilize":
                    action_data = {"action_type": "fertilize", "fertilizer_type": human_fert_type_value, "fertilizer_qty": float(human_fert_qty_value), "parameters": {}}
                elif human_action_type == "pesticide":
                    action_data = {"action_type": "pesticide", "pesticide_qty": float(human_pesticide_qty_value), "parameters": {}}
                elif human_action_type == "wait":
                    action_data = {"action_type": "wait", "days": int(human_days_value), "parameters": {}}
                else:
                    action_data = {"action_type": "sell", "sell_quantity": float(human_sell_qty_value), "parameters": {}}

            res = await web_manager.step_environment(action_data)
            obs = res.get("observation") or {}
            done_now = bool(obs.get("done", False) or res.get("done", False))

            current_history = list(current_history) + [obs]

            reward = res.get("reward", obs.get("reward", None))
            explain = obs.get("explainability", {}) if obs else {}
            action_reason = explain.get("action_reason") if isinstance(explain, dict) else None

            log_line = f"[STEP] action={action_data} reward={reward} done={done_now}"
            if action_reason and explain_toggle:
                log_line += f" reason={action_reason}"

            new_logs = (current_logs or "") + log_line + "\n"
            metrics = {
                "yield_score": obs.get("yield_score"),
                "profit_score": obs.get("profit_score"),
                "efficiency_score": obs.get("efficiency_score"),
                "sustainability_score": obs.get("sustainability_score"),
                "final_score": obs.get("final_score"),
            }

            return (
                obs,
                explain if explain_toggle else {},
                new_logs,
                "Status: done" if done_now else "Status: running",
                current_history,
                _plot_metrics(current_history),
                metrics,
                _build_history_table(current_history),
            )

        async def do_auto_run(
            mode_value: str,
            explain_toggle: bool,
            task_value: str,
            scenario_value: str,
            current_history: List[Dict[str, Any]],
        ):
            if not current_history:
                res0 = await web_manager.reset_environment({"task": task_value, "scenario": scenario_value, "seed": 42})
                current_history = [res0.get("observation") or {}]

            logs_text = ""
            # Build minimal logs from the observation-explainability only.
            for _ in range(8):
                last_obs = current_history[-1] if current_history else {}
                if bool(last_obs.get("done", False)):
                    break

                if mode_value.startswith("AI"):
                    action_data = await k2_choose_action(observation=last_obs, task=task_value)
                else:
                    # In auto-run, fallback to wait in human mode unless you wire per-step human control.
                    action_data = {"action_type": "wait", "days": 1, "parameters": {}}

                res = await web_manager.step_environment(action_data)
                obs = res.get("observation") or {}
                current_history.append(obs)
                reward = res.get("reward", obs.get("reward", None))
                logs_text += f"[STEP] action={action_data} reward={reward} done={obs.get('done', False)}\n"

            final_obs = current_history[-1] if current_history else {}
            explain = final_obs.get("explainability", {}) if final_obs else {}
            metrics = {
                "yield_score": final_obs.get("yield_score"),
                "profit_score": final_obs.get("profit_score"),
                "efficiency_score": final_obs.get("efficiency_score"),
                "sustainability_score": final_obs.get("sustainability_score"),
                "final_score": final_obs.get("final_score"),
            }
            return (
                final_obs,
                explain if explain_toggle else {},
                logs_text,
                "Status: done",
                current_history,
                _plot_metrics(current_history),
                metrics,
                _build_history_table(current_history),
            )

        reset_btn.click(
            fn=do_reset,
            inputs=[task, scenario, seed, explain_mode],
            outputs=[state_json, ai_json, logs, done_box, episode_history, main_plot, metrics_json, history_table],
        )

        step_btn.click(
            fn=do_step,
            inputs=[
                mode,
                explain_mode,
                task,
                scenario,
                action_type,
                human_crop,
                human_amount,
                human_fert_type,
                human_fert_qty,
                human_pesticide_qty,
                human_days,
                human_sell_qty,
                episode_history,
                logs,
            ],
            outputs=[state_json, ai_json, logs, done_box, episode_history, main_plot, metrics_json, history_table],
        )

        auto_btn.click(
            fn=do_auto_run,
            inputs=[mode, explain_mode, task, scenario, episode_history],
            outputs=[state_json, ai_json, logs, done_box, episode_history, main_plot, metrics_json, history_table],
        )

    return demo

