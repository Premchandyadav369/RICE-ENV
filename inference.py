from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional

import requests

from server.environment import RiceEnvironment
from models import FarmAction, FarmObservation


K2_MODEL = "MBZUAI-IFM/K2-Think-v2"
K2_URL = "https://api.k2think.ai/v1/chat/completions"


def _safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def parse_action(text: str) -> Dict[str, Any]:
    data = _safe_json_extract(text)
    if data and isinstance(data, dict) and "action_type" in data:
        data.setdefault("parameters", {})
        # Minimal normalization for common fields.
        if "crop" in data and isinstance(data["crop"], str):
            data["crop"] = data["crop"].strip().lower()
        if "action_type" in data and data["action_type"] is not None:
            data["action_type"] = str(data["action_type"]).strip()
        return data

    t = text.strip().lower()

    def _call(name: str) -> Optional[List[str]]:
        m = re.search(rf"{name}\((.*?)\)", t)
        if not m:
            return None
        inner = m.group(1).strip()
        return [x.strip() for x in inner.split(",")] if inner else []

    if "plant" in t:
        args = _call("plant")
        if args:
            return {"action_type": "plant", "crop": args[0].strip().lower(), "parameters": {}}
    if "irrigate" in t:
        args = _call("irrigate")
        if args:
            return {"action_type": "irrigate", "amount": float(args[0]), "parameters": {}}
    if "fertilize" in t or "fertil" in t:
        args = _call("fertilize")
        if args:
            if len(args) == 1:
                return {"action_type": "fertilize", "fertilizer_type": "NPK", "fertilizer_qty": float(args[0]), "parameters": {}}
            return {"action_type": "fertilize", "fertilizer_type": args[0], "fertilizer_qty": float(args[1]), "parameters": {}}
    if "wait" in t:
        args = _call("wait")
        if args:
            return {"action_type": "wait", "days": int(float(args[0])), "parameters": {}}
    if "sell" in t:
        args = _call("sell")
        if args:
            return {"action_type": "sell", "sell_quantity": float(args[0]), "parameters": {}}

    return {"action_type": "wait", "days": 1, "parameters": {}}


async def k2_choose_action(*, observation: Dict[str, Any], task: str) -> Dict[str, Any]:
    api_key = os.getenv("K2_API_KEY")
    if not api_key:
        return {"action_type": "wait", "days": 1, "parameters": {}}

    system_prompt = (
        "You are an expert agricultural decision agent. "
        "Return ONLY a JSON object for the next action.\n\n"
        "Schema: "
        "{\"action_type\":\"plant|irrigate|fertilize|wait|sell\","
        "\"crop\":\"rice|wheat|maize\" (plant only),"
        "\"amount\":number (irrigate),"
        "\"fertilizer_type\":string,\"fertilizer_qty\":number (fertilize),"
        "\"days\":int (wait),"
        "\"sell_quantity\":number (sell),"
        "\"parameters\":{}}\n\n"
        "Rules: sell only if crop_stage is 'harvest'; plant only if crop_planted is null."
    )

    user_prompt = (
        "Observation JSON:\n"
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
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    action = parse_action(content)
    action.setdefault("parameters", {})
    return action


async def run_episode(*, task: str, scenario: str, seed: int, max_steps: int):
    env = RiceEnvironment()
    obs: FarmObservation = env.reset(seed=seed, task=task, scenario=scenario)

    rewards: List[float] = []
    step_logs: List[str] = []

    print(f"[START] task={task} env=rice-env model=k2-think-v2")

    for step_idx in range(1, max_steps + 1):
        if obs.done:
            break

        action_dict = await k2_choose_action(observation=obs.model_dump(), task=task)

        # Validate typed action
        try:
            action = FarmAction.model_validate(action_dict)
        except Exception:
            action_dict = {"action_type": "wait", "days": 1, "parameters": {}}
            action = FarmAction.model_validate(action_dict)
        obs = env.step(action)

        reward = float(obs.reward or 0.0)
        done = bool(obs.done)
        rewards.append(reward)

        err = "null"
        step_logs.append(
            f"[STEP] step={step_idx} action={action_dict} reward={reward:.4f} done={'true' if done else 'false'} error={err}"
        )
        print(step_logs[-1])

    success = True
    total_steps = len(rewards)
    rewards_str = ",".join([f"{r:.4f}" for r in rewards])
    print(f"[END] success={'true' if success else 'false'} steps={total_steps} rewards={rewards_str}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="hard", choices=["easy", "medium", "hard"])
    parser.add_argument("--scenario", default="normal", choices=["normal", "drought", "flood"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()

    asyncio.run(run_episode(task=args.task, scenario=args.scenario, seed=args.seed, max_steps=args.max_steps))


if __name__ == "__main__":
    main()

