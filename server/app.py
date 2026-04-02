from __future__ import annotations

import os

from openenv.core.env_server import create_app

from models import FarmAction, FarmObservation
from .environment import RiceEnvironment
from ui.gradio_app import build_gradio


def env_factory():
    # EpisodeConfig is static for now; can be extended using env vars.
    return RiceEnvironment()


max_concurrent = int(os.getenv("MAX_CONCURRENT_ENVS", "100"))

# create_app registers OpenEnv endpoints (/health, /ws, /reset, /step, /state) and mounts UI at /web.
app = create_app(
    env_factory,
    FarmAction,
    FarmObservation,
    env_name="rice-env",
    max_concurrent_envs=max_concurrent,
    gradio_builder=build_gradio,
)

