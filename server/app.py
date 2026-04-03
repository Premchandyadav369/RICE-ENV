from __future__ import annotations

import os

from fastapi.responses import RedirectResponse
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


@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/web")


def main():
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=True)


if __name__ == "__main__":
    main()
