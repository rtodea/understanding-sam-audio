from contextlib import asynccontextmanager

from fastapi import FastAPI

from .model_registry import get_model
from .ws_handler import router as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up the model before the first request arrives so the first
    # WebSocket client does not time out waiting for the GPU to load.
    get_model()
    yield


app = FastAPI(title="SAM-Audio WebRTC Server", lifespan=lifespan)
app.include_router(ws_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
