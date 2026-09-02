"""Aura Studio Backend - FastAPI server for image generation with WebSocket progress."""

import asyncio
import base64
import ctypes
import io
import json
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from engines.diffusion import DiffusionEngine, GenerationCancelled, resolve_model_type
from api.models import ModelManager
from api.catalog import catalog_for_family
import preprocessors
from preprocessors import available_preprocessors

# Global state
engine: Optional[DiffusionEngine] = None
model_manager: Optional[ModelManager] = None
parent_pid: Optional[int] = None
shutting_down = False

# Serializes heavy engine work (load / generate). While a generation runs,
# /models/load answers 409 instead of swapping the model mid-flight.
engine_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    global model_manager
    data_dir = os.environ.get("AURA_DATA_DIR", str(Path.home() / ".aura-studio"))
    model_manager = ModelManager(data_dir=data_dir)
    print(f"[Aura] Data directory: {data_dir}")
    yield
    global engine, shutting_down
    shutting_down = True
    if engine:
        engine.unload()


app = FastAPI(title="Aura Studio Engine", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Models ──────────────────────────────────────────────────


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    # Image modes ("txt2img" | "img2img" | "inpaint" | "controlnet"); images
    # are raw base64 PNG (no data: prefix), matching the result wire format.
    mode: str = "txt2img"
    init_image_base64: str = ""
    mask_image_base64: str = ""
    strength: float = 0.6
    # ControlNet: conditioning input + which weights to steer with.
    control_image_base64: str = ""
    controlnet_model_id: str = ""
    control_type: str = ""          # canny / depth / … (preprocessor hint)
    controlnet_conditioning_scale: float = 1.0


class GenerateResponse(BaseModel):
    image_base64: str
    seed: int
    elapsed: float
    width: int
    height: int


class ModelInfo(BaseModel):
    id: str
    name: str
    type: str
    local: bool
    downloaded: bool
    recommended: bool = False
    description: str = ""
    default_size: int = 512


class DownloadModelRequest(BaseModel):
    model_id: str
    model_type: str = ""
    # Optional fnmatch pattern restricting which files are fetched,
    # e.g. "*Q6_K*.gguf" for one quantization of a GGUF repo.
    file_pattern: str = ""


# ── REST API Routes ──────────────────────────────────────────────────


def _decode_image(data: str):
    """Decode a base64 PNG (no data: prefix) into a PIL RGB image, or None.

    Raises ValueError on garbage so it is safe to call inside the generation
    worker thread (the WS layer turns the error into an error frame).
    """
    if not data:
        return None
    try:
        return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid image payload: {e}") from e


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engine_loaded": engine is not None and engine.loaded,
        "device": engine.device if engine else "none",
        "model_id": engine.model_id if engine else None,
    }


@app.get("/models", response_model=list[ModelInfo])
async def list_models():
    if model_manager is None:
        return []
    return model_manager.get_models()


@app.post("/models/download")
async def download_model(req: DownloadModelRequest):
    """Download a model (catalog or custom). Returns immediately; progress is pushed via polling."""
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    try:
        task_id = model_manager.download_model_async(req.model_id, req.model_type, req.file_pattern)
        return {"status": "started", "model_id": req.model_id, "task_id": task_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/download/{task_id}")
async def get_download_progress(task_id: str):
    """Poll download progress for a task."""
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    progress = model_manager.get_download_progress(task_id)
    if progress is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return progress


@app.post("/models/delete")
async def delete_model(req: DownloadModelRequest):
    """Delete a downloaded model's local files (a loaded engine keeps running from memory)."""
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    deleted = model_manager.delete_model(req.model_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="No local files found for this model")
    return {"status": "deleted", "model_id": req.model_id}


# ── ControlNet ───────────────────────────────────────────────────────


@app.get("/controlnets")
async def list_controlnets():
    """ControlNet weight catalog with download state."""
    if model_manager is None:
        return []
    return model_manager.list_controlnets()


def _resolve_controlnet_source(controlnet_id: str) -> str:
    """Prefer the app-managed download dir for ControlNet weights, falling
    back to the HF repo id (same policy as base-model loading)."""
    if not controlnet_id:
        return ""
    local = model_manager.get_local_path(controlnet_id, kind="controlnet") if model_manager else None
    return local or controlnet_id


def _prepare_control_image(control_b64: str, control_type: str):
    """Decode + preprocess a ControlNet reference image into the conditioning
    image (canny edges / depth map / passthrough). Raises RuntimeError with a
    user-readable message — safe to call inside worker threads."""
    img = _decode_image(control_b64)
    if img is None:
        return None
    try:
        return preprocessors.preprocess(img, control_type)[0]
    except Exception as e:
        raise RuntimeError(f"Preprocessing failed ({control_type or 'passthrough'}): {e}") from e


@app.post("/controlnets/download")
async def download_controlnet(req: DownloadModelRequest):
    """Download ControlNet weights; poll /models/download/{task_id} for progress."""
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    try:
        task_id = model_manager.download_model_async(
            req.model_id, req.model_type, req.file_pattern, kind="controlnet"
        )
        return {"status": "started", "model_id": req.model_id, "task_id": task_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/load")
async def load_model(req: DownloadModelRequest):
    """Load a model into the engine.

    Runs in a worker thread (loading takes tens of seconds); the previous
    engine is unloaded first so VRAM is not held by a stale pipeline.
    """
    global engine
    if engine_lock.locked():
        raise HTTPException(
            status_code=409,
            detail="Engine is busy (loading or generating). Try again shortly.",
        )
    async with engine_lock:
        # 'custom' (the UI label for user-added models) is not a real family —
        # resolve it by sniffing so FLUX/SDXL customs pick the right pipeline.
        mt = resolve_model_type(req.model_id, req.model_type)

        # Prefer the app-managed download dir so what the user downloaded is
        # what actually gets loaded; fall back to the HF hub cache otherwise.
        local_path = model_manager.get_local_path(req.model_id) if model_manager else None

        old_engine = engine
        if old_engine is not None:
            print(f"[Aura] Unloading previous model: {old_engine.model_id}")
            await asyncio.to_thread(old_engine.unload)

        new_engine = DiffusionEngine(model_id=req.model_id, device="cuda")
        try:
            await asyncio.to_thread(new_engine.load_model, req.model_id, mt, local_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        engine = new_engine
        return {
            "status": "loaded",
            "model_id": req.model_id,
            "device": engine.device,
            "model_type": engine.model_type,
        }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Generate an image (REST, no progress updates).

    model_id is informational only — the engine always uses its loaded model.
    """
    if engine is None or not engine.loaded:
        raise HTTPException(status_code=503, detail="Engine not initialized. Load a model first.")
    if engine_lock.locked():
        raise HTTPException(status_code=409, detail="Engine is busy. Try again shortly.")
    try:
        try:
            init_img = _decode_image(req.init_image_base64)
            mask_img = _decode_image(req.mask_image_base64)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        try:
            control_img = _prepare_control_image(req.control_image_base64, req.control_type)
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        cn_source = _resolve_controlnet_source(req.controlnet_model_id)
        async with engine_lock:
            result = await asyncio.to_thread(
                engine.generate,
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                width=req.width,
                height=req.height,
                num_inference_steps=req.num_inference_steps,
                guidance_scale=req.guidance_scale,
                seed=req.seed,
                mode=req.mode,
                init_image=init_img,
                mask_image=mask_img,
                strength=req.strength,
                control_image=control_img,
                controlnet_source=cn_source,
                controlnet_conditioning_scale=req.controlnet_conditioning_scale,
            )
        return GenerateResponse(
            image_base64=result["image_base64"],
            seed=result["seed"],
            elapsed=result["elapsed"],
            width=result["width"],
            height=result["height"],
        )
    except GenerationCancelled:
        raise HTTPException(status_code=499, detail="Generation cancelled")


@app.get("/engine/status")
async def engine_status():
    if engine is None or not engine.loaded:
        return {"loaded": False, "model_id": None, "device": "none"}
    return {
        "loaded": engine.loaded,
        "model_id": engine.model_id,
        "device": engine.device,
        "model_type": engine.model_type,
    }


@app.get("/capabilities")
async def capabilities():
    """Generation modes the CURRENTLY LOADED model supports, so the UI can
    grey out what a family can't do (e.g. Z-Image has no ControlNet)."""
    if engine is None or not engine.loaded:
        return {
            "loaded": False,
            "model_type": None,
            "modes": {},
            "preprocessors": available_preprocessors(),
        }
    modes = {m: engine.supports_mode(m) for m in ("txt2img", "img2img", "inpaint", "controlnet")}
    # A family is only CN-capable if the catalog also has weights for it.
    if modes.get("controlnet") and not catalog_for_family(engine.model_type):
        modes["controlnet"] = False
    return {
        "loaded": True,
        "model_type": engine.model_type,
        "modes": modes,
        "preprocessors": available_preprocessors(),
        "controlnets": model_manager.list_controlnets() if model_manager else [],
    }


# ── WebSocket: Real‑time Generation ─────────────────────────────────


async def _safe_send(websocket: WebSocket, payload: dict) -> None:
    try:
        await websocket.send_json(payload)
    except Exception:
        pass


@app.websocket("/ws/generate")
async def ws_generate(websocket: WebSocket):
    """WebSocket endpoint for generation with real-time progress streaming.

    Client sends: JSON with generation params.
    Server streams: progress updates, then complete/cancelled/error.
    Client may send {"type": "cancel"} at any time to interrupt generation.
    """
    await websocket.accept()

    if engine is None or not engine.loaded:
        await _safe_send(websocket, {"type": "error", "message": "No model loaded"})
        await websocket.close()
        return

    try:
        data = await websocket.receive_json()
    except WebSocketDisconnect:
        return

    steps = int(data.get("num_inference_steps", 20))
    loop = asyncio.get_running_loop()

    async def send_progress(current: int, total: int):
        await _safe_send(websocket, {
            "type": "progress",
            "step": current,
            "total": total,
            "progress_pct": round(current / total * 100, 1),
        })

    def progress_cb(current: int, total: int):
        # Called from the worker thread running the pipeline.
        asyncio.run_coroutine_threadsafe(send_progress(current, total), loop)

    await _safe_send(websocket, {
        "type": "progress",
        "step": 0,
        "total": steps,
        "progress_pct": 0,
        "status": "starting",
    })

    if engine_lock.locked():
        await _safe_send(websocket, {"type": "error", "message": "Engine is busy. Try again shortly."})
        return

    raw_seed = data.get("seed", None)
    try:
        seed = int(float(raw_seed)) if raw_seed is not None else None
    except (TypeError, ValueError):
        seed = None  # garbage seed falls back to random rather than crashing

    mode = data.get("mode") or "txt2img"
    init_b64 = data.get("init_image_base64") or ""
    mask_b64 = data.get("mask_image_base64") or ""
    control_b64 = data.get("control_image_base64") or ""
    try:
        strength = float(data.get("strength", 0.6))
    except (TypeError, ValueError):
        strength = 0.6
    try:
        cn_scale = float(data.get("controlnet_conditioning_scale", 1.0))
    except (TypeError, ValueError):
        cn_scale = 1.0
    cn_source = _resolve_controlnet_source(data.get("controlnet_model_id") or "")
    control_type = data.get("control_type") or ""

    def run_generation():
        # Runs in a worker thread: base64 image decoding (~MBs) and CN
        # preprocessing must not touch the event loop.
        control_img = _prepare_control_image(control_b64, control_type)
        return engine.generate(
            prompt=data.get("prompt", ""),
            negative_prompt=data.get("negative_prompt", ""),
            width=int(data.get("width", 1024)),
            height=int(data.get("height", 1024)),
            num_inference_steps=steps,
            guidance_scale=float(data.get("guidance_scale", 7.5)),
            seed=seed,
            progress_callback=progress_cb,
            mode=mode,
            init_image=_decode_image(init_b64),
            mask_image=_decode_image(mask_b64),
            strength=strength,
            control_image=control_img,
            controlnet_source=cn_source,
            controlnet_conditioning_scale=cn_scale,
        )

    async with engine_lock:
        # The pipeline runs in a worker thread so the event loop stays free:
        # progress frames are delivered in real time and /health keeps answering.
        gen_task = asyncio.create_task(asyncio.to_thread(run_generation))

        # Listen for {"type": "cancel"} while the generation runs.
        while not gen_task.done():
            recv_task = asyncio.create_task(websocket.receive_text())
            done, _pending = await asyncio.wait(
                {gen_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if gen_task in done:
                recv_task.cancel()
                break
            try:
                msg = json.loads(recv_task.result())
                if isinstance(msg, dict) and msg.get("type") == "cancel":
                    print("[Aura] Cancel requested via WebSocket")
                    engine.cancel()
            except WebSocketDisconnect:
                engine.cancel()  # client went away; stop wasting GPU
            except Exception:
                pass  # malformed frame — keep listening

        try:
            result = await gen_task
        except GenerationCancelled:
            await _safe_send(websocket, {"type": "cancelled"})
            return
        except WebSocketDisconnect:
            return
        except Exception as e:
            await _safe_send(websocket, {"type": "error", "message": str(e)})
            return

    await _safe_send(websocket, {
        "type": "complete",
        "image_base64": result["image_base64"],
        "seed": result["seed"],
        "elapsed": result["elapsed"],
        "width": result["width"],
        "height": result["height"],
    })


# ── Parent Process Watch ────────────────────────────────────────────


def _process_alive(pid: int) -> bool:
    """Cross-platform liveness probe that never signals the target process.

    os.kill(pid, 0) is a POSIX idiom; on Windows it would terminate the
    target, so use OpenProcess/GetExitCodeProcess there instead.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False  # process is gone (or inaccessible)
        try:
            exit_code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return exit_code.value == STILL_ACTIVE
            return False
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True  # exists but belongs to another user
    except OSError:
        return False


def watch_parent():
    """Watch for parent process death and exit if the parent is gone."""
    global parent_pid, shutting_down
    if parent_pid is None:
        return
    while not shutting_down:
        time.sleep(2)
        if not _process_alive(parent_pid):
            time.sleep(0.5)
            os._exit(0)


def main():
    global parent_pid

    if "--parent-pid" in sys.argv:
        idx = sys.argv.index("--parent-pid")
        if idx + 1 < len(sys.argv):
            parent_pid = int(sys.argv[idx + 1])

    if parent_pid:
        t = threading.Thread(target=watch_parent, daemon=True)
        t.start()

    port = int(os.environ.get("AURA_PORT", "8766"))

    print(f"[Aura] Starting engine server on port {port}...")
    if parent_pid:
        print(f"[Aura] Watching parent PID: {parent_pid}")

    # ws_max_size: init/mask/control images ride the same socket as base64
    # PNG frames — uvicorn's 16 MiB default rejects large-photo payloads.
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info", ws_max_size=64 * 1024 * 1024)


if __name__ == "__main__":
    main()
