# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
./run.bat               # Windows: starts backend, polls /health (2 s interval, 60 s cap), then starts Vite; any key stops both (kills by port)
npm run dev             # Vite dev server on port 1420 (strictPort — tauri devUrl depends on it)
npm run backend:dev     # FastAPI engine on 127.0.0.1:8766 (script cds into backend/ itself)
npm run tauri:dev       # Tauri desktop shell (runs its own `npm run dev` via beforeDevCommand)

npm run build           # tsc && vite build → dist/
npm run tauri:build     # Desktop bundle (targets "all"; NSIS currentUser on Windows)
python scripts/build_sidecar.py   # PyInstaller → src-tauri/binaries/aura-engine-<target-triple>.exe
```

No single command starts all three processes: `tauri:dev` starts shell + Vite but **not** the Python backend (nothing spawns it), and `run.bat` starts backend + Vite but not Tauri — run the backend in its own terminal (`npm run backend:dev`, or `python main.py` directly from inside `backend/`). Prefer `scripts/build_sidecar.py` over `npm run build:sidecar` — the latter emits a plain `aura-engine` name Tauri cannot resolve.

There is no test suite and no linter configured. Verify frontend changes with `npm run build`; backend changes with `python -X utf8 -m py_compile main.py engines/diffusion.py api/models.py` from inside `backend/`.

Backend env vars: `AURA_PORT` (default 8766), `AURA_DATA_DIR` (default `~/.aura-studio`, holds `models/cache` and `models/custom`).

## Architecture

Three processes talking over HTTP/WebSocket on localhost:

1. **Frontend** (`src/`) — React 18 + TypeScript + Vite + Tailwind + Zustand. Four pages (Generate / Models / History / Settings) via react-router; all shared state lives in the single store `src/stores/appStore.ts`. Pages are thin views over the store; all backend I/O goes through `src/api/backend.ts`. UI prefs persist to localStorage (`aura-studio-ui`: engine URL + params); generation history persists to IndexedDB (`src/lib/historyDB.ts`, max 50 entries).
2. **Backend engine** (`backend/`) — FastAPI/uvicorn bound to 127.0.0.1:8766, wrapping Hugging Face diffusers pipelines. A global `engine: DiffusionEngine` is created only by `POST /models/load` and swapped atomically after the new model finishes loading; both generation endpoints use whatever it has loaded.
3. **Tauri 2 shell** (`src-tauri/`) — minimal Rust wrapper exposing `get_backend_url` and `start_backend` (a stub that spawns nothing). No JS file imports `@tauri-apps/*` or calls `invoke()` — the frontend is plain web code that happens to run in a webview; image save/copy use browser APIs.

### Frontend ↔ backend protocol

The base URL `http://127.0.0.1:8766` is hardcoded independently in several places (`src/api/backend.ts` — the one actually used, `run.bat`, `src-tauri/src/lib.rs`). The store also holds a user-editable copy applied at startup via `setApiBaseUrl`. Changing the port means touching all of them plus `AURA_PORT`. Generation params/results convert camelCase (store) ↔ snake_case (wire): `negativePrompt`→`negative_prompt`, `steps`→`num_inference_steps`, `image_base64`→`imageBase64` — but model and download state stays snake_case end-to-end (`ModelInfo.default_size`, `DownloadProgress.task_id/progress_pct`); don't "fix" those.

- **Generation** runs over WebSocket `/ws/generate`: client sends one params JSON on open; if no model is loaded the server immediately sends `{type:"error", message:"No model loaded"}`, otherwise it streams `{type:"progress", step, total, progress_pct}` frames (first carries `status:"starting"`) then `{type:"complete", image_base64, seed, elapsed, width, height}`, `{type:"cancelled"}`, or `{type:"error", message}`. The client may send `{type:"cancel"}` mid-generation; the server interrupts the pipeline (raises out of the diffusers step callback) and replies `cancelled`. Generation runs in a worker thread (`asyncio.to_thread`) behind an asyncio lock, so progress frames stream in real time and other endpoints stay responsive; concurrent load/generate attempts get `409` / an "Engine is busy" error frame. The REST `POST /generate` exists but is unused by the UI.
- **Model download**: `POST /models/download` body `{model_id, model_type?, file_pattern?}` → `{task_id}`, polled every 1 s at `GET /models/download/{task_id}` until `completed`/`error`. Progress is byte-level (tqdm interception). `file_pattern` is an fnmatch pattern (e.g. `*Q6_K*.gguf`) passed to `snapshot_download` as `allow_patterns` — essential for multi-quant GGUF repos that total ~80 GB. A directory only counts as downloaded per `_is_complete_snapshot`: any `.gguf` file, or `model_index.json` plus at least one weight file — partial downloads stay listed with `downloaded:false`.
- **Model load**: `POST /models/load` prefers `ModelManager.get_local_path(model_id)` (the app-managed download dir) over the repo id, so what was downloaded is what gets loaded. The UI labels custom models `type:"custom"`; the backend resolves that via `resolve_model_type()` (explicit family wins only if it's in `KNOWN_MODEL_TYPES`, otherwise substring sniff) before choosing a pipeline class. If the local dir holds a `.gguf` file, loading switches to the two-step GGUF path (`_load_gguf_pipeline`): `Transformer.from_single_file(path, config=base_repo, subfolder="transformer", quantization_config=GGUFQuantizationConfig(compute_dtype=…))`, then `pipe_cls.from_pretrained(gguf_base_repo(model_id), transformer=…)`. The `subfolder` is mandatory when passing `config=` as a repo id — these repos have no root `config.json` (only `model_index.json`; component configs live in `transformer/`, `text_encoder/`, …), and omitting it makes diffusers request `<repo>/config.json` and fail. Pipeline-level `from_single_file` does NOT support GGUF, and the base-repo components (text encoder/VAE) are fetched from HF on first load (~8 GB for Z-Image). GGUF needs `diffusers >= 0.36` plus the `gguf` package; imports are guarded behind `_GGUF_AVAILABLE`.
- **GGUF quirks** (both hit in real use): (1) `gguf_base_repo()` resolution order is explicit map → per-family default (`GGUF_FAMILY_BASE`: any z-image GGUF finetune pulls components from `Tongyi-MAI/Z-Image-Turbo`) → `-GGUF` suffix strip; the suffix heuristic alone breaks third-party ports like `lesliemore/z-image-turbo-nsfw-v2-GGUF`, whose stripped repo doesn't exist on HF. (2) Some community GGUF exports store `x_pad_token`/`cap_pad_token` squeezed to 1-D ([dim] instead of diffusers' [1, dim]) and the single-file loader shape-checks strictly — `_install_zimage_gguf_pad_token_fix()` (runs at import) wraps `SINGLE_FILE_LOADABLE_CLASSES["ZImageTransformer2DModel"]["checkpoint_mapping_fn"]` to unsqueeze them; keep that wrapper idempotent via its `_aura_pad_fix` marker attribute.
- **Liveness**: `GET /health` polled every 5 s from `Layout.tsx` (drives the sidebar connection dot and the Active-model indicator; `health.model_id` backfills the engine's loaded model after restarts).
- Images travel only as base64 PNG strings in JSON; the backend never writes generated images to disk.

### Backend internals

- `main.py` — endpoints, globals, lifespan. `engines/diffusion.py` — `MODEL_REGISTRY`, model-type sniffing (`detect_model_type` / `resolve_model_type`), `DiffusionEngine.load_model/generate/unload`. `api/models.py` — catalog, `ModelManager` with in-memory download tasks.
- **Import layout**: `main.py` uses absolute imports (`from engines.diffusion import ...`), so `backend/` itself is the import root — run `python main.py` from inside `backend/`; `python -m backend.main` from the repo root fails.
- torch/diffusers are imported at module level in `engines/diffusion.py` (slow startup, no lazy loading).
- `websockets` (in requirements.txt) is required by uvicorn for ANY WebSocket route — plain `uvicorn` ships without a WS protocol implementation and `/ws/generate` then fails with "Unsupported upgrade request" while every REST endpoint still works, masking the breakage.
- Model families: sd15 (512), sd21 (768), sdxl (1024), flux (1024), z-image (1024); type is inferred from repo-id substrings (`z-image`/`z_image`/`zimage` → z-image; GGUF base repos resolve via `gguf_base_repo`, see the GGUF quirks note under Model load). Device is CUDA-or-CPU only (no MPS): `/models/load` always passes `device="cuda"`, silently clamped to `cpu` inside `DiffusionEngine.__init__` when CUDA is absent. Dtype is chosen per family: CPU → fp32; flux/z-image → bf16 (fp16 produces NaNs on FLUX-class models); others → fp16, with a safetensors/format fallback chain in `_load_pipeline`. When free VRAM < need × 1.2 the pipeline uses `enable_model_cpu_offload()` instead of `.to("cuda")` — z-image's 14 GB estimate makes this effectively always-on for 8–12 GB cards.
- z-image generation params are forced server-side: steps clamped to ≤ 8 and `guidance_scale = 0.0` (distilled turbo model, CFG off) regardless of what the client sent.
- Concurrency: `DiffusionEngine._lock` is a `threading.RLock` (generate's auto-load branch re-enters it); FastAPI serializes heavy work with `engine_lock` (asyncio). Don't downgrade either.

### Known gaps (current state of the system, not bugs to fix unprompted)

- The sidecar chain is half-built: `capabilities/default.json` grants execute on `binaries/aura-engine` (sidecar), but `tauri.conf.json` has `externalBin: []` and Rust never spawns the engine (see also the build-command caveat above).
- `POST /generate` ignores its `model_id` field — only the currently loaded engine's model is used.
- Download tasks live in memory only: restarting the engine orphans a running download's progress entry, though `snapshot_download` resumes into the same directory on retry.
