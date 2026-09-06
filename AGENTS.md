# AGENTS.md

Aura Studio — cross-platform AI image generation studio (Apache-2.0). Three localhost processes: React frontend (`src/`), FastAPI diffusion engine (`backend/`), Tauri 2 desktop shell (`src-tauri/`).

## Commands

```bash
./run.bat              # Windows: backend + Vite (NOT Tauri); any key stops both (kills by port)
npm run dev            # Vite on port 1420 — strictPort, tauri devUrl depends on it
npm run backend:dev    # FastAPI engine on 127.0.0.1:8766 (script cds into backend/ itself)
npm run tauri:dev      # Tauri shell + Vite, but does NOT spawn the Python backend

npm run build          # tsc && vite build — the only frontend verification available
npm run tauri:build    # Desktop bundle (NSIS currentUser on Windows)
python scripts/build_sidecar.py   # PyInstaller → src-tauri/binaries/aura-engine-<target-triple>.exe
```

- No test suite, no linter. Verify backend edits from inside `backend/` with
  `python -X utf8 -m py_compile main.py engines/diffusion.py api/models.py preprocessors.py`.
- Live-backend E2E scripts (start `backend:dev` first; scripts default to port **8799** while the backend defaults to **8766** — pass the port as argv[1]):
  `python scripts/e2e_modes_test.py 8766` (txt2img/img2img/inpaint) and
  `python scripts/regression_size_mismatch.py 8766` (input-size≠params-size crash, Z-Image 16-px alignment, mask re-binarization). Artifacts `e2e_*.png` are gitignored.
- No single command starts all three processes. Backend env vars: `AURA_PORT` (8766), `AURA_DATA_DIR` (`~/.aura-studio`, holds `models/cache` + `models/custom`).

## Architecture boundaries

1. **Frontend `src/`** — React 18 + TypeScript + Vite + Tailwind + Zustand + react-router. Four pages (Generate/Models/History/Settings) are thin views over the single store `src/stores/appStore.ts`; ALL backend I/O goes through `src/api/backend.ts`. UI prefs persist to localStorage (`aura-studio-ui`); history to IndexedDB (`src/lib/historyDB.ts`, max 50 entries).
2. **Backend `backend/`** — FastAPI/uvicorn on 127.0.0.1:8766 wrapping HF diffusers. `backend/` is its own import root (absolute imports like `from engines.diffusion import ...`): run `python main.py` from inside `backend/`; `python -m backend.main` from the repo root fails. `main.py` = endpoints/globals; `engines/diffusion.py` = MODEL_REGISTRY, model-type sniffing, `DiffusionEngine`; `api/models.py` = catalog + `ModelManager` (in-memory download tasks); `preprocessors.py` = ControlNet conditioning (canny needs opencv, guarded by `_CV2_AVAILABLE`).
3. **Tauri `src-tauri/`** — minimal Rust wrapper (`get_backend_url`, `start_backend` stub). No frontend file imports `@tauri-apps/*` or calls `invoke()` — it's plain web code in a webview; image save/copy use browser APIs.

## Protocol & conventions

- **Port 8766 is hardcoded in several places**: `src/api/backend.ts` (the one actually used), `run.bat`, `src-tauri/src/lib.rs`. Changing it means touching all of them plus `AURA_PORT`.
- **camelCase ↔ snake_case** between store and wire: `negativePrompt`→`negative_prompt`, `steps`→`num_inference_steps`, `initImage`→`init_image_base64`, result `image_base64`→`imageBase64`. But model/download state stays snake_case end-to-end (`ModelInfo.default_size`, `DownloadProgress.task_id/progress_pct`) — don't "fix" those.
- **Generation** runs over WebSocket `/ws/generate`: one params JSON on open → `{type:"progress", step, total, progress_pct}` frames (first has `status:"starting"`) → `complete`/`cancelled`/`error`. Client cancels with `{type:"cancel"}`. Heavy work runs in `asyncio.to_thread` behind `engine_lock` (asyncio) + `DiffusionEngine._lock` (`threading.RLock`) — don't downgrade either. The REST `POST /generate` exists but the UI never calls it.
- **Modes**: `txt2img | img2img | inpaint | controlnet`, selected by `mode` field with `init_image_base64` / `mask_image_base64` / `control_image_base64`, `strength`, `controlnet_model_id`, `control_type`, `controlnet_conditioning_scale`. Capability discovery via `GET /capabilities`; ControlNet list via `GET /controlnets`. Mode-pipeline assembly (`_assemble_mode_pipeline`) must pass `safety_checker=None, feature_extractor=None` explicitly for sd15/sd21 (the txt2img pipeline was loaded without them, so they never appear in shared components), and the legacy SD inpaint class takes no `width`/`height` kwargs (canvas comes from the resized init image + mask). `ModelManager.delete_model` covers cache + custom + controlnet dirs.
- **Model download**: `POST /models/download` `{model_id, model_type?, file_pattern?}` → `{task_id}`, poll 1 s. `file_pattern` is an fnmatch `allow_patterns` — essential for multi-quant GGUF repos (~80 GB). A dir counts as downloaded only per `_is_complete_snapshot` (partial stays `downloaded:false`); `snapshot_download` resumes on retry.
- **Model load** prefers the app-managed local path. GGUF: two-step load (`Transformer.from_single_file(..., config=base_repo, subfolder="transformer", quantization_config=GGUFQuantizationConfig(...))` then `pipe_cls.from_pretrained(gguf_base_repo(...), transformer=...)`). The transformer class follows the architecture (`_gguf_transformer_class`: z-image → ZImage, `flux.2/flux2` in repo id → Flux2Transformer2DModel, else FluxTransformer2DModel — before this, every FLUX GGUF failed inside the z-image class); the pipeline class is whatever the base repo's `model_index.json` declares (`_pipeline_class_from_repo_config`, e.g. Flux2KleinPipeline). The `subfolder` is mandatory (these repos have no root `config.json`). `gguf_base_repo()` resolution: explicit map (`GGUF_BASE_REPO_MAP` — includes city96/FLUX.1-schnell|dev-gguf → black-forest-labs, whose suffix strip would point at repos that don't exist) → per-family default (`GGUF_FAMILY_BASE`) → `-GGUF` suffix strip (heuristic alone breaks third-party ports). Community GGUF exports with squeezed 1-D pad tokens are handled by `_install_zimage_gguf_pad_token_fix()` — keep that wrapper idempotent via its `_aura_pad_fix` marker. GGUF needs `diffusers >= 0.36` + `gguf` pkg; imports guarded by `_GGUF_AVAILABLE`. A bare LoRA `.safetensors` (no model_index.json) is rejected with a clear "is a LoRA adapter" RuntimeError (`_find_lora_single_file`), never fed to `from_pretrained`.
- **Model families**: sd15 (512), sd21 (768), sdxl (1024), flux (1024), z-image (1024); type inferred from repo-id substrings (FLUX.2/klein models are family `flux` — the engine branches to Flux2* classes internally). Device CUDA-or-CPU only (no MPS); dtype: CPU→fp32, flux/z-image→bf16 (fp16 NaNs), others→fp16. When free VRAM < need × 1.2 → `enable_model_cpu_offload()`. z-image params forced server-side: steps ≤ 8, `guidance_scale = 0.0`; sizes aligned to 16 px (other families 8 px).
- **FLUX GGUF on small VRAM** (`_apply_device_strategy`): for any GGUF `flux`-family model (FLUX.1 and FLUX.2 alike), model-level offload OOMs (the multi-GB text encoder stays resident while the quantized transformer comes up) and `enable_sequential_cpu_offload()` cannot run at all — accelerate's meta-device init hits `KeyError: None` on diffusers' GGUF parameter type. The working scheme keeps the quantized transformer+VAE resident on GPU and wraps EVERY text encoder (FLUX.1 has two: CLIP `text_encoder` + T5-XXL `text_encoder_2`; an unhooked one stays on CPU and crashes on CUDA inputs) in per-module `accelerate.cpu_offload`. FLUX.2 has no img2img/controlnet pipelines in diffusers 0.40 — `supports_mode` reports txt2img only.
- **HF gating breaks some catalog entries**: `stabilityai/stable-diffusion-2-1` and `black-forest-labs/FLUX.1-*` now return 401 for anonymous file downloads (metadata API still readable). Workarounds in place: `GGUF_BASE_REPO_MAP` points `city96/FLUX.1-schnell-gguf` at `unsloth/FLUX.1-schnell` (ungated full rehost of the Apache-2.0 model) for its text-encoder/VAE components, and sd21-family testing used the ungated `nlightcho/stable-diffusion-2-1` mirror. The official gated repos still need an HF token.
- **transformers 5.x on Windows**: `core_model_loading.GLOBAL_WORKERS` (ThreadPoolExecutor tensor materialization) segfaults intermittently mid text-encoder load, killing the whole engine. diffusion.py pins it to 1 at import — single-threaded, stable. Don't remove the patch.
- **Download worker** retries `snapshot_download` up to 8× with growing backoff (network resets here can last minutes; HF retries single requests but not an interrupted snapshot). `HfHubHTTPError` (404 etc.) fails fast. Failed model *loads* now `gc.collect()` + `torch.cuda.empty_cache()` so a half-moved pipeline can't OOM the next load.
- `websockets` (requirements.txt) is required by uvicorn for ANY WebSocket route — without it `/ws/generate` fails "Unsupported upgrade request" while REST still works, masking the breakage.
- Images travel only as base64 PNG in JSON; backend never writes generated images to disk.
- `.gitattributes`: LF everywhere, but `*.bat` must stay CRLF (cmd.exe misparses LF batch files).

## Known gaps (current state — don't fix unprompted)

- Sidecar chain half-built: capability grants `binaries/aura-engine`, but `tauri.conf.json` has `externalBin: []` and Rust never spawns the engine.
- `POST /generate` ignores its `model_id` field (only the loaded engine's model is used).
- Download tasks are in-memory only; restarting the engine orphans a running task's progress entry.

## Docs to read first

- `CLAUDE.md` — deep architecture/protocol notes, but written before the img2img/inpaint/controlnet modes landed; trust the code for anything mode- or ControlNet-related.
- `README.md` — user-facing usage, supported models, API reference, troubleshooting.
