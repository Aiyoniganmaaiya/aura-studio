# Aura Studio

> **PREVIEW — v0.2.0**
> All five model families and all four generation modes are live and verified on real hardware. Desktop packaging is the last piece still landing — until then the app runs in your browser, exactly like the packaged version will.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Status: Preview](https://img.shields.io/badge/status-preview-orange)](#preview)

Cross-platform AI Image Generation Studio — Z-Image, FLUX, SDXL and Stable Diffusion running entirely on your own machine. No accounts, no API keys, no uploads: your prompts never leave your computer.

## Features

- **Four generation modes** — text-to-image, image-to-image, inpainting with a brush-painted mask, and ControlNet-guided generation, all behind one tab bar.
- **Five model families, auto-detected** — SD 1.5 / SD 2.1 / SDXL / FLUX / Z-Image. Drop in any Hugging Face `org/name` and the engine picks the right pipeline.
- **GGUF quantized models** — run Z-Image Turbo, FLUX.1 Schnell or FLUX.2 Klein from a single 5–7 GB file instead of a 20+ GB checkpoint. Verified working on an 8 GB laptop GPU.
- **ControlNet with automatic preprocessing** — canny edges and depth maps are computed locally from your reference photo before they reach the model.
- **Smart VRAM management** — the engine measures free VRAM per model and automatically chooses between full-GPU residency, component offload, or a hybrid GGUF scheme (quantized transformer resident, text encoders streamed layer by layer).
- **Resumable, retry-hardened downloads** — byte-level progress; interrupted snapshots resume instead of restarting, surviving flaky networks.
- **Local & private** — generation runs through a local Python engine; nothing is sent to the cloud (model weights are fetched from Hugging Face once, then cached).
- **Real-time progress & real cancellation** — a live progress ring streams each denoising step over WebSocket; Cancel interrupts the pipeline mid-run and frees the GPU immediately.
- **Persistent history** — your last 50 generations survive restarts (IndexedDB), with one-click re-open, download or delete.
- **Reproducible seeds** — fix a seed to iterate on a prompt; every result records the seed it used.

## Requirements

| | Minimum | Recommended |
|---|---|---|
| OS | Windows 10/11, macOS, Linux | Windows 11 |
| Python | 3.10 – 3.13 | 3.13 (dev-tested) |
| Node.js | 18+ | 20 LTS |
| GPU | — (CPU mode works) | NVIDIA with ≥ 8 GB VRAM |
| Disk | 2 GB free | 20+ GB (models are 5–24 GB each) |

> **GPU users:** install the CUDA build of PyTorch for 10–50× faster generation — see [Troubleshooting](#troubleshooting).

## Quick Start

### One command (Windows)

```bat
run.bat
```

This starts the Python engine, waits for it to become healthy, then starts the UI. Press any key in its window to stop both.

### Manual (any OS)

```bash
# Terminal 1 — inference engine (must run from backend/)
cd backend
pip install -r requirements.txt
python main.py

# Terminal 2 — web UI
npm install
npm run dev
```

Open **http://localhost:1420**. The sidebar dot turns green when the UI reaches the engine at `http://127.0.0.1:8766`.

### First generation

1. Open **Models → Download Custom Model**, enter `unsloth/Z-Image-Turbo-GGUF` with file filter `*Q6_K*.gguf` (~6 GB), and download — or pick any catalog model.
2. When the download finishes, click **Load** — the engine fetches the base components (~8 GB, one-time) and then loads in seconds.
3. Go to **Generate**, type a prompt, press `Ctrl+Enter`.

## Usage Guide

### Generate

The main workspace: prompt panel on the right, live canvas on the left. Mode tabs at the top switch what you feed the model:

| Mode | What you provide | Good for |
|---|---|---|
| **Text** | A prompt (plus optional negative prompt) | Everything from scratch |
| **Image** | A reference photo + strength slider | Restyling / re-rendering an existing image |
| **Inpaint** | An image + a mask you paint on the canvas | Changing one region, keeping the rest pixel-identical |
| **ControlNet** | A reference photo + a ControlNet weight | Locking composition to edges, depth or pose |

The tabs that need an input image unlock as soon as you drop one in; modes a model can't run are disabled automatically (e.g. ControlNet hides for Z-Image, image modes hide for FLUX.2).

Other controls:

| Control | What it does |
|---|---|
| Negative prompt | Things to avoid (`blurry, low quality…`). Most effective with SD 1.5 / SDXL. |
| Size presets | Common resolutions; SD 1.5 is natively 512², SD 2.1 768², SDXL/FLUX/Z-Image prefer 1024². Sizes snap to each family's latent alignment automatically. |
| Steps | Denoising iterations. Z-Image is distilled and capped at 8; FLUX Schnell at 4. |
| Guidance scale | How strictly the image follows the prompt (CFG). Hidden for CFG-distilled models (Schnell, Z-Image Turbo). |
| Seed | Fix it to reproduce an image exactly. Random mode shows the seed used under the result. |
| `Ctrl+Enter` | Generate from anywhere on the page. |
| Cancel | Interrupts the run on the GPU — not just the connection. |

Every finished image offers **Download** (named after your prompt + seed) and **Copy** (straight to clipboard where the WebView allows it), plus **Inpaint** and **As Reference** shortcuts that feed the result straight into the next run.

### Models

Curated catalog plus anything from Hugging Face:

- **Download** runs in the background with true byte-level progress and resumes after interruptions.
- **Load** swaps the active model; the old one is unloaded first, so VRAM doesn't stack up. The button shows a spinner while weights stream into memory.
- **Trash icon** deletes a downloaded model's files to reclaim disk space (re-downloadable anytime).
- **ControlNet weights** have their own download action and show up in the ControlNet tab once ready.
- **Custom model box** accepts any diffusers-compatible repo ID, e.g. `prompthero/openjourney-v4`. For multi-file repos (GGUF quants, fp8 variants) add a **file filter** — see [GGUF models](#gguf-models-quantized).

Loading a model applies sensible defaults for its family (size, steps, guidance), so you don't have to know them. **Capabilities** (which modes the loaded model supports, which preprocessors are available) are served by the engine at `GET /capabilities` and drive the UI.

### History

Everything you generate is saved locally (up to 50 images) and survives restarts. Click a thumbnail to reopen it full-size with all its parameters restored; hover for delete; **Clear all** wipes the store.

### Settings

- **Engine URL** — change where the UI looks for the backend (default `http://127.0.0.1:8766`); useful if you moved the engine to another port or machine.
- **Engine status** — live view of connection, loaded model, compute device (GPU/CPU).
- **Clear history** — same as History page.

## Supported Models

Everything below has been load- and generation-verified on real hardware (RTX 4060 Laptop, 8 GB VRAM). GGUF first-loads fetch the unquantized base components once and cache them for all quants.

| Model | Family | Download | Notes |
|---|---|---|---|
| Z-Image Turbo GGUF ⭐ | z-image | Q6_K ≈ 6 GB + ~8 GB components* | 8 steps, CFG off; the lightest path to modern quality |
| FLUX.1 Schnell GGUF | FLUX.1 | Q3_K_S ≈ 5 GB + ~10 GB components* | 4 steps; text-to-image |
| FLUX.2 Klein GGUF | FLUX.2 | Q4_K_M ≈ 5.6 GB + ~16 GB components* | Text-to-image; Mistral-based text encoder |
| SDXL 1.0 | SDXL | ≈ 7 GB (fp16 fetched on load) | All four modes incl. ControlNet |
| RealVisXL V4.0 | SDXL | ≈ 7 GB | Photorealistic SDXL finetune |
| Stable Diffusion 2.1 | SD 2.1 | ≈ 5 GB | 768² native |
| Stable Diffusion 1.5 | SD 1.5 | ≈ 4 GB | Lightest option, 512² native, ControlNet-ready |
| ControlNet weights (canny/depth/openpose) | add-on | 1.4–2.5 GB each | Auto-preprocessed conditioning images |
| Any HF diffusers model | auto-detected | varies | Paste `org/name` in Models |

\* One-time; shared across quants of the same base model.

**A note on gated repositories.** Some upstream repos (`stabilityai/stable-diffusion-2-1`, `black-forest-labs/FLUX.1-*`) require accepting a license or signing in before file downloads. The engine maps known GGUF repos to ungated component mirrors automatically (e.g. city96's FLUX.1 GGUF pulls its text encoder/VAE from unsloth's Apache-2.0 rehost). For the official repos themselves, run `hf auth login` once — the engine then picks up your token.

### GGUF models (quantized)

In **Models → Download Custom Model**, enter the repo ID and a *file filter*:

```
Model ID:    unsloth/Z-Image-Turbo-GGUF
File filter: *Q6_K*.gguf
```

Multi-quant repos ship every quantization (some total ~80 GB) — the filter downloads just your pick. The engine loads the quantized transformer from that file and pulls the remaining components (text encoder, VAE, scheduler) from the base repo on first load.

Three things worth knowing:

- **First load downloads extra GB** on top of the GGUF file (the base-repo components). This is one-time and shared across all quants of that base.
- **8 GB cards run FLUX/Z-Image GGUFs via a hybrid offload scheme** — the quantized transformer stays resident on the GPU (it's small when quantized) while the multi-GB text encoders stream through layer by layer. Expect roughly a minute per image rather than seconds; 12–16 GB cards keep more resident.
- Requires `diffusers >= 0.36` and the `gguf` package (both already in requirements.txt). Known single-file quirks — pad-token shapes, base-repo resolution for third-party ports — are handled by the engine.

## Architecture

Three processes talking over localhost HTTP + WebSocket:

```
┌────────────────────────┐         ┌─────────────────────────────┐
│  React UI (port 1420)  │  HTTP   │  Python Engine (port 8766)  │
│  Vite · Zustand · TW   │◄───────►│  FastAPI · uvicorn          │
│                        │   WS    │                             │
│  src/                  │         │  backend/                   │
│  ├─ pages/             │         │  ├─ main.py    REST + WS    │
│  │   GeneratePage      │         │  ├─ api/models.py           │
│  │   ModelsPage        │         │  │   download/load/delete   │
│  │   HistoryPage       │         │  ├─ preprocessors.py        │
│  │   SettingsPage      │         │  │   canny/depth detection  │
│  ├─ stores/appStore.ts │         │  └─ engines/diffusion.py    │
│  │   single source of  │         │      pipelines, VRAM, GGUF  │
│  │   UI state          │         │                             │
│  ├─ api/backend.ts     │         │  ~/.aura-studio/            │
│  │   typed client      │         │  └─ models/cache,<custom>   │
│  └─ lib/historyDB.ts   │         │                             │
│      IndexedDB history │         │  Hugging Face Hub           │
└────────────────────────┘         │  (downloaded once, cached)  │
                                   └─────────────────────────────┘
```

Key design points:

- **Generation never blocks the event loop.** The pipeline runs in a worker thread (`asyncio.to_thread`) behind an asyncio lock, so progress frames stream per-step and `/health` keeps answering during heavy work.
- **Cancellation crosses the boundary.** The UI sends `{"type":"cancel"}`; the engine raises out of the diffusers step callback, freeing the GPU immediately.
- **Mode pipelines share components.** img2img/inpaint/controlnet variants are assembled around the loaded model's UNet/encoders, so switching modes costs no additional VRAM.
- **One camelCase↔snake_case seam.** Generation params/results convert at `api/backend.ts`; everything else passes through unchanged.
- **Tauri shell is optional.** The JS currently never calls Rust commands — the app is fully usable as a browser app today. Desktop packaging ships the engine as a PyInstaller sidecar (see Roadmap).

## Configuration

Environment variables (engine side):

| Variable | Default | Purpose |
|---|---|---|
| `AURA_PORT` | `8766` | Port the FastAPI engine listens on |
| `AURA_DATA_DIR` | `~/.aura-studio` | Where downloaded models are stored |
| `HF_TOKEN` / `hf auth login` | — | Token for gated Hugging Face repos |

UI settings persist in localStorage (`aura-studio-ui`: engine URL + generation params); history lives in IndexedDB (`aura-studio`, up to 50 entries).

## API Reference

REST endpoints served by the engine at `http://127.0.0.1:8766`:

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | — | `{status, engine_loaded, device, model_id}` |
| GET | `/capabilities` | — | `{loaded, model_type, modes{}, preprocessors[], controlnets[]}` |
| GET | `/models` | — | `ModelInfo[]` (catalog + downloaded + custom) |
| POST | `/models/download` | `{model_id, model_type?, file_pattern?}` | `{status, model_id, task_id}` — returns immediately |
| GET | `/models/download/{task_id}` | — | `{task_id, status, progress_pct, elapsed, error}` |
| POST | `/models/load` | `{model_id, model_type?}` | `{status, model_id, device, model_type}` — `409` while generating |
| POST | `/models/delete` | `{model_id}` | `{status, model_id}` — covers cache, custom and ControlNet dirs |
| GET | `/controlnets` | — | ControlNet catalog with download state |
| POST | `/controlnets/download` | `{model_id}` | `{status, task_id}` — same polling as above |

WebSocket protocol on `/ws/generate`:

```
Client → Server : {"prompt": "...", "negative_prompt": "", "width": 1024,
                   "height": 1024, "num_inference_steps": 4,
                   "guidance_scale": 3.5, "seed": null, "mode": "txt2img",
                   "init_image_base64": "", "mask_image_base64": "",
                   "control_image_base64": "", "strength": 0.6,
                   "controlnet_model_id": "", "control_type": "canny"}

Server → Client : {"type": "progress", "step": 1, "total": 4, "progress_pct": 25.0}
                  {"type": "complete", "image_base64": "...", "seed": 123,
                   "elapsed": 2.31, "width": 1024, "height": 1024}
                  {"type": "cancelled"}
                  {"type": "error",  "message": "No model loaded"}

Client → Server : {"type": "cancel"}   // interrupts the running pipeline
```

## Troubleshooting

**Generation is very slow (minutes per image)**
You're on CPU. Install the CUDA wheel matching your toolkit, e.g.
`pip install torch --index-url https://download.pytorch.org/whl/cu124`
Then restart the engine — the badge in the sidebar should read *GPU accelerated*.

**CUDA out of memory**
Close other GPU apps, lower resolution (1024² → 768²), or switch to a smaller model family (SDXL → SD 1.5). The engine measures free VRAM per model and offloads to RAM automatically, trading speed for capacity — on 8 GB cards FLUX/Z-Image GGUFs run via a hybrid scheme (quantized transformer resident, text encoders streamed), so they work where naive loading would OOM.

**Model download fails with an auth/401 error**
The upstream repo is gated (`stable-diffusion-2-1`, official `FLUX.1-*`). Run `hf auth login` once, or use the GGUF route — known GGUF repos automatically pull their components from ungated mirrors.

**Black/gray images from FLUX**
FLUX is bf16-only; forcing fp16 yields NaNs. This project already selects bf16 for FLUX automatically — if you see this anyway, your GPU likely lacks bf16 support and you should stick to SDXL/SD 1.5.

**Download stuck or interrupted**
Hugging Face network issues (resets are common on some proxies). Downloads resume from where they stopped — the engine retries an interrupted snapshot several times with backoff. Corporate proxies may need `HF_ENDPOINT=https://hf-mirror.com`.

**Port already in use**
Set `AURA_PORT` before starting the engine and update the URL in **Settings → Backend Connection**.

**UI shows "Disconnected" but the engine window is open**
Check the engine actually bound to 8766 (its startup log prints the port), then confirm the URL under Settings. If you changed `AURA_PORT`, update the UI to match.

**First generation after loading a model is slow**
Normal warm-up: CUDA kernels compile and caches fill on the first run. Subsequent generations are much faster.

**Loading a GGUF model seems stuck / downloads more files**
On first load the engine fetches the base-repo components (text encoder, VAE, scheduler) on top of your GGUF file. Watch the engine console; it's downloading, not hung. Later loads and other quants reuse those files.

**Z-Image / FLUX generation takes ~a minute per image on an 8 GB card**
Expected: the multi-GB text encoders can't all fit in VRAM next to the transformer, so they stream through layer by layer. A 12–16 GB card keeps more resident and runs noticeably faster.

**Nothing happens when generating / "Unsupported upgrade request" in the engine log**
The live-progress WebSocket needs a protocol implementation that plain `uvicorn` does not bundle. `pip install "websockets>=12.0"` (already in requirements.txt) and restart the engine.

## Development

```bash
npm run dev           # Vite dev server on :1420 (strict port)
npm run build         # type-check + production bundle
npm run backend:dev   # engine from repo root
npm run tauri:dev     # desktop shell in dev mode
npm run tauri:build   # desktop bundle (NSIS installer on Windows)
python scripts/build_sidecar.py   # PyInstaller single-file engine binary
python scripts/e2e_modes_test.py 8766       # live E2E: txt2img/img2img/inpaint
python scripts/regression_size_mismatch.py 8766  # size-alignment regression
```

Conventions worth knowing before you edit:

- **Backend imports resolve from `backend/`.** `main.py` does `from engines.diffusion import …`, so always run it from inside `backend/` (`run.bat` and `backend:dev` already do).
- **The camelCase seam is intentional.** Convert at the edge (`src/api/backend.ts`, WS payload builders) — don't let snake_case leak into components or camelCase into wire payloads.
- **Model metadata stays snake_case** end-to-end (`default_size`, `progress_pct`, `task_id`) since it maps 1:1 to backend models.
- **State flows through `appStore`.** Pages read via selectors; cross-page effects (e.g. loading a model retunes params) belong in store actions like `setActiveModel`.
- **Port 8766 is hardcoded in several places** (`src/api/backend.ts`, `run.bat`, `src-tauri/src/lib.rs`) plus `AURA_PORT` — changing it means touching all of them.
- Verify frontend changes with `npm run build` (tsc catches the cross-file drift) and backend changes with `python -X utf8 -m py_compile main.py engines/diffusion.py api/models.py preprocessors.py` from inside `backend/`.
- Deeper agent/architecture notes live in `AGENTS.md`.

## Roadmap

- [ ] Bundle the engine as a Tauri sidecar and auto-launch it with the desktop app
- [ ] In-app Hugging Face token setting for gated repositories
- [ ] LoRA adapter support on top of base models
- [ ] Gallery export / batch folders
- [ ] Updater for the packaged desktop app
- [ ] macOS & Linux desktop packaging

---

Built with React, Tailwind CSS, Zustand, FastAPI, diffusers and Tauri 2.
