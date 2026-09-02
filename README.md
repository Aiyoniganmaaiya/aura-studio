# Aura Studio

> **⚠️ PREVIEW — v0.1.0**
> This is an early preview release. Expect bugs, incomplete features, and breaking changes. APIs, UI layout, and model support may change without notice. Please report issues and share feedback — it helps shape what Aura Studio becomes.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Status: Preview](https://img.shields.io/badge/status-预览-orange)](#warning-preview)

Cross-platform AI Image Generation Studio — FLUX, SDXL and Stable Diffusion running entirely on your own machine. No accounts, no API keys, no uploads: your prompts never leave your computer.

> **🧪 本项目当前为预览版本 (v0.1.0)** — 可能存在不稳定或未完成的功能，欢迎提交 Issue 和 PR。

## Features

- **Local & private** — generation runs through a local Python engine; nothing is sent to the cloud (model weights are fetched from Hugging Face once, then cached).
- **Real-time progress** — a live progress ring streams each denoising step over WebSocket.
- **Real cancellation** — pressing Cancel actually interrupts the pipeline mid-run instead of leaving the GPU crunching in the background.
- **Model manager** — one-click download of curated models, byte-level download progress, plus any Hugging Face diffusion model via its `org/name` ID.
- **Smart parameters** — switching models auto-applies recommended size, steps and guidance for that model family; FLUX Schnell correctly locks to its 4-step turbo sampler.
- **Persistent history** — your last 50 generations survive restarts (IndexedDB), with one-click re-open, download or delete.
- **Reproducible seeds** — fix a seed to iterate on a prompt; every result records the seed it used.

## Requirements

| | Minimum | Recommended |
|---|---|---|
| OS | Windows 10/11, macOS, Linux | Windows 11 |
| Python | 3.10 – 3.13 | 3.13 (dev-tested) |
| Node.js | 18+ | 20 LTS |
| GPU | — (CPU mode works) | NVIDIA with ≥ 8 GB VRAM |
| Disk | 2 GB free | 15+ GB (models are 2–24 GB each) |

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

1. Open **Models**, click **Download** on *FLUX.1 Schnell* (~2 GB, recommended default).
2. When the download finishes, click **Load** — first load takes ~30–60 s.
3. Go to **Generate**, type a prompt, press `Ctrl+Enter`.

## Usage Guide

### Generate

The main workspace: prompt panel on the right, live canvas on the left.

| Control | What it does |
|---|---|
| Prompt | Free-form description of what you want. Click **Need an idea?** for rotating examples. |
| Negative prompt | Things to avoid (`blurry, low quality…`). Most effective with SD 1.5 / SDXL. |
| Size presets | Common resolutions; SD 1.5 is natively 512², SDXL/FLUX prefer 1024². |
| Steps | Denoising iterations. More = more refined, slower. FLUX Schnell is fixed at 4. |
| Guidance scale | How strictly the image follows the prompt (CFG). Higher = more literal, can burn out at extremes. Hidden for Schnell, which doesn't use it. |
| Seed | Fix it to reproduce an image exactly. Random mode shows the seed used under the result. |
| `Ctrl+Enter` | Generate from anywhere on the page. |
| Cancel | Interrupts the run on the GPU — not just the connection. |

Every finished image offers **Download** (named after your prompt + seed) and **Copy** (straight to clipboard where the WebView allows it).

### Models

Curated catalog plus anything from Hugging Face:

- **Download** runs in the background with true byte-level progress.
- **Load** swaps the active model; the old one is unloaded first, so VRAM doesn't stack up. The button shows a spinner while weights stream into memory.
- **Trash icon** deletes a downloaded model's files to reclaim disk space (re-downloadable anytime).
- **Custom model box** accepts any diffusers-compatible repo ID, e.g. `prompthero/openjourney-v4` or `nitrosocke/mo-di-diffusion`. Custom models are auto-detected as SD 1.5 / SD 2.1 / SDXL / FLUX / Z-Image by their config. For multi-file repos (GGUF quants, fp8 variants) add a **file filter** — see [GGUF models](#gguf-models-quantized).

Loading a model also applies sensible defaults for its family (size, steps, guidance), so you don't have to know them.

### History

Everything you generate is saved locally (up to 50 images) and survives restarts. Click a thumbnail to reopen it full-size with all its parameters restored; hover for delete; **Clear all** wipes the store.

### Settings

- **Engine URL** — change where the UI looks for the backend (default `http://127.0.0.1:8766`); useful if you moved the engine to another port or machine.
- **Engine status** — live view of connection, loaded model, compute device (GPU/CPU).
- **Clear history** — same as History page.

## Supported Models

| Model | Family | Download | VRAM (approx.) | Notes |
|---|---|---|---|---|
| FLUX.1 Schnell ⭐ recommended | FLUX | ~2 GB | 16 GB* | 4 steps, fastest quality/time ratio |
| FLUX.1 Dev | FLUX | ~24 GB | 16 GB+ | Higher fidelity, needs guidance ~3.5 |
| SDXL 1.0 | SDXL | ~7 GB | 10 GB | Great with negative prompts |
| RealVisXL V4.0 | SDXL | ~7 GB | 10 GB | Photorealistic SDXL finetune |
| Stable Diffusion 1.5 | SD 1.5 | ~4 GB | 4 GB | Lightest option, 512² native |
| DreamShaper | SD 1.5 | ~4 GB | 4 GB | Stylized illustrations |
| Z-Image Turbo (GGUF) | z-image | 3–12 GB + ~8 GB** | 8 GB (offload) | Quantized 6B turbo; 8 steps, CFG off |
| Any HF diffusers model | auto-detected | varies | varies | Paste `org/name` in Models |

\* On smaller GPUs the engine automatically falls back to sequential CPU offload instead of failing — slower, but it works.

### GGUF models (quantized)

Single-file GGUF checkpoints work for **Z-Image Turbo** today (FLUX-style GGUF repos load the same way). In **Models → Download Custom Model**, enter the repo ID and a *file filter*:

```
Model ID:    unsloth/Z-Image-Turbo-GGUF
File filter: *Q6_K*.gguf
```

Multi-quant repos ship every quantization (the unsloth one totals ~80 GB) — the filter downloads just your pick (~6 GB for Q6_K). The engine then loads the quantized transformer from that file and pulls the remaining components (Qwen3-4B text encoder, VAE, scheduler) from the base `Tongyi-MAI/Z-Image-Turbo` repo.

Two things worth knowing:

- **First load downloads ~8 GB extra** on top of the GGUF file (the base-repo components). This is one-time and shared across all Z-Image quants.
- **8 GB cards run it via automatic CPU offload** — the transformer (~6 GB as Q6_K) and the text encoder (~8 GB bf16) can't coexist in VRAM, so components take turns on the GPU. Expect roughly a minute per image rather than seconds.
- Requires `diffusers >= 0.36` and `pip install gguf` (both already in requirements.txt).

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
│  │   HistoryPage       │         │  └─ engines/diffusion.py    │
│  │   SettingsPage      │         │      pipeline lifecycle     │
│  ├─ stores/appStore.ts │         │                             │
│  │   single source of  │         │  ~/.aura-studio/            │
│  │   UI state          │         │  └─ models/cache/<weights>  │
│  ├─ api/backend.ts     │         │                             │
│  │   typed client      │         │  Hugging Face Hub           │
│  └─ lib/historyDB.ts   │         │  (downloaded once, cached)  │
│      IndexedDB history │         │                             │
└────────────────────────┘         └─────────────────────────────┘
```

Key design points:

- **Generation never blocks the event loop.** The pipeline runs in a worker thread (`asyncio.to_thread`) behind an asyncio lock, so progress frames stream per-step and `/health` keeps answering during heavy work.
- **Cancellation crosses the boundary.** The UI sends `{"type":"cancel"}`; the engine raises out of the diffusers step callback, freeing the GPU immediately.
- **One camelCase↔snake_case seam.** Generation params/results convert at `api/backend.ts`; everything else passes through unchanged.
- **Tauri shell is optional.** The JS currently never calls Rust commands — the app is fully usable as a browser app today. Desktop packaging ships the engine as a PyInstaller sidecar (see Roadmap).

## Configuration

Environment variables (engine side):

| Variable | Default | Purpose |
|---|---|---|
| `AURA_PORT` | `8766` | Port the FastAPI engine listens on |
| `AURA_DATA_DIR` | `~/.aura-studio` | Where downloaded models are stored |

UI settings persist in localStorage (`aura-studio-ui`: engine URL + generation params); history lives in IndexedDB (`aura-studio`, up to 50 entries).

## API Reference

REST endpoints served by the engine at `http://127.0.0.1:8766`:

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | — | `{status, engine_loaded, device, model_id}` |
| GET | `/models` | — | `ModelInfo[]` (catalog + downloaded + custom) |
| POST | `/models/download` | `{model_id, model_type?, file_pattern?}` | `{status, model_id, task_id}` — returns immediately |
| GET | `/models/download/{task_id}` | — | `{task_id, status, progress_pct, elapsed, error}` |
| POST | `/models/load` | `{model_id, model_type?}` | `{status, model_id, device}` — `409` while generating |
| POST | `/models/delete` | `{model_id, model_type?}` | `{status, model_id}` |

WebSocket protocol on `/ws/generate`:

```
Client → Server : {"prompt": "...", "negative_prompt": "", "width": 1024,
                   "height": 1024, "num_inference_steps": 4,
                   "guidance_scale": 3.5, "seed": null, "model_id": "..."}

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
Close other GPU apps, lower resolution (1024² → 768²), or switch to a smaller model family (SDXL → SD 1.5). The engine offloads to RAM automatically when VRAM is tight, trading speed for capacity.

**Black/gray images from FLUX**
FLUX is bf16-only; forcing fp16 yields NaNs. This project already selects bf16 for FLUX automatically — if you see this anyway, your GPU likely lacks bf16 support and you should stick to SDXL/SD 1.5.

**Download stuck at 0%**
Hugging Face rate-limiting or network issues. Downloads resume from where they stopped when retried; corporate proxies may need `HF_ENDPOINT=https://hf-mirror.com`.

**Port already in use**
Set `AURA_PORT` before starting the engine and update the URL in **Settings → Backend Connection**.

**UI shows "Disconnected" but the engine window is open**
Check the engine actually bound to 8766 (its startup log prints the port), then confirm the URL under Settings. If you changed `AURA_PORT`, update the UI to match.

**First generation after loading a model is slow**
Normal warm-up: CUDA kernels compile and caches fill on the first run. Subsequent generations are much faster.

**Loading a GGUF model seems stuck / downloads more files**
On first load the engine fetches the base-repo components (text encoder, VAE, scheduler — ~8 GB for Z-Image) on top of your GGUF file. Watch the engine console; it's downloading, not hung. Later loads and other quants reuse those files.

**Z-Image generation takes ~a minute per image on an 8 GB card**
Expected: quantized transformer + 8 GB-class text encoder can't fit together in 8 GB VRAM, so they take turns via CPU offload. A 12–16 GB card keeps everything resident and runs it in seconds.

**Nothing happens when generating / "Unsupported upgrade request" in the engine log**
The live-progress WebSocket needs a protocol implementation that plain `uvicorn` does not bundle. `pip install "websockets>=12.0"` (already in requirements.txt) and restart the engine.

## Development

```bash
npm run dev          # Vite dev server on :1420 (strict port)
npm run build        # type-check + production bundle
npm run backend:dev  # engine from repo root
npm run tauri:dev    # desktop shell in dev mode
npm run build:sidecar # PyInstaller single-file engine binary
```

Conventions worth knowing before you edit:

- **Backend imports resolve from `backend/`.** `main.py` does `from engines.diffusion import …`, so always run it from inside `backend/` (`run.bat` and `backend:dev` already do).
- **The camelCase seam is intentional.** Convert at the edge (`src/api/backend.ts`, WS payload builders) — don't let snake_case leak into components or camelCase into wire payloads.
- **Model metadata stays snake_case** end-to-end (`default_size`, `progress_pct`, `task_id`) since it maps 1:1 to backend models.
- **State flows through `appStore`.** Pages read via selectors; cross-page effects (e.g. loading a model retunes params) belong in store actions like `setActiveModel`.
- Verify frontend changes with `npm run build` (tsc catches the cross-file drift) and backend changes with `python -m py_compile main.py engines/diffusion.py api/models.py`.

## Roadmap

- [ ] Bundle the engine as a Tauri sidecar (`npm run build:sidecar` exists; wiring + auto-launch pending)
- [ ] In-app image-to-image and ControlNet inputs
- [ ] Gallery export / batch folders
- [ ] LoRA support
- [ ] Updater for the packaged desktop app

---

Built with React, Tailwind CSS, Zustand, FastAPI, diffusers and Tauri 2.
