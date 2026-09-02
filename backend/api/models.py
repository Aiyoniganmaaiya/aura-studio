"""Model Manager - Handles model discovery, download, and management."""

import fnmatch
import os
import shutil
import threading
import time
import uuid
from typing import Optional

from huggingface_hub import HfApi, snapshot_download  # type: ignore
from huggingface_hub.utils import HfHubHTTPError  # type: ignore
from tqdm.auto import tqdm  # type: ignore

from engines.diffusion import detect_model_type, get_default_size
from api.catalog import CONTROLNET_CATALOG

IGNORE_PATTERNS = ["*.pt", "*.ckpt"]


class ModelManager:
    """Manages available models - both built-in and user-downloaded."""

    CATALOG = [
        {
            "id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "name": "Stable Diffusion 1.5",
            "type": "sd15",
            "description": "The classic SD 1.5 - fast, lightweight, widely compatible",
            "default_size": 512,
            "recommended": True,
        },
        {
            "id": "stabilityai/stable-diffusion-2-1",
            "name": "Stable Diffusion 2.1",
            "type": "sd21",
            "description": "Improved quality with 768px native resolution",
            "default_size": 768,
            "recommended": False,
        },
        {
            "id": "stabilityai/stable-diffusion-xl-base-1.0",
            "name": "SDXL 1.0",
            "type": "sdxl",
            "description": "High quality 1024px generation with better composition",
            "default_size": 1024,
            "recommended": True,
        },
        {
            "id": "black-forest-labs/FLUX.1-schnell",
            "name": "FLUX.1 Schnell",
            "type": "flux",
            "description": "Fast FLUX model - 4 step generation, high quality",
            "default_size": 1024,
            "recommended": True,
        },
        {
            "id": "black-forest-labs/FLUX.1-dev",
            "name": "FLUX.1 Dev",
            "type": "flux",
            "description": "FLUX Dev - higher quality, more steps needed",
            "default_size": 1024,
            "recommended": False,
        },
        {
            "id": "SG161222/RealVisXL_V4.0",
            "name": "RealVisXL V4.0",
            "type": "sdxl",
            "description": "Photorealistic SDXL model",
            "default_size": 1024,
            "recommended": False,
        },
        {
            "id": "dreamshaper/dreamshaper-sdxl-turbo",
            "name": "DreamShaper SDXL Turbo",
            "type": "sdxl",
            "description": "Fast SDXL Turbo model with artistic style",
            "default_size": 1024,
            "recommended": False,
        },
    ]

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.models_dir = os.path.join(data_dir, "models")
        self.cache_dir = os.path.join(self.models_dir, "cache")
        self.custom_dir = os.path.join(self.models_dir, "custom")
        # ControlNet weights live apart from base models: they are add-ons
        # attached to whichever pipeline is loaded, not loadable pipelines.
        self.controlnet_dir = os.path.join(self.models_dir, "controlnet")

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.custom_dir, exist_ok=True)
        os.makedirs(self.controlnet_dir, exist_ok=True)

        # Download progress tracking: {task_id: dict}
        self._download_tasks: dict[str, dict] = {}
        self._tasks_lock = threading.Lock()

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _safe_name(model_id: str) -> str:
        """'org/name' -> 'org--name' (filesystem-safe directory name)."""
        return model_id.replace("/", "--")

    @staticmethod
    def _id_from_safe_name(safe_name: str) -> str:
        """Inverse of _safe_name; '--' cannot appear in HF user/repo names."""
        return safe_name.replace("--", "/")

    def _target_dir(self, model_id: str, kind: str = "base") -> str:
        if kind == "controlnet":
            return os.path.join(self.controlnet_dir, self._safe_name(model_id))
        is_catalog = any(m["id"] == model_id for m in self.CATALOG)
        return os.path.join(
            self.cache_dir if is_catalog else self.custom_dir,
            self._safe_name(model_id),
        )

    # ── Model Listing ──────────────────────────────────────────────

    def get_models(self) -> list[dict]:
        """Return list of available models with download status."""
        models = []
        for model in self.CATALOG:
            downloaded = self.get_local_path(model["id"]) is not None
            models.append({
                "id": model["id"],
                "name": model["name"],
                "type": model["type"],
                "description": model["description"],
                "default_size": model["default_size"],
                "recommended": model["recommended"],
                "local": False,
                "downloaded": downloaded,
            })

        # Add custom downloaded models. Directory names are 'org--name'; map
        # them back to real repo ids so they can actually be loaded.
        if os.path.isdir(self.custom_dir):
            for item in sorted(os.listdir(self.custom_dir)):
                item_path = os.path.join(self.custom_dir, item)
                if not os.path.isdir(item_path):
                    continue
                model_id = self._id_from_safe_name(item)
                models.append({
                    "id": model_id,
                    "name": item,
                    "type": "custom",
                    "description": f"Custom model ({detect_model_type(model_id)}) from Hugging Face",
                    "default_size": get_default_size(model_id),
                    "recommended": False,
                    "local": True,
                    # Interrupted downloads stay listed but re-offer Download
                    # (snapshot_download resumes into the same directory).
                    "downloaded": self._is_complete_snapshot(item_path),
                })

        return models

    # Weight file extensions a loadable snapshot must contain.
    _WEIGHT_SUFFIXES = (".safetensors", ".bin")

    def _is_complete_snapshot(self, path: str, require_pipeline_config: bool = True) -> bool:
        """A directory is only loadable if it has the pipeline config AND at
        least one weight file. Interrupted downloads leave partial dirs
        (some files moved into place, weights still missing) which would
        otherwise masquerade as ready-to-load and 500 on from_pretrained.
        GGUF repos are the exception: a single .gguf file IS the model.

        ControlNet weight dirs have no model_index.json (they are single
        components, not pipelines) — pass require_pipeline_config=False and a
        config.json + weight file is enough."""
        if not os.path.isdir(path):
            return False
        for _root, _dirs, files in os.walk(path):
            if any(f.endswith(".gguf") for f in files):
                return True
        if require_pipeline_config and not os.path.isfile(os.path.join(path, "model_index.json")):
            # Component-style dirs (controlnet) ship config.json instead.
            if not os.path.isfile(os.path.join(path, "config.json")):
                return False
        for _root, _dirs, files in os.walk(path):
            if any(f.endswith(self._WEIGHT_SUFFIXES) for f in files):
                return True
        return False

    def get_local_path(self, model_id: str, kind: str = "base") -> Optional[str]:
        """Local directory holding a downloaded snapshot, or None.

        Checks the app-managed cache and custom dirs — this is the bridge
        that makes downloads visible to the engine. Partial/interrupted
        downloads are deliberately not returned: they can't be loaded.
        """
        bases = (self.controlnet_dir,) if kind == "controlnet" else (self.cache_dir, self.custom_dir)
        for base in bases:
            path = os.path.join(base, self._safe_name(model_id))
            if self._is_complete_snapshot(path, require_pipeline_config=(kind != "controlnet")):
                return path
        return None

    def delete_model(self, model_id: str) -> bool:
        """Delete a downloaded model's local files. Returns True if deleted."""
        deleted = False
        for base in (self.cache_dir, self.custom_dir):
            path = os.path.join(base, self._safe_name(model_id))
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
                deleted = True
        return deleted

    # ── Download (Async with Progress) ─────────────────────────────

    def list_controlnets(self) -> list[dict]:
        """Catalog of ControlNet weights with local download state."""
        out = []
        for c in CONTROLNET_CATALOG:
            entry = dict(c)
            entry["downloaded"] = self.get_local_path(c["id"], kind="controlnet") is not None
            out.append(entry)
        return out

    def download_model_async(self, model_id: str, model_type: str = "",
                             file_pattern: str = "", kind: str = "base") -> str:
        """Start a model download in a background thread. Returns a task_id for polling.

        file_pattern (optional): fnmatch pattern restricting which repo files
        are fetched, e.g. "*Q6_K*.gguf" to grab one quantization from a
        multi-quant GGUF repo instead of the whole (~80 GB) tree.
        kind: "base" (pipeline) or "controlnet" (weight add-on) — picks the
        storage directory.
        """
        task_id = str(uuid.uuid4())[:8]

        with self._tasks_lock:
            self._download_tasks[task_id] = {
                "model_id": model_id,
                "file_pattern": file_pattern.strip(),
                "kind": kind,
                "status": "starting",
                "progress_pct": 0,
                "current": 0,
                "total": 0,
                "error": None,
                "started_at": time.time(),
                "completed_at": None,
            }

        thread = threading.Thread(
            target=self._download_worker,
            args=(model_id, model_type, task_id, file_pattern.strip(), kind),
            daemon=True,
        )
        thread.start()

        return task_id

    def _repo_total_bytes(self, repo_id: str, file_pattern: str = "") -> int:
        """Sum of downloadable file sizes (excluding ignored patterns, and
        restricted to file_pattern when one is given)."""
        try:
            info = HfApi().model_info(repo_id=repo_id, files_metadata=True)
            total = 0
            for sibling in info.siblings or []:
                name = sibling.rfilename
                if file_pattern and not fnmatch.fnmatch(name, file_pattern):
                    continue
                size = getattr(sibling, "size", None)
                if size is None and getattr(sibling, "lfs", None) is not None:
                    size = sibling.lfs.get("size")
                if size is None:
                    continue
                if any(fnmatch.fnmatch(name, pat) for pat in IGNORE_PATTERNS):
                    continue
                total += size
            return total
        except Exception as e:
            print(f"[Aura] Could not fetch file sizes for {repo_id}: {e}")
            return 0

    def _download_worker(self, model_id: str, model_type: str, task_id: str,
                         file_pattern: str = "", kind: str = "base"):
        """Background download worker with byte-level progress tracking."""

        def set_task(**fields):
            with self._tasks_lock:
                self._download_tasks[task_id].update(fields)

        counter = {"bytes": 0}
        counter_lock = threading.Lock()
        state = {"total": 0}

        def report_bytes(delta: int):
            with counter_lock:
                counter["bytes"] += delta
                done = counter["bytes"]
            total = state["total"]
            if total > 0:
                # Cap at 99 until snapshot_download returns; completion sets 100.
                pct = min(99.0, done / total * 100.0)
                set_task(progress_pct=round(pct, 1), current=done)

        try:
            target_dir = self._target_dir(model_id, kind)
            os.makedirs(target_dir, exist_ok=True)

            set_task(status="downloading", progress_pct=0)

            mt = model_type or detect_model_type(model_id)
            set_task(model_type=mt)

            state["total"] = self._repo_total_bytes(model_id, file_pattern)
            set_task(total=state["total"])

            # tqdm_class lets us tap into hf_hub's per-file byte updates.
            def make_tqdm_class():
                class _ReportingTqdm(tqdm):
                    def update(self, n=1):
                        if n and n > 0:
                            report_bytes(n)
                        return super().update(n)
                return _ReportingTqdm

            pattern_kwargs = {"allow_patterns": [file_pattern]} if file_pattern else {}
            # Networks reset mid-download (WinError 10054/10038 on some
            # proxies/CDNs); huggingface_hub retries single HEAD requests but
            # NOT an interrupted snapshot. snapshot_download resumes into the
            # same directory, so retry the whole snapshot a few times.
            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                try:
                    snapshot_download(
                        repo_id=model_id,
                        local_dir=target_dir,
                        ignore_patterns=IGNORE_PATTERNS,
                        tqdm_class=make_tqdm_class(),
                        **pattern_kwargs,
                    )
                    break
                except HfHubHTTPError:
                    raise  # repo-level failure (404 etc.) — retrying won't help
                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    print(f"[Aura] Download interrupted (attempt {attempt}/{max_attempts}): {e}")
                    print("[Aura] Retrying — snapshot_download resumes where it left off")
                    time.sleep(3 * attempt)

            with self._tasks_lock:
                started_at = self._download_tasks[task_id]["started_at"]
            elapsed = time.time() - started_at
            set_task(
                status="completed",
                progress_pct=100,
                completed_at=time.time(),
                current=state["total"],
            )
            print(f"[Aura] Model {model_id} downloaded in {elapsed:.1f}s -> {target_dir}")

        except HfHubHTTPError as e:
            error_msg = f"Hugging Face Hub error: {e}"
            print(f"[Aura] Download failed: {error_msg}")
            set_task(status="error", error=error_msg)
        except Exception as e:
            error_msg = str(e)
            print(f"[Aura] Download failed: {error_msg}")
            set_task(status="error", error=error_msg)

    def get_download_progress(self, task_id: str) -> Optional[dict]:
        """Get progress of a download task."""
        with self._tasks_lock:
            task = self._download_tasks.get(task_id)
            if task is None:
                return None
            snapshot = dict(task)

        elapsed = time.time() - snapshot["started_at"]
        return {
            "task_id": task_id,
            "model_id": snapshot["model_id"],
            "status": snapshot["status"],
            "progress_pct": snapshot["progress_pct"],
            "error": snapshot["error"],
            "elapsed": round(elapsed, 1),
        }
