"""Diffusion Engine - Wraps Hugging Face diffusers for image generation."""

import base64
import glob
import gc
import io
import json
import os
import secrets
import struct
import threading
import time
import traceback
from typing import Callable, Optional

import torch
from diffusers import (
    AutoPipelineForText2Image,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    FluxPipeline,
    FluxTransformer2DModel,
)
from PIL import Image

# GGUF single-file loading (diffusers >= 0.36). Kept behind a guard so the
# rest of the engine still works on older installs.
try:
    from diffusers import ZImagePipeline, ZImageTransformer2DModel, GGUFQuantizationConfig
    _GGUF_AVAILABLE = True
except ImportError:
    _GGUF_AVAILABLE = False

# FLUX.2 is a different architecture (Flux2Transformer2DModel / Flux2Pipeline
# family), not loadable through the FLUX.1 classes. Optional import: installs
# older than the FLUX.2 support release keep FLUX.1-only behavior.
try:
    from diffusers import Flux2Pipeline, Flux2Transformer2DModel as _Flux2Transformer2DModel
    _FLUX2_AVAILABLE = True
except ImportError:
    _Flux2Transformer2DModel = None
    _FLUX2_AVAILABLE = False

# Image-mode pipelines (img2img / inpaint). Assembled from the components of
# the already-loaded txt2img pipeline, so none of these classes are imported
# at module top level unconditionally — older diffusers installs keep txt2img.
try:
    from diffusers import (
        FluxImg2ImgPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipelineLegacy,
        StableDiffusionXLImg2ImgPipeline,
    )
    _SD_IMG_MODES_AVAILABLE = True
except ImportError:
    _SD_IMG_MODES_AVAILABLE = False

# The SDXL *legacy* inpaint class was dropped from recent diffusers releases
# (still absent in 0.40); its absence must not take the other families down.
try:
    from diffusers import StableDiffusionXLInpaintPipelineLegacy as _SDXLInpaintLegacy
except ImportError:
    _SDXLInpaintLegacy = None

try:
    from diffusers import ZImageImg2ImgPipeline, ZImageInpaintPipeline
    _ZIMAGE_MODES_AVAILABLE = True
except ImportError:
    _ZIMAGE_MODES_AVAILABLE = False

# ControlNet models + pipelines (standard-diffusers weights only).
try:
    from diffusers import (
        ControlNetModel,
        FluxControlNetModel,
        FluxControlNetPipeline,
        StableDiffusionControlNetPipeline,
        StableDiffusionXLControlNetPipeline,
    )
    _CN_AVAILABLE = True
except ImportError:
    _CN_AVAILABLE = False


# Some community Z-Image GGUF exports (e.g. lesliemore/z-image-turbo-nsfw-v2-GGUF,
# converted with tools other than the one diffusers was tested against) store the
# x/cap pad-token embeddings squeezed to 1-D ([dim] instead of [1, dim]); the
# single-file loader shape-checks strictly and refuses the whole file. Wrapping
# its z-image checkpoint converter to restore the missing batch dimension fixes
# such exports without touching correctly shaped ones (e.g. unsloth).
_ZIMAGE_PAD_TOKEN_KEYS = ("x_pad_token", "cap_pad_token", "siglip_pad_token")

# transformers 5.x materializes model tensors through a ThreadPoolExecutor
# (GLOBAL_WORKERS) whose native mmap copies segfault intermittently on Windows
# (hard crash of the whole engine, mid text-encoder load). Reads happen at
# load time, so pinning it to 1 here makes materialization single-threaded —
# marginally slower, reliably stable.
try:
    from transformers import core_model_loading as _core_model_loading
    _core_model_loading.GLOBAL_WORKERS = 1
except Exception:
    pass


def _install_zimage_gguf_pad_token_fix() -> None:
    if not _GGUF_AVAILABLE:
        return
    from diffusers.loaders import single_file_model as _single_file_model

    entry = _single_file_model.SINGLE_FILE_LOADABLE_CLASSES.get("ZImageTransformer2DModel")
    original = entry.get("checkpoint_mapping_fn") if entry else None
    if not entry or not original or getattr(original, "_aura_pad_fix", False):
        return

    def convert_with_pad_token_fix(*args, **kwargs):
        state_dict = original(*args, **kwargs)
        for key in _ZIMAGE_PAD_TOKEN_KEYS:
            tensor = state_dict.get(key)
            if tensor is not None and tensor.dim() == 1:
                state_dict[key] = tensor.unsqueeze(0)
        return state_dict

    convert_with_pad_token_fix._aura_pad_fix = True
    entry["checkpoint_mapping_fn"] = convert_with_pad_token_fix


_install_zimage_gguf_pad_token_fix()


# ── Model Registry ────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "stable-diffusion-v1-5/stable-diffusion-v1-5": {
        "type": "sd15",
        "name": "Stable Diffusion 1.5",
        "pipeline_class": StableDiffusionPipeline,
        "default_size": 512,
    },
    "runwayml/stable-diffusion-v1-5": {
        "type": "sd15",
        "name": "Stable Diffusion 1.5 (Runway)",
        "pipeline_class": StableDiffusionPipeline,
        "default_size": 512,
    },
    "stabilityai/stable-diffusion-2-1": {
        "type": "sd21",
        "name": "Stable Diffusion 2.1",
        "pipeline_class": StableDiffusionPipeline,
        "default_size": 768,
    },
    "stabilityai/stable-diffusion-xl-base-1.0": {
        "type": "sdxl",
        "name": "SDXL 1.0",
        "pipeline_class": StableDiffusionXLPipeline,
        "default_size": 1024,
    },
    "black-forest-labs/FLUX.1-schnell": {
        "type": "flux",
        "name": "FLUX.1 Schnell",
        "pipeline_class": FluxPipeline,
        "default_size": 1024,
    },
    "black-forest-labs/FLUX.1-dev": {
        "type": "flux",
        "name": "FLUX.1 Dev",
        "pipeline_class": FluxPipeline,
        "default_size": 1024,
    },
    "SG161222/RealVisXL_V4.0": {
        "type": "sdxl",
        "name": "RealVisXL V4.0",
        "pipeline_class": StableDiffusionXLPipeline,
        "default_size": 1024,
    },
    "dreamshaper/dreamshaper-sdxl-turbo": {
        "type": "sdxl",
        "name": "DreamShaper SDXL Turbo",
        "pipeline_class": StableDiffusionXLPipeline,
        "default_size": 1024,
    },
}


KNOWN_MODEL_TYPES = ("sd15", "sd21", "sdxl", "flux", "z-image")

# GGUF repos ship only the transformer; the rest of the pipeline comes from
# the base (unquantized) repo. Known mappings, plus a per-family default:
# any z-image GGUF finetune — however exotic its repo name — shares the
# official base's text encoder / VAE / scheduler architecture.
GGUF_BASE_REPO_MAP = {
    "unsloth/Z-Image-Turbo-GGUF": "Tongyi-MAI/Z-Image-Turbo",
    # GGUF-only repos: the suffix strip would produce an org/repo that has
    # never existed (city96 publishes no unquantized pipeline repo). Point at
    # unsloth's ungated rehost — black-forest-labs/FLUX.1-* now require auth
    # for file downloads (401 anonymously), which would brick first-load.
    "city96/FLUX.1-schnell-gguf": "unsloth/FLUX.1-schnell",
    "city96/FLUX.1-dev-gguf": "black-forest-labs/FLUX.1-dev",
}

GGUF_FAMILY_BASE = {
    "z-image": "Tongyi-MAI/Z-Image-Turbo",
}


def gguf_base_repo(model_id: str, model_type: Optional[str] = None) -> str:
    """Best-guess unquantized repo for a GGUF repo id.

    Order: explicit map → family default → '-GGUF' suffix strip. The family
    default beats the suffix heuristic because third-party GGUF ports strip
    to repo ids that don't exist (lesliemore/z-image-turbo-nsfw-v2-GGUF →
    lesliemore/z-image-turbo-nsfw-v2); their components still come from the
    official family base.
    """
    if model_id in GGUF_BASE_REPO_MAP:
        return GGUF_BASE_REPO_MAP[model_id]
    if model_type in GGUF_FAMILY_BASE:
        return GGUF_FAMILY_BASE[model_type]
    for suffix in ("-GGUF", "-gguf"):
        if model_id.endswith(suffix):
            candidate = model_id[: -len(suffix)]
            if "/" in candidate:
                return candidate
    return model_id


def resolve_model_type(model_id: str, model_type: Optional[str] = None) -> str:
    """Trust an explicit known type; otherwise sniff from the repo id.

    'custom' (how the UI labels user-added models) and empty strings are
    NOT valid families — they must be resolved by sniffing, or FLUX/SDXL
    customs would load through the wrong pipeline class.
    """
    if model_type in KNOWN_MODEL_TYPES:
        return model_type
    return detect_model_type(model_id)


def detect_model_type(model_id: str) -> str:
    """Detect model type from its ID (heuristic substring sniff)."""
    lower = model_id.lower()
    if "z-image" in lower or "z_image" in lower or "zimage" in lower:
        return "z-image"
    if "flux" in lower:
        return "flux"
    if "xl" in lower or "sdxl" in lower or "realvis" in lower or "dreamshaper" in lower:
        return "sdxl"
    if "2-1" in lower or "2.1" in lower or "sd21" in lower or "diffusion-2" in lower:
        return "sd21"
    return "sd15"


def _repo_id_is_flux2(model_id: str) -> bool:
    """FLUX.2 (incl. klein) shares the 'flux' family's dtype/VRAM profile but
    needs its own transformer + pipeline classes."""
    lower = (model_id or "").lower()
    return "flux.2" in lower or "flux2" in lower or "flux_2" in lower


def _pipeline_class_from_model_index_file(path: str) -> Optional[type]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            cls_name = json.load(f).get("_class_name")
        import diffusers
        cls = getattr(diffusers, cls_name, None) if cls_name else None
        return cls if isinstance(cls, type) else None
    except Exception:
        return None


def _pipeline_class_from_repo_config(repo: str) -> Optional[type]:
    """The pipeline class a repo's own model_index.json declares.

    FLUX.2 repos declare Flux2Pipeline / Flux2KleinPipeline; trusting the
    repo beats substring sniffing, which would route FLUX.2 through the
    FLUX.1 classes and fail on the first weight. Works for local snapshot
    dirs and (via hf_hub_download) plain repo ids; None falls back to the
    registry/sniff path.
    """
    local = os.path.join(repo, "model_index.json")
    if os.path.isfile(local):
        return _pipeline_class_from_model_index_file(local)
    try:
        from huggingface_hub import hf_hub_download
        fetched = hf_hub_download(repo_id=repo, filename="model_index.json")
        return _pipeline_class_from_model_index_file(fetched)
    except Exception:
        return None


def _gguf_transformer_class(model_id: str, model_type: str):
    """Single-file loader class for a GGUF transformer, keyed by architecture."""
    if model_type == "z-image":
        return ZImageTransformer2DModel
    if _repo_id_is_flux2(model_id):
        if not _FLUX2_AVAILABLE:
            raise RuntimeError(
                "FLUX.2 GGUF requires a diffusers release with FLUX.2 support "
                "(Flux2Transformer2DModel) — upgrade: pip install -U diffusers"
            )
        return _Flux2Transformer2DModel
    return FluxTransformer2DModel


def _safetensors_is_lora(path: str) -> bool:
    """Cheap header probe: LoRA adapters carry lora_A/lora_B weight names."""
    try:
        with open(path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_len))
        return any("lora_a" in k.lower() or "lora_b" in k.lower() for k in header)
    except Exception:
        return False


def _find_lora_single_file(path: Optional[str]) -> Optional[str]:
    """If a model dir is a bare LoRA safetensors (no pipeline layout), return it."""
    if not path or not os.path.isdir(path):
        return None
    if os.path.isfile(os.path.join(path, "model_index.json")):
        return None
    for st in sorted(glob.glob(os.path.join(path, "**", "*.safetensors"), recursive=True)):
        if _safetensors_is_lora(st):
            return st
    return None


def get_pipeline_class(model_id: str, model_type: Optional[str] = None):
    """Get the appropriate pipeline class for a model."""
    # Check registry first
    if model_id in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_id]["pipeline_class"]

    # Detect from type hint
    mt = resolve_model_type(model_id, model_type)
    type_map = {
        "z-image": ZImagePipeline if _GGUF_AVAILABLE else None,
        "flux": FluxPipeline,
        "sdxl": StableDiffusionXLPipeline,
        "sd21": StableDiffusionPipeline,
        "sd15": StableDiffusionPipeline,
    }
    cls = type_map.get(mt, AutoPipelineForText2Image)
    if cls is None:
        raise RuntimeError(
            "Z-Image support requires diffusers >= 0.36 — upgrade diffusers to load this model."
        )
    return cls


def get_default_size(model_id: str, model_type: Optional[str] = None) -> int:
    """Get default image size for a model."""
    if model_id in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_id]["default_size"]
    mt = resolve_model_type(model_id, model_type)
    size_map = {"z-image": 1024, "flux": 1024, "sdxl": 1024, "sd21": 768, "sd15": 512}
    return size_map.get(mt, 512)


def get_recommended_settings(model_type: str) -> dict:
    """Recommended steps/guidance per model family."""
    presets = {
        "sd15": {"steps": 20, "guidance_scale": 7.5},
        "sd21": {"steps": 20, "guidance_scale": 7.5},
        "sdxl": {"steps": 25, "guidance_scale": 7.0},
        "flux": {"steps": 25, "guidance_scale": 3.5},  # schnell caps to 4/none at generate-time
        "z-image": {"steps": 8, "guidance_scale": 1.0},  # distilled turbo: CFG off at generate-time
        "custom": {"steps": 20, "guidance_scale": 7.5},
    }
    return presets.get(model_type, presets["custom"])


class GenerationCancelled(Exception):
    """Raised inside the denoising loop when the user requests cancellation."""


# Rough VRAM need (GB) per model family — used to decide between direct GPU
# placement and CPU-offload. z-image assumes a Q6_K-class GGUF transformer
# (~6 GB quantized); its Qwen3-4B text encoder (~8 GB bf16) never fits beside
# it on consumer cards, so this family realistically always offloads.
_VRAM_NEED_GB = {"sd15": 4, "sd21": 5, "sdxl": 10, "flux": 16, "z-image": 14, "custom": 6}


# ── Engine ────────────────────────────────────────────────────────────


class DiffusionEngine:
    """Engine for running diffusion models for image generation.

    Thread-safety: load_model / generate / unload are serialized by the
    engine-wide lock; the FastAPI layer runs them via asyncio.to_thread and
    guards them with an asyncio lock, so no two heavy operations overlap.
    """

    def __init__(
        self,
        model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        device: str = "cuda",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.model_type = detect_model_type(model_id)
        self.pipeline = None
        self.loaded = False

        # Assembled image-mode pipelines (img2img / inpaint / controlnet),
        # keyed by mode. They SHARE components with the txt2img pipeline, so
        # each extra mode costs no additional model VRAM.
        self._mode_pipelines: dict = {}

        # RLock (not Lock): generate() holds this while calling load_model,
        # which re-acquires it on the same thread — a plain Lock would
        # deadlock there forever.
        self._lock = threading.RLock()
        self._cancel_requested = False

        print(f"[Aura] Engine initialized. Device: {self.device}, Model: {model_id}")

    # ── Loading ───────────────────────────────────────────────────────

    def _resolve_dtype(self, model_type: str) -> torch.dtype:
        """Pick a dtype that actually works for the given model/device.

        - FLUX and Z-Image are bf16-native (fp16 is untested / NaN-prone for
          both), so use bf16 when the GPU supports it.
        - fp16 on CPU hits many unsupported/slow kernels; use fp32 there.
        - GGUF weights keep their quantized storage regardless; this dtype is
          the compute dtype they dequantize to.
        """
        if self.device != "cuda":
            return torch.float32
        if model_type in ("flux", "z-image"):
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float16

    def _load_gguf_pipeline(self, gguf_path: str, model_type: str):
        """Load a GGUF single-file checkpoint (officially the ONLY supported
        GGUF route — pipeline-level from_single_file does not take .gguf).

        The quantized transformer comes from the file; every other component
        (text encoder, tokenizer, VAE, scheduler) is pulled from the base
        unquantized repo on first load. The transformer class follows the
        architecture (z-image / FLUX.2 / FLUX.1) and the pipeline class is
        whatever the base repo's model_index.json declares.
        """
        if not _GGUF_AVAILABLE:
            raise RuntimeError(
                "GGUF loading requires diffusers >= 0.36. Upgrade: pip install -U diffusers"
            )
        base_repo = gguf_base_repo(self.model_id, model_type)
        pipe_cls = (_pipeline_class_from_repo_config(base_repo)
                    or get_pipeline_class(self.model_id, model_type))
        transformer_cls = _gguf_transformer_class(self.model_id, model_type)
        print(f"[Aura] GGUF transformer: {os.path.basename(gguf_path)} | components from {base_repo}")

        transformer = transformer_cls.from_single_file(
            gguf_path,
            config=base_repo,
            # The config repo has no ROOT config.json (only model_index.json);
            # component configs live in subfolders. Passing config without
            # subfolder makes diffusers look for <repo>/config.json and fail.
            subfolder="transformer",
            quantization_config=GGUFQuantizationConfig(compute_dtype=self.dtype),
            torch_dtype=self.dtype,
        )
        pipe = pipe_cls.from_pretrained(base_repo, transformer=transformer, torch_dtype=self.dtype)
        return pipe

    @staticmethod
    def _find_gguf_file(local_path: Optional[str]) -> Optional[str]:
        """If a downloaded dir holds a GGUF checkpoint, return its path."""
        if not local_path or not os.path.isdir(local_path):
            return None
        matches = sorted(glob.glob(os.path.join(local_path, "**", "*.gguf"), recursive=True))
        return matches[0] if matches else None

    def _load_pipeline(self, source: str, model_type: str):
        """from_pretrained with graceful fallbacks.

        Tries, in order: fp16 variant safetensors → plain safetensors → any
        format. `source` may be a repo id OR a local directory (downloaded
        models), which is how downloaded files actually reach the pipeline.
        """
        pipe_cls = get_pipeline_class(source if source in MODEL_REGISTRY else self.model_id, model_type)
        # Prefer the pipeline class the weights themselves declare (FLUX.2
        # repos are 'flux' to the sniffer but need Flux2*Pipeline classes).
        if os.path.isdir(source):
            pipe_cls = _pipeline_class_from_model_index_file(
                os.path.join(source, "model_index.json")) or pipe_cls
        elif model_type == "flux" and _repo_id_is_flux2(self.model_id):
            pipe_cls = (_pipeline_class_from_repo_config(self.model_id)
                        or (Flux2Pipeline if _FLUX2_AVAILABLE else pipe_cls))
        # Only SD1.x/2.x pipelines accept safety_checker; SDXL/FLUX/Z-Image do not.
        safety_kwargs = {} if model_type in ("sdxl", "flux", "z-image") else {"safety_checker": None}

        attempts = [
            {"dtype": self.dtype, "use_safetensors": True, "variant": "fp16"},
            {"dtype": self.dtype, "use_safetensors": True},
            {"dtype": self.dtype},
        ]

        last_error: Optional[Exception] = None
        for kwargs in attempts:
            try:
                return pipe_cls.from_pretrained(source, **safety_kwargs, **kwargs)
            except TypeError:
                raise  # wrong arguments for this pipeline class — don't retry blindly
            except Exception as e:  # missing variant / missing safetensors / etc.
                last_error = e
        raise last_error  # type: ignore[misc]

    def _apply_device_strategy(self, pipe, model_type: str, gguf: bool = False):
        """Place the pipeline on the best available device/memory config.

        If the GPU has enough free VRAM, keep everything on-device (fast).
        Otherwise fall back to accelerate's cpu offload instead of OOM-ing.
        `gguf` marks pipelines whose transformer holds GGUF-quantized params
        (they cannot go through accelerate's sequential-offload meta init).
        """
        if self.device != "cuda":
            pipe.to("cpu")
            return

        need_gb = _VRAM_NEED_GB.get(model_type, 8)
        try:
            free_bytes, _total = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024 ** 3)
        except Exception:
            free_gb = 0.0

        if free_gb >= need_gb * 1.2:
            pipe.to("cuda")
            # PyTorch 2.x already uses fused scaled-dot-product attention;
            # attention slicing only slows it down there.
            if int(torch.__version__.split(".")[0]) < 2 and hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
            print(f"[Aura] Model on cuda ({free_gb:.1f} GB free >= {need_gb} GB needed)")
        else:
            if (gguf and model_type == "flux"
                    and getattr(pipe, "text_encoder", None) is not None):
                # FLUX GGUF on small cards (covers FLUX.1 and FLUX.2 alike).
                # Sequential (op-level) offload cannot run at all —
                # accelerate's meta-device init chokes on diffusers' GGUF
                # parameter type (KeyError: None) — while model-level offload
                # OOMs: the multi-GB text encoder stays resident while the
                # quantized transformer comes up. Split the difference:
                # keep the transformer+VAE resident on the GPU and stream the
                # text encoder layer by layer via accelerate's per-module
                # cpu_offload.
                try:
                    from accelerate import cpu_offload as _accel_cpu_offload
                    pipe.transformer.to(self.device)
                    if getattr(pipe, "vae", None) is not None:
                        pipe.vae.to(self.device)
                    # Flux-family pipelines carry TWO text encoders (CLIP +
                    # T5-XXL for FLUX.1, a single Mistral for FLUX.2) — every
                    # one of them must be hooked, or an unhooked encoder stays
                    # on CPU while the pipeline feeds it CUDA inputs.
                    offloaded = []
                    for enc_name in ("text_encoder", "text_encoder_2"):
                        enc = getattr(pipe, enc_name, None)
                        if enc is not None:
                            _accel_cpu_offload(enc, execution_device=torch.device(self.device))
                            offloaded.append(enc_name)
                    print(f"[Aura] FLUX GGUF: transformer+VAE resident on GPU, "
                          f"{', '.join(offloaded)} layer-offloaded ({free_gb:.1f} GB free)")
                    return
                except Exception as inner:
                    print(f"[Aura] FLUX.2 GGUF hybrid placement failed "
                          f"({type(inner).__name__}: {inner}); falling back")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            if hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload()  # moves weights itself
                print(f"[Aura] Only {free_gb:.1f} GB free (< {need_gb} GB needed); using cpu offload")
            else:
                pipe.to("cuda")

    def load_model(self, model_id: str, model_type: Optional[str] = None,
                   local_path: Optional[str] = None):
        """Load a diffusion model into memory.

        Args:
            model_id: Repo id (used for type detection / reporting).
            local_path: Local directory holding the downloaded snapshot. When
                provided it takes precedence over `model_id`, so models
                downloaded by the app are what actually gets loaded.
        """
        with self._lock:
            mt = resolve_model_type(model_id, model_type)
            self.model_type = mt
            self.dtype = self._resolve_dtype(mt)
            source = local_path or model_id
            gguf_file = self._find_gguf_file(local_path)

            print(f"[Aura] Loading model: {model_id} (dtype={self.dtype}, source={gguf_file or source})")

            try:
                if gguf_file:
                    pipe = self._load_gguf_pipeline(gguf_file, mt)
                else:
                    lora_file = _find_lora_single_file(source)
                    if lora_file:
                        raise RuntimeError(
                            f"{os.path.basename(lora_file)} is a LoRA adapter, not a "
                            "standalone model — it weights a base checkpoint and cannot "
                            "be loaded as a pipeline. Add the base model it was trained "
                            "for instead."
                        )
                    pipe = self._load_pipeline(source, mt)
                self._apply_device_strategy(pipe, mt, gguf=gguf_file is not None)

                self.pipeline = pipe
                self.model_id = model_id
                self.loaded = True
                # Components belong to the new model — stale mode pipelines
                # from the previous model must not be reused.
                self._mode_pipelines = {}
                print(f"[Aura] Model loaded: {model_id} (type: {mt})")
                return True
            except GenerationCancelled:
                raise
            except Exception as e:
                self.pipeline = None
                self.loaded = False
                # A failed load can leave half-moved weights on the GPU (they
                # are only garbage-collected lazily); release them so the next
                # attempt — possibly a smaller model — is not OOM-ed by a
                # ghost of the failed one.
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[Aura] Failed to load model {model_id}: {type(e).__name__}: {e}")
                traceback.print_exc()
                raise

    # ── Image modes (img2img / inpaint) ──────────────────────────────

    # Per-family availability. sd21 uses the same SD classes as sd15.
    _MODE_FAMILIES = {
        "img2img": {"sd15", "sd21", "sdxl", "flux", "z-image"},
        "inpaint": {"sd15", "sd21", "sdxl", "z-image"},
        "controlnet": {"sd15", "sdxl", "flux"},
    }

    def supports_mode(self, mode: str) -> bool:
        if mode == "txt2img":
            return True
        # FLUX.2 has no img2img/controlnet pipeline classes in diffusers yet —
        # reporting them available would crash at assembly time.
        if self.model_type == "flux" and _repo_id_is_flux2(self.model_id or ""):
            return False
        families = self._MODE_FAMILIES.get(mode)
        if not families or self.model_type not in families:
            return False
        # Family-level import guards decide the rest.
        if self.model_type == "z-image":
            return _ZIMAGE_MODES_AVAILABLE
        if mode == "controlnet":
            return _CN_AVAILABLE
        if mode == "inpaint" and self.model_type == "sdxl":
            return _SD_IMG_MODES_AVAILABLE and _SDXLInpaintLegacy is not None
        return _SD_IMG_MODES_AVAILABLE

    def _mode_unavailable_error(self, mode: str) -> RuntimeError:
        if self.model_type == "z-image" and not _ZIMAGE_MODES_AVAILABLE:
            return RuntimeError(
                f"{mode} for Z-Image requires diffusers >= 0.36 — upgrade diffusers."
            )
        if mode == "controlnet" and not _CN_AVAILABLE:
            return RuntimeError(
                "ControlNet requires a newer diffusers — upgrade: pip install -U diffusers"
            )
        if (self.model_type == "sdxl" and mode == "inpaint"
                and _SDXLInpaintLegacy is None):
            return RuntimeError(
                "SDXL inpainting relies on the legacy pipeline class that this "
                "diffusers version no longer ships. Use an inpaint-specific "
                "SDXL checkpoint via img2img instead, or change diffusers version."
            )
        if self.model_type in ("sd15", "sd21", "sdxl", "flux") and not _SD_IMG_MODES_AVAILABLE:
            return RuntimeError(
                f"{mode} requires a newer diffusers — upgrade: pip install -U diffusers"
            )
        if (self.model_type == "flux" and _repo_id_is_flux2(self.model_id or "")):
            return RuntimeError(
                f"FLUX.2 does not support '{mode}' yet — diffusers has no "
                "FLUX.2 img2img/controlnet pipelines. Use txt2img for this model."
            )
        return RuntimeError(
            f"Mode '{mode}' is not supported by model type '{self.model_type}'."
        )

    @staticmethod
    def _shared_components(pipe) -> dict:
        """Pull the loadable components off a pipeline so image-mode variants
        can be assembled around the SAME module objects (zero extra VRAM).
        Whatever the family doesn't have (e.g. transformer vs unet) is absent."""
        comps = {
            "scheduler": pipe.scheduler,
            "vae": pipe.vae,
        }
        for attr in ("text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2",
                     "transformer", "unet"):
            value = getattr(pipe, attr, None)
            if value is not None:
                comps[attr] = value
        return comps

    def _assemble_mode_pipeline(self, mode: str, controlnet=None):
        """Build an img2img/inpaint/controlnet pipeline from the loaded
        txt2img components. `controlnet` is a loaded ControlNetModel /
        FluxControlNetModel for controlnet mode."""
        mt = self.model_type
        comps = self._shared_components(self.pipeline)

        # The SD-family classes above are only referenced by these branches;
        # z-image has its own guard inside its branch below.
        if (mt in ("sd15", "sd21", "sdxl", "flux") and mode != "controlnet"
                and not _SD_IMG_MODES_AVAILABLE):
            raise self._mode_unavailable_error(mode)

        try:
            if mt in ("sd15", "sd21"):
                if mode == "img2img":
                    # Legacy classes take safety kwargs; keep them quiet. The
                    # two safety fields are REQUIRED constructor args but the
                    # txt2img pipeline was loaded with safety_checker=None, so
                    # they never appear in comps — pass them explicitly.
                    pipe = StableDiffusionImg2ImgPipeline(
                        **comps, safety_checker=None, feature_extractor=None,
                        requires_safety_checker=False,
                    )
                elif mode == "inpaint":  # legacy: regular 4-channel unet + mask concat
                    pipe = StableDiffusionInpaintPipelineLegacy(
                        **comps, safety_checker=None, feature_extractor=None,
                        requires_safety_checker=False,
                    )
                else:  # controlnet
                    pipe = StableDiffusionControlNetPipeline(
                        **comps, controlnet=controlnet, safety_checker=None,
                        feature_extractor=None, requires_safety_checker=False,
                    )
            elif mt == "sdxl":
                if mode == "img2img":
                    pipe = StableDiffusionXLImg2ImgPipeline(**comps)
                elif mode == "inpaint":
                    if _SDXLInpaintLegacy is None:
                        raise self._mode_unavailable_error(mode)
                    pipe = _SDXLInpaintLegacy(**comps)
                else:
                    pipe = StableDiffusionXLControlNetPipeline(**comps, controlnet=controlnet)
            elif mt == "flux":
                if mode == "img2img":
                    pipe = FluxImg2ImgPipeline(**comps)
                elif mode == "controlnet":
                    pipe = FluxControlNetPipeline(**comps, controlnet=controlnet)
                else:
                    raise self._mode_unavailable_error(mode)
            elif mt == "z-image":
                if not _ZIMAGE_MODES_AVAILABLE:
                    raise self._mode_unavailable_error(mode)
                cls = ZImageImg2ImgPipeline if mode == "img2img" else ZImageInpaintPipeline
                pipe = cls(**comps)
            else:
                raise self._mode_unavailable_error(mode)
        except TypeError as e:
            raise RuntimeError(
                f"Could not assemble {mt} {mode} pipeline from loaded components: {e}"
            )
        return pipe

    def _load_controlnet_model(self, source: str):
        """Load ControlNet weights from a repo id or local directory."""
        cls = FluxControlNetModel if self.model_type == "flux" else ControlNetModel
        print(f"[Aura] Loading ControlNet weights: {source}")
        return cls.from_pretrained(source, torch_dtype=self.dtype)

    def _get_mode_pipeline(self, mode: str, controlnet_source: Optional[str] = None):
        """Return the pipeline for a generation mode, assembling + caching on
        first use. Device strategy is applied once per assembled instance.

        Note on cpu-offload: accelerate's offload hooks are registered per
        pipeline instance, so two cached pipelines alternate cleanly only in
        .to(cuda) setups; under offload, the most recently ASSEMBLED pipeline
        owns the hooks — acceptable since a session sticks to one mode.
        """
        if mode == "txt2img":
            if self.pipeline is None:
                self.load_model(self.model_id)
            return self.pipeline

        if not self.supports_mode(mode):
            raise self._mode_unavailable_error(mode)

        cache_key = f"controlnet:{controlnet_source}" if mode == "controlnet" else mode
        cached = self._mode_pipelines.get(cache_key)
        if cached is not None:
            return cached

        if self.pipeline is None:
            self.load_model(self.model_id)

        controlnet = None
        if mode == "controlnet":
            if not controlnet_source:
                raise RuntimeError("ControlNet generation needs a controlnet model id")
            controlnet = self._load_controlnet_model(controlnet_source)

        print(f"[Aura] Assembling '{mode}' pipeline for {self.model_type}")
        pipe = self._assemble_mode_pipeline(mode, controlnet=controlnet)
        self._apply_device_strategy(pipe, self.model_type)
        self._mode_pipelines[cache_key] = pipe
        return pipe

    # ── Generation ────────────────────────────────────────────────────

    def cancel(self):
        """Request cancellation of a running generation (thread-safe)."""
        self._cancel_requested = True

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        mode: str = "txt2img",
        init_image=None,
        mask_image=None,
        strength: float = 0.6,
        control_image=None,
        controlnet_source: Optional[str] = None,
        controlnet_conditioning_scale: float = 1.0,
    ) -> dict:
        """Generate an image from a text prompt (txt2img) or an input image.

        Args:
            progress_callback: Called with (current_step, total_steps) during generation.
            mode: "txt2img" | "img2img" | "inpaint" | "controlnet".
            init_image: PIL image required by img2img/inpaint.
            mask_image: PIL mask (white = repaint), inpaint only.
            strength: img2img/inpaint transform amount in [0.01, 1].
            controlnet_source: repo id or local dir of ControlNet weights (controlnet mode).
            control_image: reference/conditioning PIL image (controlnet mode).
            controlnet_conditioning_scale: CN steer strength, usually 0.5–1.0.
        Raises:
            GenerationCancelled: if cancel() was requested mid-generation.
            RuntimeError: on unsupported mode/model combinations.
        """
        with self._lock:
            return self._generate_locked(
                prompt, negative_prompt, width, height,
                num_inference_steps, guidance_scale, seed, progress_callback,
                mode, init_image, mask_image, strength,
                control_image, controlnet_source, controlnet_conditioning_scale,
            )

    def _generate_locked(self, prompt, negative_prompt, width, height,
                         num_inference_steps, guidance_scale, seed,
                         progress_callback, mode="txt2img",
                         init_image=None, mask_image=None, strength=0.6,
                         control_image=None, controlnet_source=None,
                         controlnet_conditioning_scale=1.0) -> dict:
        pipe = self._get_mode_pipeline(mode, controlnet_source=controlnet_source)

        model_type = self.model_type
        self._cancel_requested = False

        # Auto-adjust size based on model type
        if model_type in ("sdxl", "flux", "z-image"):
            width = max(256, min(width, 2048))
            height = max(256, min(height, 2048))
        else:
            width = max(256, min(width, 1024))
            height = max(256, min(height, 1024))

        # Round to the family's latent alignment. Z-Image's pipelines reject
        # sizes not divisible by vae_scale_factor*2 == 16 outright.
        mult = 16 if model_type == "z-image" else 8
        width = (width // mult) * mult
        height = (height // mult) * mult

        # Image-mode inputs must match the (rounded) output size exactly: the
        # Z-Image img2img/inpaint pipelines encode the input at its OWN pixel
        # size while noise/mask latents follow the width/height arguments —
        # any mismatch crashes with a tensor shape error. Resize here so the
        # client can send any combination safely. Masks are re-binarized after
        # scaling so interpolation never leaves grey edges in the mask.
        if mode in ("img2img", "inpaint") and init_image is not None \
                and init_image.size != (width, height):
            init_image = init_image.resize((width, height), Image.LANCZOS)
        if mode == "inpaint" and mask_image is not None:
            if mask_image.size != (width, height):
                mask_image = mask_image.resize((width, height), Image.LANCZOS)
            mask_image = mask_image.point(lambda p: 255 if p >= 128 else 0)
        if mode == "controlnet" and control_image is not None \
                and control_image.size != (width, height):
            control_image = control_image.resize((width, height), Image.LANCZOS)

        # Set seed (cryptographically random when unset, so retries differ)
        actual_seed = seed if seed is not None else secrets.randbits(31)
        generator = torch.Generator(device="cpu" if self.device != "cuda" else self.device)
        generator.manual_seed(actual_seed)

        start_time = time.time()

        cancelled_check = self

        # Set up progress callback
        if progress_callback is not None:
            def on_step(pipe, step_index, timestep, callback_kwargs):
                if cancelled_check._cancel_requested:
                    # Raising here unwinds the denoising loop immediately.
                    raise GenerationCancelled()
                progress_callback(step_index + 1, num_inference_steps)
                return callback_kwargs

            callback_fn = on_step
        else:
            callback_fn = None

        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt if negative_prompt else None,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "callback_on_step_end": callback_fn,
        }

        # Image-mode inputs. The pipelines resize init/mask to width×height.
        if mode in ("img2img", "inpaint"):
            if init_image is None:
                raise RuntimeError(f"Mode '{mode}' requires an input image")
            kwargs["image"] = init_image
            kwargs["strength"] = max(0.01, min(float(strength), 1.0))
            if mode == "inpaint":
                if mask_image is None:
                    raise RuntimeError("Mode 'inpaint' requires a mask image")
                kwargs["mask_image"] = mask_image
        elif mode == "controlnet":
            if control_image is None:
                raise RuntimeError("Mode 'controlnet' requires a control image")
            kwargs["image"] = control_image
            kwargs["controlnet_conditioning_scale"] = max(
                0.05, min(float(controlnet_conditioning_scale), 2.0)
            )

        # Remove None values
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        # Remove callback if None (some pipelines don't accept None)
        if callback_fn is None:
            kwargs.pop("callback_on_step_end", None)

        # FLUX-specific adjustments
        if model_type == "flux":
            kwargs.pop("negative_prompt", None)
            if "schnell" in self.model_id.lower():
                kwargs.pop("guidance_scale", None)
                kwargs["num_inference_steps"] = min(num_inference_steps, 4)

        # Z-Image Turbo is CFG-distilled (official usage: guidance 0, ~8 steps).
        if model_type == "z-image":
            kwargs["num_inference_steps"] = min(num_inference_steps, 8)
            kwargs["guidance_scale"] = 0.0

        # The legacy SD inpaint class has no width/height args — it takes the
        # canvas from the init image + mask, which _generate_locked already
        # resized to the requested (aligned) size.
        if model_type in ("sd15", "sd21") and mode == "inpaint":
            kwargs.pop("width", None)
            kwargs.pop("height", None)

        result = pipe(**kwargs)
        image = result.images[0]

        elapsed = time.time() - start_time
        print(f"[Aura] Generated in {elapsed:.2f}s | Seed: {actual_seed}")

        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "image_base64": img_base64,
            "seed": actual_seed,
            "elapsed": round(elapsed, 2),
            "width": width,
            "height": height,
        }

    # ── Lifecycle ─────────────────────────────────────────────────────

    def unload(self):
        """Free up GPU/system memory held by the pipeline."""
        with self._lock:
            self._mode_pipelines = {}  # assembled variants share the components
            if self.pipeline is not None:
                del self.pipeline
                self.pipeline = None
                self.loaded = False
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("[Aura] Model unloaded, memory freed")
