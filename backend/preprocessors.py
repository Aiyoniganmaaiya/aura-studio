"""ControlNet conditioning-image preprocessors.

Turn a user's reference photo into the conditioning image a ControlNet
weight expects (edge maps, depth maps, …). Types without a cheap local
detector (openpose, scribble, mlsd) pass the user's image straight through —
those workflows expect an already-prepared conditioning image.
"""

import threading

import numpy as np
import torch
from PIL import Image

# Canny needs OpenCV; kept behind a guard so the rest of the engine works
# without it (the UI hides canny preprocessing when unavailable).
try:
    import cv2  # type: ignore
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


def cv2_available() -> bool:
    return _CV2_AVAILABLE


def available_preprocessors() -> list[str]:
    names = []
    if _CV2_AVAILABLE:
        names.append("canny")
    names.append("depth")  # transformers-based, downloads on first use
    names.append("passthrough")
    return names


def preprocess_canny(image: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    if not _CV2_AVAILABLE:
        raise RuntimeError(
            "Canny preprocessing requires opencv-python-headless — pip install -U opencv-python-headless"
        )
    arr = np.array(image.convert("RGB"))
    edges = cv2.Canny(arr, low, high)
    # ControlNets were trained on 3-channel condition images.
    return Image.fromarray(edges).convert("RGB")


_depth_lock = threading.Lock()
_depth_pipe = None  # cached transformers depth-estimation pipeline


def preprocess_depth(image: Image.Image) -> Image.Image:
    """Depth map via depth-anything-small (~100 MB, downloaded on first use).

    Output follows the MiDaS-style convention ControlNets were trained with:
    brighter = closer.
    """
    global _depth_pipe
    with _depth_lock:
        if _depth_pipe is None:
            from transformers import pipeline as hf_pipeline

            device = 0 if torch.cuda.is_available() else -1
            print("[Aura] Loading depth preprocessor (LiheYoung/depth-anything-small-hf)...")
            _depth_pipe = hf_pipeline(
                "depth-estimation",
                model="LiheYoung/depth-anything-small-hf",
                device=device,
            )
        result = _depth_pipe(image.convert("RGB"))
    # result["depth"] is a normalized PIL "L" image (bright = close).
    return result["depth"].convert("RGB")


def preprocess(image: Image.Image, control_type: str) -> tuple[Image.Image, str]:
    """Dispatch on control type. Returns (condition_image, actual_method).

    Unknown/passthrough types return the input unchanged so users can feed a
    ready-made conditioning image directly.
    """
    ct = (control_type or "").lower()
    if "canny" in ct:
        return preprocess_canny(image), "canny"
    if "depth" in ct:
        return preprocess_depth(image), "depth"
    return image.copy(), "passthrough"
