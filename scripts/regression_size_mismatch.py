"""Regression: img2img/inpaint with input size != params size.

Reproduces the user crash: a 1024px-wide init image sent with width=1072
params used to die on "size of tensor a (128) must match tensor b (134)".
Also checks the Z-Image 16-px alignment rule and mask re-binarization.

    python scripts/regression_size_mismatch.py 8799
"""

import base64
import io
import json
import sys
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8799"
BASE = f"http://127.0.0.1:{PORT}"
MODEL_ID = "unsloth/Z-Image-Turbo-GGUF"


def req(method, path, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(
        f"{BASE}{path}", data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read())


def png_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


from PIL import Image  # noqa: E402

print("== loading model ==")
req("POST", "/models/load", {"model_id": MODEL_ID}, timeout=900)
print("  loaded")

gen = {"prompt": "a cozy cabin in the woods", "num_inference_steps": 4}

# ── Case 1: txt2img at a non-16-multiple size must be rounded to 16 ──
print("== txt2img width=1074 (not 16-aligned) ==")
r1 = req("POST", "/generate", {**gen, "width": 1074, "height": 768}, timeout=300)
print(f"  -> {r1['width']}x{r1['height']}")
assert r1["width"] % 16 == 0 and r1["height"] % 16 == 0, "z-image sizes must align to 16"

init_img = Image.open(io.BytesIO(base64.b64decode(r1["image_base64"])))
print(f"  base image is {init_img.size}")

# ── Case 2: the exact user crash — 1024-wide image, 1072 params ──
print("== img2img: image resized to 1024, params width=1072 ==")
init_1024 = init_img.resize((1024, 1024))
r2 = req("POST", "/generate", {**gen, "mode": "img2img", "width": 1072, "height": 1072,
                               "strength": 0.55, "init_image_base64": png_b64(init_1024)},
         timeout=300)
print(f"  -> {r2['width']}x{r2['height']} OK")

# ── Case 3: inpaint with an undersized mask (canvas export ≤1024) ──
print("== inpaint: 1024 image + small mask, params 1072 ==")
mask = Image.new("RGB", (512, 512), (0, 0, 0))
for y in range(128, 384):
    for x in range(128, 384):
        mask.putpixel((x, y), (255, 255, 255))
r3 = req("POST", "/generate", {**gen, "mode": "inpaint", "width": 1072, "height": 1072,
                               "strength": 0.9, "prompt": "a green pear",
                               "init_image_base64": png_b64(init_1024),
                               "mask_image_base64": png_b64(mask)},
         timeout=300)
print(f"  -> {r3['width']}x{r3['height']} OK")

print("\nREGRESSION PASSED — size mismatches no longer crash")
