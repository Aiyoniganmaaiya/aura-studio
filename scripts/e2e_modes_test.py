"""One-shot E2E check for the multi-mode engine (txt2img/img2img/inpaint).

Run with the backend already listening on AURA_PORT (default here 8799):
    python scripts/e2e_modes_test.py
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


def req(method: str, path: str, body: dict | None = None, timeout: float = 60) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(
        f"{BASE}{path}", data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read())


def png_b64(path: str) -> str:
    return base64.b64encode(open(path, "rb").read()).decode()


def save_result_img(result: dict, name: str) -> None:
    open(name, "wb").write(base64.b64decode(result["image_base64"]))
    print(f"  saved {name} ({result['width']}x{result['height']}, seed {result['seed']}, {result['elapsed']}s)")


print("== capabilities before load ==")
print(json.dumps(req("GET", "/capabilities"), indent=1))

print(f"== loading {MODEL_ID} (this can take a few minutes) ==")
t0 = time.time()
load = req("POST", "/models/load", {"model_id": MODEL_ID}, timeout=900)
print(f"  loaded in {time.time()-t0:.0f}s on {load['device']} ({load.get('model_type')})")

caps = req("GET", "/capabilities")
print("== capabilities after load ==")
print(json.dumps(caps["modes"], indent=1))
assert caps["modes"]["img2img"] and caps["modes"]["inpaint"], "z-image must support img2img/inpaint"
assert not caps["modes"]["controlnet"], "z-image must NOT claim controlnet"

gen = {
    "prompt": "a red apple on a wooden table",
    "width": 512,
    "height": 512,
    "num_inference_steps": 4,
}

print("== txt2img ==")
save_result_img(req("POST", "/generate", gen, timeout=300), "e2e_txt.png")
init = png_b64("e2e_txt.png")

print("== img2img (strength 0.55) ==")
save_result_img(req("POST", "/generate", {**gen, "mode": "img2img",
                                          "init_image_base64": init,
                                          "strength": 0.55}, timeout=300), "e2e_i2i.png")

print("== inpaint (center square mask) ==")
# Build a white-square-on-black mask via PIL so we don't depend on fixtures.
from PIL import Image
mask = Image.new("RGB", (512, 512), (0, 0, 0))
for y in range(128, 384):
    for x in range(128, 384):
        mask.putpixel((x, y), (255, 255, 255))
buf = io.BytesIO(); mask.save(buf, format="PNG")
save_result_img(req("POST", "/generate", {**gen, "mode": "inpaint",
                                          "init_image_base64": init,
                                          "mask_image_base64": base64.b64encode(buf.getvalue()).decode(),
                                          "strength": 0.9,
                                          "prompt": "a green pear"}, timeout=300), "e2e_inpaint.png")

print("== preprocessor sanity (canny + depth) ==")
sys.path.insert(0, "backend")  # run from repo root OR backend/
import importlib
try:
    pp = importlib.import_module("preprocessors")
except ModuleNotFoundError:
    sys.path.insert(0, ".")
    pp = importlib.import_module("preprocessors")
img = Image.open("e2e_txt.png").convert("RGB")
edges = pp.preprocess(img, "canny")[0]
depth = pp.preprocess(img, "depth")[0]
assert edges.size == img.size and depth.size == img.size
edges.save("e2e_canny.png"); depth.save("e2e_depth.png")
print(f"  canny {edges.size} depth {depth.size} OK")

print("\nALL E2E CHECKS PASSED")
