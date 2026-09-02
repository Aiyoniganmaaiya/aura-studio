"""ControlNet weight catalog — the curated, standard-diffusers set Aura offers.

Every entry loads through the plain ControlNetModel / FluxControlNetModel
classes (no private task-conditioning protocols). The xinsir SDXL *union*
weights would cover all types with one download, but their diffusers
integration requires a bespoke task-embedding step; single-type weights are
the reliable path until that protocol is pinned down.
"""

CONTROLNET_CATALOG = [
    {
        "id": "lllyasviel/control_v11p_sd15_canny",
        "name": "Canny · SD 1.5",
        "family": "sd15",
        "description": "Locks composition to line art and edges — the classic precise-control setup.",
        "control_types": ["canny"],
        "preprocessors": ["canny", "passthrough"],
        "size_hint": "~1.4 GB",
    },
    {
        "id": "lllyasviel/control_v11f1p_sd15_depth",
        "name": "Depth · SD 1.5",
        "family": "sd15",
        "description": "Preserves 3D layout and spatial depth from a reference photo.",
        "control_types": ["depth"],
        "preprocessors": ["depth", "passthrough"],
        "size_hint": "~1.4 GB",
    },
    {
        "id": "lllyasviel/control_v11p_sd15_openpose",
        "name": "OpenPose · SD 1.5",
        "family": "sd15",
        "description": "Reproduces human pose skeletons. Feed it an already-extracted pose image.",
        "control_types": ["openpose"],
        "preprocessors": ["passthrough"],
        "size_hint": "~1.4 GB",
    },
    {
        "id": "xinsir/controlnet-canny-sdxl-1.0",
        "name": "Canny · SDXL",
        "family": "sdxl",
        "description": "High-fidelity edge control for SDXL — among the strongest canny models available.",
        "control_types": ["canny", "lineart", "soft-edge"],
        "preprocessors": ["canny", "passthrough"],
        "size_hint": "~2.5 GB",
    },
    {
        "id": "SargeZT/controlnet-depth-sdxl-1.0",
        "name": "Depth · SDXL",
        "family": "sdxl",
        "description": "SDXL depth control trained with depth-anything maps (pairs with our preprocessor).",
        "control_types": ["depth"],
        "preprocessors": ["depth", "passthrough"],
        "size_hint": "~2.5 GB",
    },
    {
        "id": "InstantX/FLUX.1-dev-Controlnet-Canny",
        "name": "Canny · FLUX.1-dev",
        "family": "flux",
        "description": "Edge control for FLUX.1 Dev. VRAM-hungry — expect CPU offload on 8 GB cards.",
        "control_types": ["canny"],
        "preprocessors": ["canny", "passthrough"],
        "size_hint": "~4.8 GB",
    },
]


def catalog_for_family(family: str) -> list[dict]:
    return [c for c in CONTROLNET_CATALOG if c["family"] == family]


def find_controlnet(controlnet_id: str) -> dict | None:
    for c in CONTROLNET_CATALOG:
        if c["id"] == controlnet_id:
            return c
    return None
