/**
 * Model family helpers — frontend mirror of the backend's
 * engines/diffusion.py heuristics (detect_model_type / get_recommended_settings).
 */

import { ModelInfo } from "../stores/appStore";

export type ModelFamily = "sd15" | "sd21" | "sdxl" | "flux" | "z-image";

export function detectFamily(modelId: string): ModelFamily {
  const lower = modelId.toLowerCase();
  if (lower.includes("z-image") || lower.includes("z_image") || lower.includes("zimage"))
    return "z-image";
  if (lower.includes("flux")) return "flux";
  if (lower.includes("xl") || lower.includes("sdxl") || lower.includes("realvis") || lower.includes("dreamshaper"))
    return "sdxl";
  if (lower.includes("2-1") || lower.includes("2.1") || lower.includes("sd21") || lower.includes("diffusion-2"))
    return "sd21";
  return "sd15";
}

export function familyOf(model: ModelInfo | null | undefined): ModelFamily {
  if (!model) return "sd15";
  if (["sd15", "sd21", "sdxl", "flux", "z-image"].includes(model.type))
    return model.type as ModelFamily;
  return detectFamily(model.id);
}

export function isFluxSchnell(modelId: string): boolean {
  const lower = modelId.toLowerCase();
  return lower.includes("flux") && lower.includes("schnell");
}

export const FAMILY_DEFAULT_SIZE: Record<ModelFamily, number> = {
  sd15: 512,
  sd21: 768,
  sdxl: 1024,
  flux: 1024,
  "z-image": 1024,
};

export const FAMILY_SETTINGS: Record<ModelFamily, { steps: number; guidanceScale: number }> = {
  sd15: { steps: 20, guidanceScale: 7.5 },
  sd21: { steps: 20, guidanceScale: 7.5 },
  sdxl: { steps: 25, guidanceScale: 7.0 },
  flux: { steps: 25, guidanceScale: 3.5 },
  // Distilled turbo — CFG off, ~8 steps (backend clamps regardless).
  "z-image": { steps: 8, guidanceScale: 1.0 },
};

/**
 * Frontend mirror of the backend's mode support (engines/diffusion.py
 * _MODE_FAMILIES + the ControlNet catalog). Used to grey out modes a model
 * family can't run instead of failing at generate time.
 */
export interface FamilyCapabilities {
  img2img: boolean;
  inpaint: boolean;
  controlnet: boolean;
}

export const FAMILY_CAPABILITIES: Record<ModelFamily, FamilyCapabilities> = {
  sd15: { img2img: true, inpaint: true, controlnet: true },
  sd21: { img2img: true, inpaint: true, controlnet: false },
  sdxl: { img2img: true, inpaint: true, controlnet: true },
  flux: { img2img: true, inpaint: false, controlnet: true },
  // Z-Image ships no ControlNet weights or pipeline class in diffusers.
  "z-image": { img2img: true, inpaint: true, controlnet: false },
};

export function capabilitiesFor(family: ModelFamily): FamilyCapabilities {
  return FAMILY_CAPABILITIES[family] ?? FAMILY_CAPABILITIES.sd15;
}
