import { create } from "zustand";
import { persist } from "zustand/middleware";
import { setBaseUrl as setApiBaseUrl } from "../api/backend";
import { FAMILY_DEFAULT_SIZE, FAMILY_SETTINGS, familyOf, isFluxSchnell } from "../lib/modelUtils";

export type GenerationMode = "txt2img" | "img2img" | "inpaint" | "controlnet";

export interface GenerationParams {
  prompt: string;
  negativePrompt: string;
  width: number;
  height: number;
  steps: number;
  guidanceScale: number;
  seed: number | null;
  modelId: string;
  // Optional pro-mode fields (absent on persisted pre-upgrade entries —
  // always read with defaults).
  mode?: GenerationMode;
  /** img2img / inpaint transform amount, 0.01–1. */
  strength?: number;
  batchCount?: number;
  /** ControlNet: selected control type slug and conditioning scale. */
  controlType?: string;
  cnWeight?: number;
  cnModelId?: string;
}

export interface GenerationResult {
  id: string;
  imageBase64: string;
  prompt: string;
  negativePrompt: string;
  params: GenerationParams;
  seed: number;
  elapsed: number;
  timestamp: number;
}

export interface ModelInfo {
  id: string;
  name: string;
  type: string;
  description?: string;
  default_size?: number;
  recommended?: boolean;
  local: boolean;
  downloaded: boolean;
}

export interface ProgressInfo {
  step: number;
  total: number;
  progressPct: number;
  status: string;
}

export type ToastKind = "success" | "error" | "info";

export interface ToastItem {
  id: number;
  kind: ToastKind;
  message: string;
}

/** In-memory cap for the session result list; persistence trims separately. */
const MAX_SESSION_RESULTS = 100;

/** Working images for the image modes — raw base64 PNG, never persisted. */
export interface WorkspaceImages {
  /** img2img source. */
  initImage: string | null;
  /** inpaint mask (white = repaint). */
  maskImage: string | null;
  /** ControlNet reference image. */
  controlImage: string | null;
}

interface AppState extends WorkspaceImages {
  // Backend connection
  backendUrl: string;
  backendConnected: boolean;

  // Generation params
  params: GenerationParams;
  isGenerating: boolean;

  // Real-time progress
  progress: ProgressInfo | null;

  // Results (session view over the IndexedDB history)
  results: GenerationResult[];
  currentResult: GenerationResult | null;

  // Models
  models: ModelInfo[];
  activeModel: ModelInfo | null;

  // Engine status
  engineLoaded: boolean;
  engineDevice: string;
  engineModelId: string | null;

  // Toasts
  toasts: ToastItem[];

  // Actions
  setBackendUrl: (url: string) => void;
  setBackendConnected: (connected: boolean) => void;
  updateParams: (params: Partial<GenerationParams>) => void;
  setParams: (params: GenerationParams) => void;
  setInitImage: (b64: string | null) => void;
  setMaskImage: (b64: string | null) => void;
  setControlImage: (b64: string | null) => void;
  setIsGenerating: (v: boolean) => void;
  setProgress: (progress: ProgressInfo | null) => void;
  addResult: (result: GenerationResult) => void;
  setResults: (results: GenerationResult[]) => void;
  removeResult: (id: string) => void;
  clearResults: () => void;
  setCurrentResult: (result: GenerationResult | null) => void;
  setModels: (models: ModelInfo[]) => void;
  setActiveModel: (model: ModelInfo) => void;
  setEngineStatus: (loaded: boolean, device: string, modelId?: string | null) => void;
  pushToast: (kind: ToastKind, message: string) => void;
  dismissToast: (id: number) => void;
}

const defaultParams: GenerationParams = {
  prompt: "",
  negativePrompt: "",
  width: 1024,
  height: 1024,
  steps: 4,
  guidanceScale: 3.5,
  seed: null,
  modelId: "black-forest-labs/FLUX.1-schnell",
};

let toastCounter = 0;

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      backendUrl: "http://127.0.0.1:8766",
      backendConnected: false,

      initImage: null,
      maskImage: null,
      controlImage: null,

      params: { ...defaultParams },
      isGenerating: false,
      progress: null,

      results: [],
      currentResult: null,

      models: [],
      activeModel: null,

      engineLoaded: false,
      engineDevice: "",
      engineModelId: null,

      toasts: [],

      setBackendUrl: (url) => {
        const trimmed = url.trim().replace(/\/+$/, "");
        setApiBaseUrl(trimmed);
        set({ backendUrl: trimmed });
      },
      setBackendConnected: (connected) => set({ backendConnected: connected }),

      updateParams: (partial) =>
        set((state) => ({
          params: { ...state.params, ...partial },
        })),

      setParams: (params) => set({ params }),

      setInitImage: (b64) => set({ initImage: b64 }),
      setMaskImage: (b64) => set({ maskImage: b64 }),
      setControlImage: (b64) => set({ controlImage: b64 }),

      setIsGenerating: (isGenerating) => set({ isGenerating }),
      setProgress: (progress) => set({ progress }),

      addResult: (result) =>
        set((state) => ({
          results: [result, ...state.results].slice(0, MAX_SESSION_RESULTS),
          currentResult: result,
        })),

      // Replace the whole list (used when hydrating from IndexedDB history).
      setResults: (results) => set({ results }),

      removeResult: (id) =>
        set((state) => ({
          results: state.results.filter((r) => r.id !== id),
          currentResult:
            state.currentResult?.id === id ? null : state.currentResult,
        })),

      clearResults: () => set({ results: [], currentResult: null }),

      setCurrentResult: (result) => set({ currentResult: result }),

      setModels: (models) => set({ models }),

      // Switching models also applies that family's recommended size and
      // sampler settings, so a fresh model always produces sensible output.
      setActiveModel: (model) =>
        set((state) => {
          const family = familyOf(model);
          const size = model.default_size ?? FAMILY_DEFAULT_SIZE[family];
          const preset = FAMILY_SETTINGS[family];
          const schnell = isFluxSchnell(model.id);
          return {
            activeModel: model,
            params: {
              ...state.params,
              modelId: model.id,
              width: size,
              height: size,
              steps: schnell ? 4 : preset.steps,
              guidanceScale: preset.guidanceScale,
            },
          };
        }),

      setEngineStatus: (loaded, device, modelId) =>
        set({
          engineLoaded: loaded,
          engineDevice: device,
          engineModelId: modelId ?? null,
        }),

      pushToast: (kind, message) => {
        const id = ++toastCounter + Date.now();
        set((state) => ({ toasts: [...state.toasts.slice(-4), { id, kind, message }] }));
        setTimeout(() => get().dismissToast(id), 4500);
      },

      dismissToast: (id) =>
        set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) })),
    }),
    {
      name: "aura-studio-ui",
      partialize: (s) => ({ backendUrl: s.backendUrl, params: s.params }),
      onRehydrateStorage: () => (state) => {
        // Re-apply the persisted URL to the API client after hydration.
        if (state?.backendUrl) setApiBaseUrl(state.backendUrl);
      },
    }
  )
);
