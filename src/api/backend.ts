/**
 * Aura Studio Backend API Client
 * Communicates with the Python FastAPI inference engine.
 */

import type { GenerationMode, GenerationParams, ModelInfo } from "../stores/appStore";

const DEFAULT_BASE = "http://127.0.0.1:8766";

let baseUrl = DEFAULT_BASE;

export function setBaseUrl(url: string) {
  baseUrl = url || DEFAULT_BASE;
}

export function getBaseUrl() {
  return baseUrl;
}

async function request<T>(
  path: string,
  options: RequestInit & { timeoutMs?: number } = {}
): Promise<T> {
  const { timeoutMs, ...init } = options;
  const url = `${baseUrl}${path}`;
  const res = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init.headers ?? {}),
    },
    signal: timeoutMs !== undefined ? AbortSignal.timeout(timeoutMs) : init.signal,
  });

  if (!res.ok) {
    let detail = await res.text();
    // FastAPI errors are JSON: {"detail": "..."} — surface the readable part.
    try {
      detail = JSON.parse(detail).detail ?? detail;
    } catch {
      // keep raw text
    }
    throw new Error(detail || `API Error ${res.status}`);
  }

  return res.json() as Promise<T>;
}

// ── Health ──────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  engine_loaded: boolean;
  device: string;
  model_id: string | null;
}

export async function checkHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health", { timeoutMs: 4000 });
}

// ── Models ──────────────────────────────────────────────────────────

export async function listModels(): Promise<ModelInfo[]> {
  return request<ModelInfo[]>("/models");
}

export async function downloadModel(
  modelId: string,
  modelType: string,
  filePattern?: string
): Promise<{ status: string; model_id: string; task_id: string }> {
  return request("/models/download", {
    method: "POST",
    body: JSON.stringify({
      model_id: modelId,
      model_type: modelType,
      ...(filePattern?.trim() ? { file_pattern: filePattern.trim() } : {}),
    }),
  });
}

export interface DownloadProgress {
  task_id: string;
  model_id: string;
  status: string;
  progress_pct: number;
  error: string | null;
  elapsed: number;
}

export async function getDownloadProgress(
  taskId: string
): Promise<DownloadProgress> {
  return request<DownloadProgress>(`/models/download/${taskId}`, { timeoutMs: 4000 });
}

export async function loadModel(
  modelId: string,
  modelType: string
): Promise<{ status: string; model_id: string; device: string; model_type?: string }> {
  return request("/models/load", {
    method: "POST",
    body: JSON.stringify({ model_id: modelId, model_type: modelType }),
  });
}

export async function deleteModel(
  modelId: string,
  modelType: string
): Promise<{ status: string; model_id: string }> {
  return request("/models/delete", {
    method: "POST",
    body: JSON.stringify({ model_id: modelId, model_type: modelType }),
  });
}

// ── Capabilities ─────────────────────────────────────────────────────

export interface CapabilitiesResponse {
  loaded: boolean;
  model_type: string | null;
  modes: Partial<Record<GenerationMode, boolean>>;
  preprocessors: string[];
}

export async function getCapabilities(): Promise<CapabilitiesResponse> {
  return request<CapabilitiesResponse>("/capabilities", { timeoutMs: 4000 });
}

// ── ControlNets ──────────────────────────────────────────────────────

export interface ControlNetInfo {
  id: string;
  name: string;
  family: string;
  description: string;
  /** Control types this weight provides (canny / depth / openpose / …). */
  control_types: string[];
  /** How the reference image is turned into a conditioning image. */
  preprocessors: string[];
  /** Human hint like "~1.4 GB". */
  size_hint?: string;
  downloaded: boolean;
}

export async function listControlnets(): Promise<ControlNetInfo[]> {
  return request<ControlNetInfo[]>("/controlnets");
}

export async function downloadControlnet(
  controlnetId: string
): Promise<{ status: string; model_id: string; task_id: string }> {
  return request("/controlnets/download", {
    method: "POST",
    body: JSON.stringify({ model_id: controlnetId }),
  });
}

// ── Generation (WebSocket with real-time progress + cancellation) ────

export interface GenerateResponse {
  image_base64: string;
  seed: number;
  elapsed: number;
  width?: number;
  height?: number;
}

export type WSMessage =
  | { type: "progress"; step: number; total: number; progress_pct: number; status?: string }
  | { type: "complete"; image_base64: string; seed: number; elapsed: number; width: number; height: number }
  | { type: "cancelled" }
  | { type: "error"; message: string };

export interface WSHandlers {
  /** `status` is "starting" on the first frame (engine warming up — moving
   * weights to GPU) and undefined on real denoise steps. */
  onProgress: (step: number, total: number, status?: string) => void;
  onComplete: (result: GenerateResponse) => void;
  onError: (message: string) => void;
  onCancelled?: () => void;
}

/**
 * Run a generation over WebSocket. Returns an abort function that asks the
 * backend to cancel the in-flight pipeline before closing the socket.
 *
 * Image-mode fields (init/mask/control images) are raw base64 PNG strings
 * without the data: prefix — same wire format as incoming results.
 */
export function generateImageWS(
  params: GenerationParams,
  extras: { initImage?: string | null; maskImage?: string | null; controlImage?: string | null },
  handlers: WSHandlers
): () => void {
  const { onProgress, onComplete, onError, onCancelled } = handlers;
  const mode: GenerationMode = params.mode ?? "txt2img";
  const ws = new WebSocket(`${baseUrl.replace("http", "ws")}/ws/generate`);
  let closed = false;

  ws.onopen = () => {
    ws.send(JSON.stringify({
      prompt: params.prompt,
      negative_prompt: params.negativePrompt,
      width: params.width,
      height: params.height,
      num_inference_steps: params.steps,
      guidance_scale: params.guidanceScale,
      seed: params.seed,
      model_id: params.modelId,
      mode,
      ...(mode === "img2img" || mode === "inpaint"
        ? {
            init_image_base64: extras.initImage ?? "",
            strength: params.strength ?? 0.6,
            ...(mode === "inpaint" ? { mask_image_base64: extras.maskImage ?? "" } : {}),
          }
        : {}),
      ...(mode === "controlnet"
        ? {
            control_image_base64: extras.controlImage ?? "",
            controlnet_model_id: params.cnModelId ?? "",
            control_type: params.controlType ?? "",
            controlnet_conditioning_scale: params.cnWeight ?? 1.0,
          }
        : {}),
    }));
  };

  ws.onmessage = (event) => {
    if (closed) return;
    let msg: WSMessage;
    try {
      msg = JSON.parse(event.data);
    } catch (e) {
      console.error("WS parse error:", e);
      return;
    }

    switch (msg.type) {
      case "progress":
        onProgress(msg.step, msg.total, msg.status);
        break;
      case "complete":
        closed = true;
        onComplete({
          image_base64: msg.image_base64,
          seed: msg.seed,
          elapsed: msg.elapsed,
          width: msg.width,
          height: msg.height,
        });
        ws.close();
        break;
      case "cancelled":
        closed = true;
        onCancelled?.();
        ws.close();
        break;
      case "error":
        closed = true;
        onError(msg.message);
        ws.close();
        break;
    }
  };

  ws.onerror = () => {
    if (!closed) {
      closed = true;
      onError("Cannot reach the backend engine. Is it running?");
    }
  };

  ws.onclose = () => {
    closed = true;
  };

  // Abort: tell the server to interrupt the pipeline, then disconnect.
  return () => {
    if (closed) return;
    closed = true;
    try {
      ws.send(JSON.stringify({ type: "cancel" }));
    } catch {
      // socket already gone — nothing to cancel remotely
    }
    try {
      ws.close();
    } catch {
      // ignore
    }
  };
}
