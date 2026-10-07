import { useCallback, useEffect, useState } from "react";
import { camerasFrom, isPipelineControlAvailable, listCameras } from "../services/pipelineControl";
import { useDesktopRuntime } from "./useDesktopRuntime";

export interface CameraDevice {
  id: string;
  label: string;
  kind: "built-in" | "external";
  simulated?: boolean;
}

const STORAGE_KEY = "neurogrip.camera";

const BUILT_IN_HINTS = /(integrated|built-?in|internal|facetime|laptop|isight|front|rear|surface|hp\s?(true|wide)\s?vision|hp.*(camera|webcam)|webcam hd$)/i;

/** Best-effort classification — browsers don't expose connection type, so we infer from the label. */
export function classifyCamera(label: string): "built-in" | "external" {
  return BUILT_IN_HINTS.test(label) ? "built-in" : "external";
}

/**
 * Desktop camera list.
 *
 * Phase 2: in the Electron app the cameras are the ones the *Python pipeline* can actually
 * open (`camera.list`), addressed by their real OpenCV index — not the browser's
 * `deviceId`s, which the pipeline cannot use. Unavailable indices are not offered.
 */
async function listBackendCameras(): Promise<CameraDevice[]> {
  const res = await listCameras();
  return camerasFrom(res)
    .filter((c) => c.available || c.active)
    .map<CameraDevice>((c) => ({
      id: String(c.index),
      label: c.available ? c.label : `${c.label} (unavailable)`,
      kind: classifyCamera(c.label),
      simulated: false,
    }));
}

/** Shown when the browser can't enumerate devices (permission denied / no hardware / sandboxed). */
const SIMULATED: CameraDevice[] = [
  { id: "sim-builtin", label: "Built-in Webcam (simulated)", kind: "built-in", simulated: true },
  { id: "sim-external", label: "External USB Webcam (simulated)", kind: "external", simulated: true },
];

export function useCameraDevices() {
  const backendControlled = isPipelineControlAvailable();
  const runtime = useDesktopRuntime();
  const [devices, setDevices] = useState<CameraDevice[]>(() => backendControlled ? [] : SIMULATED);
  const [selectedId, setSelectedIdState] = useState<string>(() => localStorage.getItem(STORAGE_KEY) ?? "");

  const refresh = useCallback(async () => {
    if (backendControlled) {
      // Real devices, from the process that owns the camera.
      const cams = await listBackendCameras();
      setDevices(cams.length ? cams : []);
      return;
    }
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      const cams = all
        .filter((d) => d.kind === "videoinput" && d.deviceId)
        .map<CameraDevice>((d, i) => {
          const label = d.label || `Camera ${i + 1}`;
          return { id: d.deviceId, label, kind: classifyCamera(label) };
        });
      setDevices(cams.length ? cams : SIMULATED);
    } catch {
      setDevices(SIMULATED);
    }
  }, [backendControlled]);

  useEffect(() => {
    if (backendControlled) {
      // Wait until Electron confirms the Python pipeline is ready; don't offer fallback
      // cameras or issue discovery requests against a backend that is still starting.
      if (runtime?.backend === "ready") void refresh();
      return;
    }
    void refresh();
    const md = navigator.mediaDevices;
    md?.addEventListener?.("devicechange", refresh);
    return () => md?.removeEventListener?.("devicechange", refresh);
  }, [refresh, backendControlled, runtime?.backend]);

  // fall back to the first device if the stored one is missing
  const selected: CameraDevice =
    devices.find((d) => d.id === selectedId) ??
    devices[0] ?? { id: "", label: backendControlled ? "No camera reported by pipeline" : "No camera", kind: "external" };

  const select = useCallback((id: string) => {
    localStorage.setItem(STORAGE_KEY, id);
    setSelectedIdState(id);
  }, []);

  return { devices, selected, select, refresh };
}
