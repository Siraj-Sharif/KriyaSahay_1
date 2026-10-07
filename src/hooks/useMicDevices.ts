import { useCallback, useEffect, useMemo, useState } from "react";

export type MicKind = "default" | "built-in" | "headset" | "external" | "virtual";

export interface MicDevice {
  /** "" means "system default" (no deviceId constraint). */
  id: string;
  label: string;
  kind: MicKind;
}

export type MicPermission = "unknown" | "granted" | "prompt" | "denied";

const STORAGE_KEY = "neurogrip.mic.v2";

/** Loopback / virtual inputs (still real "sound inputs" in Windows). Checked first. */
const VIRTUAL_HINTS = /(stereo mix|what u hear|loopback|virtual|voicemeeter|vb-audio|cable output|monitor of|obs )/i;
/** Wireless / wired headsets and earbuds. Checked before built-in so "Headset (Realtek)" isn't taken for the laptop mic. */
const HEADSET_HINTS = /(headset|head-?phone|hands-?free|earbud|ear-?buds?|\bbuds\b|airpods|\bpods\b|bluetooth|\bbt\b|jabra|bose|sony|wh-|wf-|beats|galaxy buds|oneplus buds|boat|earphone|earpiece|neckband|tws)/i;
const BUILT_IN_HINTS = /(microphone array|\barray\b|internal|built-?in|integrated|intel.*smart sound|macbook|laptop|conexant|synaptics|digital microphone|dmic|realtek)/i;

/** Browsers don't expose connection type, so we infer it from the device name. */
export function classifyMic(label: string): Exclude<MicKind, "default"> {
  if (VIRTUAL_HINTS.test(label)) return "virtual";
  if (HEADSET_HINTS.test(label)) return "headset";
  if (BUILT_IN_HINTS.test(label)) return "built-in";
  return "external";
}

const isAliasId = (id: string) => id === "default" || id === "communications";
const stripAlias = (label: string) => label.replace(/^(default|communications) - /i, "").trim();
const hasRealName = (label: string) => !!label && !/^Microphone \d+$/.test(label);

interface Stored {
  id: string;
  label: string;
}
function loadStored(): Stored {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw) as Stored;
  } catch {
    /* ignore */
  }
  return { id: "", label: "" };
}

export const isDesktopApp = () => typeof window !== "undefined" && !!window.neurogrip;

export function useMicDevices() {
  const [raw, setRaw] = useState<{ id: string; label: string; groupId: string }[]>([]);
  const [needsPermission, setNeedsPermission] = useState(true);
  const [permission, setPermission] = useState<MicPermission>("unknown");
  const [stored, setStored] = useState<Stored>(loadStored);
  // desktop: the shell grants access, so don't flash an "Allow" prompt while we auto-unlock names
  const [autoDone, setAutoDone] = useState(() => !isDesktopApp());

  const refresh = useCallback(async () => {
    try {
      const all = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "audioinput");
      setNeedsPermission(all.length === 0 || all.every((d) => !d.label));
      setRaw(all.map((d, i) => ({ id: d.deviceId, label: d.label || `Microphone ${i + 1}`, groupId: d.groupId })));
    } catch {
      setRaw([]);
    }
  }, []);

  // ----- derive the tidy device list from the raw OS list -----
  const { devices, inputsFound } = useMemo(() => {
    // Windows/Chrome add "default" and "communications" aliases that duplicate a real device.
    // Drop an alias only if its real twin (same groupId) is also listed — otherwise keep it,
    // so a headset that is *only* reported as the default/communications device never vanishes.
    const real = raw.filter((d) => !isAliasId(d.id));
    const realGroups = new Set(real.map((d) => d.groupId).filter(Boolean));
    const kept = raw.filter((d) => !isAliasId(d.id) || !d.groupId || !realGroups.has(d.groupId));

    // what Windows' default input currently is — shown on the "System default" row
    const defaultAlias = raw.find((d) => d.id === "default");
    const defaultName = defaultAlias ? stripAlias(defaultAlias.label) : "";
    const defaultLabel = hasRealName(defaultName) ? `System default · ${defaultName}` : "System default microphone";

    const list: MicDevice[] = [{ id: "", label: defaultLabel, kind: "default" }];
    const seen = new Set<string>();
    for (const d of kept) {
      const label = stripAlias(d.label);
      const key = d.id;
      if (seen.has(key)) continue;
      seen.add(key);
      list.push({ id: d.id, label, kind: classifyMic(label) });
    }
    return { devices: list, inputsFound: kept.length };
  }, [raw]);

  // ----- resolve the saved choice: by id first, then by name (ids can change between launches) -----
  const { selected, missingSelected } = useMemo(() => {
    const wantsSpecific = !!(stored.id || stored.label);
    const byId = stored.id ? devices.find((d) => d.id && d.id === stored.id) : undefined;
    const byLabel = !byId && hasRealName(stored.label) ? devices.find((d) => d.id && d.label === stored.label) : undefined;
    const found = byId ?? byLabel;
    return { selected: found ?? devices[0], missingSelected: wantsSpecific && !found && !needsPermission };
  }, [devices, stored, needsPermission]);

  // initial scan + re-scan whenever a device is plugged in / paired / removed / enabled
  useEffect(() => {
    void refresh();
    const md = navigator.mediaDevices;
    md?.addEventListener?.("devicechange", refresh);
    return () => md?.removeEventListener?.("devicechange", refresh);
  }, [refresh]);

  /** Ask for mic permission so the OS reveals real device names. */
  const requestAccess = useCallback(async () => {
    try {
      const s = await navigator.mediaDevices.getUserMedia({ audio: true });
      s.getTracks().forEach((t) => t.stop());
      setPermission("granted");
    } catch (e) {
      const name = (e as DOMException)?.name;
      setPermission(name === "NotAllowedError" || name === "SecurityError" ? "denied" : "unknown");
    }
    await refresh();
  }, [refresh]);

  // permission tracking; names appear on their own once access is granted
  useEffect(() => {
    let status: PermissionStatus | undefined;
    const sync = () => {
      if (!status) return;
      setPermission(status.state as MicPermission);
      if (status.state === "granted") void refresh();
    };
    (async () => {
      try {
        status = await navigator.permissions.query({ name: "microphone" as PermissionName });
        sync();
        status.onchange = sync;
      } catch {
        /* Permissions API unavailable — rely on label detection */
      }
    })();
    return () => {
      if (status) status.onchange = null;
    };
  }, [refresh]);

  // Desktop app: the shell already grants microphone access, so unlock device names immediately
  // instead of making the user press "Allow".
  useEffect(() => {
    if (isDesktopApp()) void requestAccess().finally(() => setAutoDone(true));
  }, [requestAccess]);

  const select = useCallback(
    (id: string) => {
      const label = devices.find((d) => d.id === id)?.label ?? "";
      const next: Stored = { id, label: id ? label : "" };
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      } catch {
        /* storage unavailable */
      }
      setStored(next);
    },
    [devices],
  );

  return { devices, selected, select, refresh, needsPermission: needsPermission && autoDone, permission, inputsFound, missingSelected, requestAccess };
}
