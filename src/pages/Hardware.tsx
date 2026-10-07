import { motion } from "framer-motion";
import { useCallback, useState, useEffect } from "react";
import { GlassCard, Icon, StatusDot } from "../components/ui";
import { usePipelineState } from "../hooks/usePipelineState";
import { useDesktopRuntime } from "../hooks/useDesktopRuntime";
import { setStopArmed as requestStopArmed, stopArmedFrom } from "../services/pipelineControl";
import { cn } from "../utils/cn";

const SERVOS = [
  { ch: 0, name: "Thumb flex", range: "0–180°" },
  { ch: 1, name: "Index flex", range: "0–180°" },
  { ch: 2, name: "Middle flex", range: "0–180°" },
  { ch: 3, name: "Ring flex", range: "0–180°" },
  { ch: 4, name: "Pinky flex", range: "0–180°" },
  { ch: 5, name: "Wrist rotate", range: "0–270°" },
];

function Spec({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-white/[0.04] py-2 last:border-0">
      <span className="mono shrink-0 text-[10px] uppercase tracking-[0.2em] text-slate-500">{k}</span>
      <span className="mono text-right text-[12px] text-slate-200">{v}</span>
    </div>
  );
}

export function Hardware() {
  const pipelineState = usePipelineState();
  const runtime = useDesktopRuntime();
  const isDesktop = Boolean(window.neurogrip?.isDesktop);
  const backendReady = isDesktop && runtime?.backend === "ready";
  const transportConnected = backendReady && pipelineState.serialConnected;
  const realSerialConnected = transportConnected && pipelineState.serialMode === "real";
  const [busySerial, setBusySerial] = useState(false);
  const [serialPorts, setSerialPorts] = useState<string[]>([]);
  const [selectedPort, setSelectedPort] = useState<string>("");
  const [baudRate, setBaudRate] = useState<number>(115200);
  const [serialStatusNote, setSerialStatusNote] = useState<string>("");
  const [isTransmitting, setIsTransmitting] = useState<boolean>(false);
  const [busySafety, setBusySafety] = useState<boolean>(false);
  const [safetyNote, setSafetyNote] = useState<string>("");

  // Ports are discovered by the Python process that owns the serial interface.
  const refreshPorts = useCallback(async () => {
    if (!window.neurogrip?.serialPorts) {
      setSerialPorts([]);
      setSelectedPort("");
      return;
    }
    try {
      const found = await window.neurogrip.serialPorts();
      const ports = Array.isArray(found) ? found : [];
      setSerialPorts(ports);
      setSelectedPort((current) => {
        const active = pipelineState.serialPort;
        if (active && active !== "--" && ports.includes(active)) return active;
        if (current && ports.includes(current)) return current;
        return ports[0] ?? "";
      });
    } catch {
      setSerialPorts([]);
      setSelectedPort("");
    }
  }, [pipelineState.serialPort]);

  useEffect(() => {
    void refreshPorts();
  }, [refreshPorts, runtime?.backend]);

  // Sync with active pipeline port if reported
  useEffect(() => {
    if (backendReady && pipelineState.serialPort && pipelineState.serialPort !== "--") {
      setSelectedPort(pipelineState.serialPort);
    }
  }, [backendReady, pipelineState.serialPort]);

  const toggleSerialReal = async () => {
    if (!window.neurogrip?.isDesktop || !backendReady) {
      setSerialStatusNote("Python desktop backend is not ready.");
      return;
    }
    if (!pipelineState.serialConnected && !selectedPort) {
      setSerialStatusNote("No serial port is available. Connect a device and rescan.");
      return;
    }
    setBusySerial(true);
    setSerialStatusNote("");
    try {
      // `serialConnected` comes from the pipeline's own `serial_info.connected`, so this
      // button can never disagree with the backend about the link state.
      if (pipelineState.serialConnected) {
        const res = await window.neurogrip.serialDisconnect();
        setSerialStatusNote(res && res.ok ? "Disconnected" : res?.error || "Disconnect failed");
      } else {
        const res = await window.neurogrip.serialConnect(selectedPort, baudRate);
        if (res && !res.ok) {
          setSerialStatusNote(res.error || "Port unavailable");
        } else {
          setSerialStatusNote(`Connected · ${selectedPort} @ ${res?.baud ?? baudRate} baud`);
        }
      }
    } catch (e: any) {
      setSerialStatusNote(e?.message || "Serial error");
    } finally {
      setBusySerial(false);
    }
  };

  /**
   * ARM / DISARM the pipeline's software STOP safety flag.
   *
   * The label is derived from the backend's own `is_armed` (never local state), and after the
   * round-trip the pipeline snapshot is the only thing the UI reads back. A rejected request
   * shows the real error instead of flipping the label optimistically.
   */
  const toggleStopArm = async () => {
    if (!backendReady) {
      setSafetyNote("The Python backend has not confirmed a live safety state.");
      return;
    }
    setBusySafety(true);
    setSafetyNote("");
    try {
      const res = await requestStopArmed(!pipelineState.isArmed);
      if (!res.ok) {
        setSafetyNote(res.error || "Safety request rejected by the pipeline");
        return;
      }
      const armed = stopArmedFrom(res);
      setSafetyNote(
        armed === null
          ? "Pipeline did not report an arming state"
          : `Pipeline reports STOP ${armed ? "ARMED" : "DISARMED"}`,
      );
    } finally {
      setBusySafety(false);
    }
  };

  const sendTestCommand = async (cmd: string) => {
    if (!window.neurogrip?.serialSend || !realSerialConnected) {
      setSerialStatusNote("A confirmed real serial connection is required; mock/demo links cannot transmit to hardware.");
      return;
    }
    setIsTransmitting(true);
    setSerialStatusNote("");
    try {
      // Goes through the pipeline's own validator + serial owner (`command.send`).
      const res = await window.neurogrip.serialSend(cmd, "manual");
      if (res && res.ok) {
        setSerialStatusNote(`Transmitted ${res.frame ?? `NG1|${cmd}`}`);
      } else {
        setSerialStatusNote(res?.error || "Send failed");
      }
    } catch (e: any) {
      setSerialStatusNote(e?.message || "Failed to transmit command");
    } finally {
      setIsTransmitting(false);
    }
  };

  return (
    <div className="flex flex-col gap-5">
      <div>
        <h2 className="text-xl font-semibold text-white">Hardware &amp; connections</h2>
        <p className="text-sm text-slate-400">Host devices and link state come from the Python owner. Downstream board and actuator telemetry is not available in the current protocol.</p>
      </div>

      {!isDesktop && (
        <div className="rounded-xl border border-amber-300/15 bg-amber-300/[0.04] px-4 py-3 text-[11px] leading-relaxed text-amber-100/80">
          Browser preview only — this view cannot open a robot serial port. No browser adapter or simulated device is shown as connected hardware.
        </div>
      )}

      {/* Live host-side chain; downstream device identity is deliberately unverified. */}
      <GlassCard title="Live signal path" icon={<Icon.Chip />} strong>
        <div className="flex flex-col items-stretch gap-2 p-5 md:flex-row md:items-center">
          {[
            { t: "Camera", s: !isDesktop ? "Not attached to Python" : !backendReady ? "Awaiting Python state" : pipelineState.cameraStatus === "connected" ? `Open · index ${pipelineState.cameraIndex}` : pipelineState.cameraError || "Not opened", tone: backendReady && pipelineState.cameraStatus === "connected" ? "ok" as const : "warn" as const, icon: <Icon.Camera className="h-5 w-5" /> },
            { t: "Python CV", s: backendReady ? "Initialized · live pipeline" : runtime?.message || "Not confirmed", tone: backendReady ? "ok" as const : "warn" as const, icon: <Icon.Eye className="h-5 w-5" /> },
            { t: "Serial link", s: realSerialConnected ? `${pipelineState.serialPort} @ ${pipelineState.baudRate}` : pipelineState.serialMode === "mock" ? "Mock · no hardware" : "Disconnected", tone: realSerialConnected ? "ok" as const : "warn" as const, icon: <Icon.Serial className="h-5 w-5" /> },
            { t: "Serial device", s: realSerialConnected ? "Port open · identity unknown" : "Not identified", tone: "warn" as const, icon: <Icon.Chip className="h-5 w-5" /> },
            { t: "Servo driver", s: "Not reported by firmware", tone: "warn" as const, icon: <Icon.Grid className="h-5 w-5" /> },
            { t: "Actuators", s: "No position telemetry", tone: "warn" as const, icon: <Icon.Hand className="h-5 w-5" /> },
          ].map((n, i, arr) => (
            <div key={n.t} className="flex flex-1 items-center gap-2">
              <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: i * 0.08 }} className={cn("relative flex flex-1 flex-col items-center gap-1 rounded-xl border p-3 text-center", n.tone === "ok" ? "border-cyan-400/25 bg-cyan-500/[0.06]" : "border-white/5 bg-white/[0.02]")}>
                <span className={n.tone === "ok" ? "text-cyan-300" : "text-slate-500"}>{n.icon}</span>
                <span className="text-xs font-semibold text-white">{n.t}</span>
                <span className="mono text-[10px] text-slate-400">{n.s}</span>
                <span className="absolute right-2 top-2"><StatusDot status={n.tone} /></span>
              </motion.div>
              {i < arr.length - 1 && (
                <div className="relative hidden h-px w-6 shrink-0 bg-white/10 md:block">
                  {n.tone === "ok" && <motion.span className="absolute top-1/2 h-1 w-1 -translate-y-1/2 rounded-full bg-cyan-400 shadow-[0_0_6px_#22d3ee]" animate={{ left: ["0%", "100%"] }} transition={{ repeat: Infinity, duration: 1.2, delay: i * 0.2, ease: "linear" }} />}
                </div>
              )}
            </div>
          ))}
        </div>
        <div className="border-t border-white/5 px-5 py-2.5 text-[10px] text-slate-500">
          A serial port opening does not identify an ESP32 or confirm the PCA9685/servos. The current NG1 protocol has no downstream identity, acknowledgement or position telemetry.
        </div>
      </GlassCard>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
        <GlassCard title="Target microcontroller" icon={<Icon.Chip />} right={<span className="mono text-[9px] uppercase tracking-wider text-amber-200/70">Design reference</span>}>
          <div className="px-5 py-2">
            <Spec k="Board target" v="ESP32-WROOM-32 · not identified" />
            <Spec k="Core target" v="Xtensa LX6 dual-core @ 240 MHz" />
            <Spec k="Flash / SRAM" v="4 MB / 520 KB · target spec" />
            <Spec k="Interface target" v="USB-UART · CP2102 expected" />
            <Spec k="Protocol" v="NG1|<COMMAND>\\n" />
            <Spec k="Firmware ID" v="Not reported by current protocol" />
            <Spec k="Uptime" v="Not reported by firmware" />
          </div>
        </GlassCard>

        <GlassCard title="Target servo driver" icon={<Icon.Grid />} right={<span className="mono text-[9px] uppercase tracking-wider text-amber-200/70">Not detected</span>}>
          <div className="px-5 py-2">
            <Spec k="Driver target" v="PCA9685 · 16-channel PWM" />
            <Spec k="Bus target" v="I²C · SDA 21 / SCL 22" />
            <Spec k="Address target" v="0x40" />
            <Spec k="PWM target" v="50 Hz" />
            <Spec k="Pulse range" v="500–2500 µs · target" />
            <Spec k="Power" v="5 V / 3 A external · design spec" />
          </div>
          <div className="border-t border-white/5 p-4">
            <div className="mono mb-2 text-[9px] uppercase tracking-[0.25em] text-slate-500">Channel map</div>
            <div className="grid grid-cols-8 gap-1">
              {Array.from({ length: 16 }).map((_, ch) => {
                const s = SERVOS.find((x) => x.ch === ch);
                return (
                  <div key={ch} title={s ? `${s.name} (${s.range})` : "Unassigned"} className={cn("flex aspect-square flex-col items-center justify-center rounded-md border text-[9px]", s ? "border-cyan-400/30 bg-cyan-500/10 text-cyan-200" : "border-white/5 bg-white/[0.02] text-slate-600")}>
                    <span className="mono">{ch}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </GlassCard>

        <GlassCard title="Target actuators" icon={<Icon.Hand />} right={<span className="mono text-[9px] uppercase tracking-wider text-amber-200/70">No telemetry</span>}>
          <div className="flex flex-col gap-2 p-4">
            {SERVOS.map((s) => (
              <div key={s.ch} className="flex items-center justify-between rounded-lg border border-white/5 bg-white/[0.02] px-3 py-2">
                <div className="flex items-center gap-3">
                  <span className="mono flex h-6 w-6 items-center justify-center rounded bg-cyan-500/10 text-[10px] text-cyan-200 ring-1 ring-cyan-400/20">{s.ch}</span>
                  <div>
                    <div className="text-xs font-medium text-slate-100">{s.name}</div>
                    <div className="mono text-[9px] text-slate-500">MG996R · {s.range}</div>
                  </div>
                </div>
                <StatusDot status="warn" label="Not reported" />
              </div>
            ))}
            <p className="mt-1 text-[11px] text-slate-500">Design target: metal-gear servos for tendon-driven fingers. The current serial protocol does not report actuator presence, position or torque.</p>
          </div>
        </GlassCard>
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <GlassCard title="Camera" icon={<Icon.Camera />}>
          <div className="px-5 py-2">
            <Spec k="Capture owner" v={isDesktop ? "Python OpenCV" : "Not attached in browser"} />
            <Spec k="Status" v={<StatusDot status={backendReady && pipelineState.cameraStatus === "connected" ? "ok" : "warn"} label={backendReady ? pipelineState.cameraStatus : "awaiting Python"} />} />
            <Spec k="Active resolution" v={backendReady && pipelineState.cameraStatus === "connected" ? pipelineState.cameraResolution : "No active capture"} />
            <Spec k="CV model" v={backendReady ? pipelineState.modelUsed : "Not reported"} />
            <Spec k="Inference" v={backendReady && pipelineState.latencyMs ? `${pipelineState.latencyMs} ms` : "--"} />
            <Spec k="Landmarks" v={backendReady ? "21 per detected hand" : "Not reported"} />
          </div>
        </GlassCard>

        <GlassCard
          title="Serial Communication"
          icon={<Icon.Serial />}
          right={
            <div className="flex items-center gap-2">
              <button
                onClick={() => void refreshPorts()}
                disabled={!backendReady}
                title="Rescan serial devices through Python"
                className="mono rounded-md border border-white/10 bg-white/[0.03] px-2 py-1 text-[10px] text-slate-300 hover:border-white/20 disabled:opacity-40"
              >
                Rescan
              </button>
              <button
                onClick={toggleSerialReal}
                disabled={busySerial || !backendReady || (!transportConnected && !selectedPort)}
                className={cn(
                  "mono rounded-md px-2.5 py-1 text-[10px] uppercase tracking-wider ring-1 transition disabled:opacity-50",
                  transportConnected
                    ? "bg-rose-500/10 text-rose-300 ring-rose-400/30 hover:bg-rose-500/20"
                    : "bg-emerald-500/10 text-emerald-300 ring-emerald-400/30 hover:bg-emerald-500/20",
                )}
              >
                {busySerial ? "…" : transportConnected ? "Disconnect" : "Connect"}
              </button>
            </div>
          }
        >
          <div className="px-5 py-2">
            <div className="flex items-center justify-between border-b border-white/[0.04] py-2">
              <span className="mono text-[10px] uppercase tracking-[0.2em] text-slate-500">Port Select</span>
              <div className="flex items-center gap-2">
                <select
                  value={selectedPort}
                  disabled={!backendReady}
                  onChange={(e) => setSelectedPort(e.target.value)}
                  aria-label="Detected serial port"
                  className="mono max-w-[180px] rounded border border-white/10 bg-black/60 px-2 py-0.5 text-[12px] text-slate-200 focus:border-cyan-400/50 focus:outline-none disabled:opacity-50"
                >
                  <option value="" className="bg-slate-900 text-slate-500">
                    {serialPorts.length ? "Select a detected port" : backendReady ? "No serial ports detected" : "Waiting for Python backend"}
                  </option>
                  {serialPorts.map((port) => (
                    <option key={port} value={port} className="bg-slate-900 text-slate-200">{port}</option>
                  ))}
                  {pipelineState.serialPort !== "--" && pipelineState.serialPort && !serialPorts.includes(pipelineState.serialPort) && (
                    <option value={pipelineState.serialPort} className="bg-slate-900 text-slate-200">{pipelineState.serialPort} · active</option>
                  )}
                </select>
                <select
                  value={baudRate}
                  onChange={(e) => setBaudRate(Number(e.target.value))}
                  className="mono rounded border border-white/10 bg-black/60 px-2 py-0.5 text-[12px] text-slate-200 focus:border-cyan-400/50 focus:outline-none"
                >
                  <option value={115200} className="bg-slate-900 text-slate-200">115200</option>
                  <option value={57600} className="bg-slate-900 text-slate-200">57600</option>
                  <option value={9600} className="bg-slate-900 text-slate-200">9600</option>
                  <option value={230400} className="bg-slate-900 text-slate-200">230400</option>
                </select>
              </div>
            </div>
            <Spec k="Status" v={<StatusDot status={transportConnected ? pipelineState.serialMode === "real" ? "ok" : "warn" : "off"} label={backendReady ? pipelineState.transportStatus : "backend not ready"} />} />
            <Spec k="Selected port" v={selectedPort || "None detected"} />
            <Spec k="Active port" v={transportConnected && pipelineState.serialPort !== "--" ? pipelineState.serialPort : "Not connected"} />
            <Spec k="Baud rate" v={pipelineState.baudRate || baudRate} />
            <Spec k="Framing" v="8N1 · newline terminated (NG1|<CMD>\n)" />
            <Spec k="Last TX" v={<span className="rounded bg-cyan-500/10 px-2 py-0.5 text-cyan-200 ring-1 ring-cyan-400/20">{pipelineState.serialMode === "mock" && pipelineState.lastTx !== "NONE" ? `SIMULATED · ${pipelineState.lastTx}` : pipelineState.lastTx}</span>} />
            <Spec k="Transport" v={backendReady ? pipelineState.transportStatus : "Backend not ready"} />
            <Spec k="Transport mode" v={pipelineState.serialMode === "real" ? "Real serial owner" : pipelineState.serialMode === "mock" ? "Mock · no hardware" : "Disabled"} />
            <Spec k={pipelineState.serialMode === "mock" ? "Simulated frames" : "Frames written"} v={String(pipelineState.framesSent)} />
            <Spec k="Pipeline TX gate" v={backendReady ? pipelineState.isTxPermitted ? "Permitted for current frame" : "No command permitted for current frame" : "Not reported"} />
            <Spec
              k="STOP safety"
              v={backendReady ? <StatusDot status={pipelineState.isArmed ? "warn" : "ok"} label={pipelineState.safetyStatus} /> : "Awaiting backend state"}
            />
            <Spec
              k="Pipeline errors"
              v={backendReady ? pipelineState.errors.length > 0 ? pipelineState.errors[pipelineState.errors.length - 1] : "None reported" : "Not reported"}
            />
            {serialStatusNote && <Spec k="Notice" v={<span className="text-cyan-300 text-[11px]">{serialStatusNote}</span>} />}
            <Spec k="Message" v={<span className="max-w-[200px] truncate text-right text-[11px]" title={pipelineState.message}>{pipelineState.message}</span>} />
          </div>

          <div className="border-t border-white/5 p-4">
            <div className="mono mb-2 text-[9px] uppercase tracking-[0.25em] text-slate-500">Manual commands · enabled only for a real serial link</div>
            <div className="grid grid-cols-5 gap-2">
              {[
                { label: "STOP", cmd: "STOP", danger: true },
                { label: "GRIP", cmd: "GRIP", danger: false },
                { label: "INDEX", cmd: "INDEX_FINGER", danger: false },
                { label: "FIST", cmd: "CLOSED_FIST", danger: false },
                { label: "REST", cmd: "REST", danger: false },
              ].map(({ label, cmd, danger }) => (
                <button
                  key={cmd}
                  disabled={!realSerialConnected || isTransmitting}
                  onClick={() => sendTestCommand(cmd)}
                  className={cn(
                    "mono rounded-lg border py-1.5 text-[10px] font-semibold uppercase tracking-wider transition disabled:opacity-50",
                    danger
                      ? "border-rose-500/30 bg-rose-500/15 text-rose-300 hover:bg-rose-500/25"
                      : "border-cyan-400/30 bg-cyan-500/10 text-cyan-200 hover:bg-cyan-500/20"
                  )}
                >
                  {label}
                </button>
              ))}
            </div>

            <div className="mono mb-2 mt-4 text-[9px] uppercase tracking-[0.25em] text-slate-500">
              STOP Safety Arming (backend-owned)
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <button
                disabled={!backendReady || busySafety}
                onClick={() => void toggleStopArm()}
                aria-pressed={backendReady && pipelineState.isArmed}
                className={cn(
                  "mono rounded-lg border py-1.5 px-3 text-[10px] font-semibold uppercase tracking-wider transition disabled:opacity-50",
                  pipelineState.isArmed
                    ? "border-amber-400/40 bg-amber-500/15 text-amber-200 hover:bg-amber-500/25"
                    : "border-cyan-400/30 bg-cyan-500/10 text-cyan-200 hover:bg-cyan-500/20"
                )}
              >
                {busySafety ? "Updating…" : !backendReady ? "Awaiting backend" : pipelineState.isArmed ? "Disarm STOP" : "Arm STOP"}
              </button>
              <span className="mono text-[11px] text-slate-400">
                {backendReady ? pipelineState.safetyStatus : "STOP arming state not reported"}
                {backendReady && pipelineState.isArmed ? " · auto-disarms on timeout" : ""}
              </span>
            </div>
            {safetyNote && <div className="mono mt-2 text-[10px] text-cyan-300">{safetyNote}</div>}
          </div>
        </GlassCard>
      </div>
    </div>
  );
}
