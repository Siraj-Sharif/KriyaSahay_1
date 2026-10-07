import { motion } from "framer-motion";
import { useState, useEffect } from "react";
import { GlassCard, Icon, StatusDot } from "../components/ui";
import { usePipelineState } from "../hooks/usePipelineState";
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
  const [busySerial, setBusySerial] = useState(false);
  const [serialPorts, setSerialPorts] = useState<string[]>([]);
  const [selectedPort, setSelectedPort] = useState<string>("COM9");
  const [baudRate, setBaudRate] = useState<number>(115200);
  const [serialStatusNote, setSerialStatusNote] = useState<string>("");
  const [isTransmitting, setIsTransmitting] = useState<boolean>(false);
  const [busySafety, setBusySafety] = useState<boolean>(false);
  const [safetyNote, setSafetyNote] = useState<string>("");

  // Load real COM ports
  const refreshPorts = async () => {
    if (typeof window !== "undefined" && (window as any).neurogrip?.serialPorts) {
      try {
        // Real COM/tty ports, listed by the running pipeline (`serial.list_ports`).
        const ports = await (window as any).neurogrip.serialPorts();
        setSerialPorts(ports || []);
        if (ports && ports.length > 0 && !ports.includes(selectedPort)) {
          if (ports.includes("COM9")) {
            setSelectedPort("COM9");
          } else {
            setSelectedPort(ports[0]);
          }
        }
      } catch {
        setSerialPorts([]);
      }
    } else {
      setSerialPorts([]);
    }
  };

  useEffect(() => {
    void refreshPorts();
  }, []);

  // Sync with active pipeline port if reported
  useEffect(() => {
    if (pipelineState.serialPort && pipelineState.serialPort !== "--") {
      setSelectedPort(pipelineState.serialPort);
    }
  }, [pipelineState.serialPort]);

  const toggleSerialReal = async () => {
    if (!window.neurogrip) return;
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
    if (!window.neurogrip?.serialSend) return;
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
        <h2 className="text-xl font-semibold text-white">Hardware Details</h2>
        <p className="text-sm text-slate-400">Physical system topology and live link diagnostics. Only verified static information shown.</p>
      </div>

      {/* Topology */}
      <GlassCard title="Signal Chain" icon={<Icon.Chip />} strong>
        <div className="flex flex-col items-stretch gap-2 p-5 md:flex-row md:items-center">
          {[
            { t: "Webcam", s: "MediaPipe Hands (Python)", ok: true, icon: <Icon.Camera className="h-5 w-5" /> },
            { t: "Vision Pipeline", s: "MediaPipe → MLP", ok: true, icon: <Icon.Eye className="h-5 w-5" /> },
            { t: "Serial Bridge", s: `${pipelineState.serialPort} @ ${pipelineState.baudRate}`, ok: pipelineState.transportStatus.includes("CONNECTED"), icon: <Icon.Serial className="h-5 w-5" /> },
            { t: "ESP32", s: "Dual-core 240 MHz", ok: true, icon: <Icon.Chip className="h-5 w-5" /> },
            { t: "PCA9685", s: "16-ch PWM · I²C 0x40", ok: true, icon: <Icon.Grid className="h-5 w-5" /> },
            { t: "Servo Hand", s: "6 × MG996R actuators", ok: true, icon: <Icon.Hand className="h-5 w-5" /> },
          ].map((n, i, arr) => (
            <div key={n.t} className="flex flex-1 items-center gap-2">
              <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: i * 0.08 }} className={cn("relative flex flex-1 flex-col items-center gap-1 rounded-xl border p-3 text-center", n.ok ? "border-cyan-400/25 bg-cyan-500/[0.06]" : "border-white/5 bg-white/[0.02]")}>
                <span className={n.ok ? "text-cyan-300" : "text-slate-500"}>{n.icon}</span>
                <span className="text-xs font-semibold text-white">{n.t}</span>
                <span className="mono text-[10px] text-slate-400">{n.s}</span>
                <span className="absolute right-2 top-2"><StatusDot status={n.ok ? "ok" : "off"} /></span>
              </motion.div>
              {i < arr.length - 1 && (
                <div className="relative hidden h-px w-6 shrink-0 bg-white/10 md:block">
                  <motion.span className="absolute top-1/2 h-1 w-1 -translate-y-1/2 rounded-full bg-cyan-400 shadow-[0_0_6px_#22d3ee]" animate={{ left: ["0%", "100%"] }} transition={{ repeat: Infinity, duration: 1.2, delay: i * 0.2, ease: "linear" }} />
                </div>
              )}
            </div>
          ))}
        </div>
      </GlassCard>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
        <GlassCard title="Microcontroller" icon={<Icon.Chip />}>
          <div className="px-5 py-2">
            <Spec k="Board" v="ESP32-WROOM-32" />
            <Spec k="Core" v="Xtensa LX6 dual-core @ 240 MHz" />
            <Spec k="Flash / SRAM" v="4 MB / 520 KB" />
            <Spec k="Interface" v="USB-UART (CP2102)" />
            <Spec k="Protocol" v="NG1|<COMMAND>\\n" />
            <Spec k="Firmware" v="neurogrip-fw (reported by device)" />
            <Spec k="Uptime" v="Not reported by firmware" />
          </div>
        </GlassCard>

        <GlassCard title="Servo Driver · PCA9685" icon={<Icon.Grid />}>
          <div className="px-5 py-2">
            <Spec k="Driver" v="PCA9685 16-ch PWM" />
            <Spec k="Bus" v="I²C · SDA 21 / SCL 22" />
            <Spec k="Address" v="0x40" />
            <Spec k="PWM freq" v="50 Hz" />
            <Spec k="Pulse range" v="500–2500 µs" />
            <Spec k="Power" v="5 V / 3 A external (per design)" />
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

        <GlassCard title="Robotic Hand Servos" icon={<Icon.Hand />}>
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
                <StatusDot status={pipelineState.serialConnected ? "ok" : "off"} label={pipelineState.serialConnected ? "Linked" : "Unlinked"} />
              </div>
            ))}
            <p className="mt-1 text-[11px] text-slate-500">Metal-gear high-torque servos driving tendon-actuated fingers. Torque: 11 kg·cm @ 6 V.</p>
          </div>
        </GlassCard>
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <GlassCard title="Camera" icon={<Icon.Camera />}>
          <div className="px-5 py-2">
            <Spec k="Model" v="MediaPipe Hands (21 landmarks)" />
            <Spec k="Capture" v="Python OpenCV (30 fps target)" />
            <Spec k="Resolution" v="1280 × 720 (configurable)" />
            <Spec k="Status" v={<StatusDot status={pipelineState.cameraStatus === "connected" ? "ok" : "warn"} label={pipelineState.cameraStatus} />} />
            <Spec k="Classifier" v="NeuroGrip Hybrid (ML + Rule-based)" />
            <Spec k="Inference" v={pipelineState.latencyMs ? `${pipelineState.latencyMs} ms` : "--"} />
          </div>
        </GlassCard>

        <GlassCard
          title="Serial Communication"
          icon={<Icon.Serial />}
          right={
            <div className="flex items-center gap-2">
              <button
                onClick={refreshPorts}
                title="Rescan COM ports"
                className="mono rounded-md border border-white/10 bg-white/[0.03] px-2 py-1 text-[10px] text-slate-300 hover:border-white/20"
              >
                Rescan
              </button>
              <button
                onClick={toggleSerialReal}
                disabled={busySerial}
                className={cn(
                  "mono rounded-md px-2.5 py-1 text-[10px] uppercase tracking-wider ring-1 transition disabled:opacity-50",
                  pipelineState.serialConnected
                    ? "bg-rose-500/10 text-rose-300 ring-rose-400/30 hover:bg-rose-500/20"
                    : "bg-emerald-500/10 text-emerald-300 ring-emerald-400/30 hover:bg-emerald-500/20",
                )}
              >
                {busySerial ? "…" : pipelineState.serialConnected ? "Disconnect" : "Connect"}
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
                  onChange={(e) => setSelectedPort(e.target.value)}
                  className="mono rounded border border-white/10 bg-black/60 px-2 py-0.5 text-[12px] text-slate-200 focus:border-cyan-400/50 focus:outline-none"
                >
                  {serialPorts.length > 0 ? (
                    serialPorts.map((p) => (
                      <option key={p} value={p} className="bg-slate-900 text-slate-200">
                        {p}
                      </option>
                    ))
                  ) : (
                    <option value="COM9" className="bg-slate-900 text-slate-200">COM9</option>
                  )}
                  {!serialPorts.includes("COM9") && <option value="COM9" className="bg-slate-900 text-slate-200">COM9 (default)</option>}
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
            <Spec k="Status" v={<StatusDot status={pipelineState.serialConnected ? "ok" : "off"} label={pipelineState.transportStatus} />} />
            <Spec k="Active Port" v={pipelineState.serialPort !== "--" ? pipelineState.serialPort : selectedPort} />
            <Spec k="Baud rate" v={pipelineState.baudRate || baudRate} />
            <Spec k="Framing" v="8N1 · newline terminated (NG1|<CMD>\n)" />
            <Spec k="Last TX" v={<span className="rounded bg-cyan-500/10 px-2 py-0.5 text-cyan-200 ring-1 ring-cyan-400/20">{pipelineState.lastTx}</span>} />
            <Spec k="Transport" v={pipelineState.transportStatus} />
            <Spec
              k="Serial mode"
              v={
                pipelineState.serialMode === "real"
                  ? "Real hardware"
                  : pipelineState.serialMode === "mock"
                    ? "Mock (no hardware)"
                    : "Disabled"
              }
            />
            <Spec k="Frames sent" v={String(pipelineState.framesSent)} />
            <Spec
              k="Safety"
              v={
                <StatusDot
                  status={pipelineState.isArmed ? "warn" : "ok"}
                  label={pipelineState.safetyStatus}
                />
              }
            />
            <Spec
              k="Pipeline errors"
              v={pipelineState.errors.length > 0 ? pipelineState.errors[pipelineState.errors.length - 1] : "None reported"}
            />
            {serialStatusNote && <Spec k="Notice" v={<span className="text-cyan-300 text-[11px]">{serialStatusNote}</span>} />}
            <Spec k="Message" v={<span className="max-w-[200px] truncate text-right text-[11px]" title={pipelineState.message}>{pipelineState.message}</span>} />
          </div>

          <div className="border-t border-white/5 p-4">
            <div className="mono mb-2 text-[9px] uppercase tracking-[0.25em] text-slate-500">Test Command Dispatch (NG1 Protocol)</div>
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
                  disabled={isTransmitting}
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
                disabled={busySafety}
                onClick={() => void toggleStopArm()}
                aria-pressed={pipelineState.isArmed}
                className={cn(
                  "mono rounded-lg border py-1.5 px-3 text-[10px] font-semibold uppercase tracking-wider transition disabled:opacity-50",
                  pipelineState.isArmed
                    ? "border-amber-400/40 bg-amber-500/15 text-amber-200 hover:bg-amber-500/25"
                    : "border-cyan-400/30 bg-cyan-500/10 text-cyan-200 hover:bg-cyan-500/20"
                )}
              >
                {pipelineState.isArmed ? "Disarm STOP" : "Arm STOP"}
              </button>
              <span className="mono text-[11px] text-slate-400">
                {pipelineState.safetyStatus}
                {pipelineState.isArmed ? " · auto-disarms on timeout" : ""}
              </span>
            </div>
            {safetyNote && <div className="mono mt-2 text-[10px] text-cyan-300">{safetyNote}</div>}
          </div>
        </GlassCard>
      </div>
    </div>
  );
}
