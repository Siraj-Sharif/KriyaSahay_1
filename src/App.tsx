import { AnimatePresence, motion } from "framer-motion";
import { useState } from "react";
import { Navbar, type Tab } from "./components/Navbar";
import { useFullscreen } from "./hooks/useFullscreen";
import { Dashboard } from "./pages/Dashboard";
import { GestureGuide } from "./pages/GestureGuide";
import { Hardware } from "./pages/Hardware";
import { VoiceAssistant } from "./pages/VoiceAssistant";
import { SystemProvider, useSystem } from "./services/SystemContext";

function Footer({ compact }: { compact?: boolean }) {
  const { frame, serial } = useSystem();
  return (
    <footer className={compact ? "shrink-0 border-t border-white/5 py-2" : "mt-8 border-t border-white/5 py-4"}>
      <div className="mx-auto flex max-w-[1700px] flex-wrap items-center justify-between gap-3 px-4 sm:px-6">
        <div className="mono flex flex-wrap items-center gap-x-5 gap-y-1 text-[10px] uppercase tracking-[0.2em] text-slate-500">
          <span>NeuroGrip v1.0</span>
          <span>CMD <span className="text-cyan-300">{frame?.handDetected ? frame.gesture : "—"}</span></span>
          <span>FPS <span className="text-slate-300">{frame?.fps ?? "--"}</span></span>
          <span>LAT <span className="text-slate-300">{frame?.latencyMs ?? "--"}ms</span></span>
          <span>TX <span className="text-slate-300">{serial?.lastCommand ?? "--"}</span></span>
        </div>
        <span className="mono text-[10px] text-slate-600">ESP32 · PCA9685 · MediaPipe · Press F11 for fullscreen</span>
      </div>
    </footer>
  );
}

function Shell() {
  const [tab, setTab] = useState<Tab>("dashboard");
  const { isFullscreen, toggle } = useFullscreen();
  // Voice Assistant is a fixed, single-window screen (no page scroll on desktop-sized windows)
  const fixed = tab === "voice";

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.6 }} className="flex h-screen flex-col overflow-hidden">
      <Navbar tab={tab} onTab={setTab} isFullscreen={isFullscreen} onToggleFullscreen={() => void toggle()} />

      <div className={fixed ? "flex min-h-0 flex-1 flex-col overflow-y-auto lg:overflow-hidden" : "min-h-0 flex-1 overflow-y-auto"}>
        <main className={fixed ? "mx-auto min-h-0 w-full max-w-[1700px] flex-1 px-4 py-3 sm:px-6" : "mx-auto max-w-[1700px] px-4 py-5 sm:px-6"}>
          <AnimatePresence mode="wait">
            <motion.div
              key={tab}
              className={fixed ? "h-full min-h-0" : undefined}
              initial={{ opacity: 0, y: 14, filter: "blur(6px)" }}
              animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
              exit={{ opacity: 0, y: -10, filter: "blur(6px)" }}
              transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
            >
              {tab === "dashboard" && <Dashboard />}
              {tab === "gestures" && <GestureGuide />}
              {tab === "hardware" && <Hardware />}
              {tab === "voice" && <VoiceAssistant />}
            </motion.div>
          </AnimatePresence>
        </main>
        <Footer compact={fixed} />
      </div>

      <AnimatePresence>
        {isFullscreen && (
          <motion.div
            initial={{ opacity: 0, scale: 1.1 }}
            animate={{ opacity: [0, 1, 0], scale: 1 }}
            transition={{ duration: 1.4, times: [0, 0.3, 1] }}
            className="pointer-events-none fixed inset-0 z-50 flex items-center justify-center"
          >
            <div className="rounded-2xl border border-cyan-400/30 bg-black/70 px-6 py-3 backdrop-blur">
              <span className="mono text-xs uppercase tracking-[0.3em] text-cyan-200">Fullscreen mode engaged</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

export default function App() {
  return (
    <SystemProvider>
      <Shell />
    </SystemProvider>
  );
}
