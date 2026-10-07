import { useEffect, useState, useSyncExternalStore } from "react";
import { getTtsSettings, getVoicesNow, subscribeTts, ttsSupported } from "../services/tts";

/** Live TTS settings + the list of available English system voices. */
export function useTts() {
  const settings = useSyncExternalStore(subscribeTts, getTtsSettings);
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>(() => getVoicesNow());

  useEffect(() => {
    if (!ttsSupported()) return;
    const update = () => setVoices(getVoicesNow());
    update();
    window.speechSynthesis.addEventListener("voiceschanged", update);
    return () => window.speechSynthesis.removeEventListener("voiceschanged", update);
  }, []);

  return { settings, voices, supported: ttsSupported() };
}
