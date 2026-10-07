import { useEffect, useState } from "react";
import type { DesktopRuntimeStatus } from "../types/desktop";

/** Runtime lifecycle state reported by Electron; absent in a plain browser preview. */
export function useDesktopRuntime(): DesktopRuntimeStatus | null {
  const [status, setStatus] = useState<DesktopRuntimeStatus | null>(null);

  useEffect(() => {
    const bridge = window.neurogrip;
    if (!bridge?.getRuntimeStatus || !bridge.onRuntimeStatus) return;

    let mounted = true;
    let receivedEvent = false;
    const unsubscribe = bridge.onRuntimeStatus((next) => {
      receivedEvent = true;
      if (mounted) setStatus(next);
    });

    void bridge.getRuntimeStatus().then((initial) => {
      // Don't let a slower IPC snapshot overwrite a newer pushed lifecycle event.
      if (mounted && !receivedEvent) setStatus(initial);
    }).catch(() => {
      // The live event channel may still recover; leave readiness unknown meanwhile.
    });

    return () => {
      mounted = false;
      unsubscribe();
    };
  }, []);

  return status;
}
