# NeuroGrip — Phase 2.1 Hardware Test Checklist

Date: 2026-10-06
Branch: `arena/e7b2eaba-neurogrip-new` (Phase 2 at `d62b217`, Phase 2.1 committed on top)

This checklist is the **physical** verification procedure that must be executed on the real
machine (Windows PC + webcam + ESP32 + microphone). Nothing in this document has been
executed in the Arena sandbox: there is no camera, no COM port, no ESP32 and no microphone
here. Everything below is derived from the code paths that the automated suites *do* cover,
so a step failing here means the physical layer disagrees with a behaviour that is already
verified in software.

## 0. Preconditions and start-up

| # | Step | Expected result | Failure observation to record |
|---|---|---|---|
| P1 | On the target PC, run `start_neurogrip.ps1` (or `start_neurogrip.bat`) from the real project directory (the launchers hardcode `D:\NeuroGrip_Project_New`). | Electron window opens; launcher starts **one** Python pipeline (`python -m neurogrip.app`). | Launcher path error, or more than one `python.exe` holding the camera/serial. |
| P2 | In the UI, confirm the Navbar "Backend Live" dot. | Dot is live; telemetry shows real values, not `Waiting for pipeline...`. | Dot off while the pipeline console is printing frames → bridge port (8765) blocked or another process owns it. |
| P3 | Open `Hardware → Signal Chain → Serial`. | `Status`, `Active Port`, `Baud rate`, `Last TX`, `Frames sent` all come from the pipeline (they change when the pipeline transmits, not when the UI is clicked). | Values that never change, or that change on a click without a transmission. |

## 1. CAMERA (real webcam)

| # | Step | Expected result | Failure observation to record |
|---|---|---|---|
| C1 | Click the camera source picker → **Rescan**. | The list is produced by the pipeline's own `camera.list` probe; unavailable indices are not offered; the configured index is always offered. | A device listed that cannot be opened; or no device listed while `cv2.VideoCapture(0)` works in a Python shell. |
| C2 | Select the real webcam (desktop mode addresses cameras by **numeric index**). | Preview switches to that device; `Hardware → Camera → Index` shows the **same index the pipeline actually holds**. | Preview unchanged; or the UI index differs from the camera the pipeline opened. Note: if the requested index cannot be opened, the OpenCV backend may fall back to another working index — the reported `index` is then the fallback, so cross-check the picture. |
| C3 | Confirm resolution and that the picture is live (wave your hand). | `Resolution` shows what the driver reports (falls back to the requested target if the driver reports 0×0); the picture moves. | A frozen frame, or a resolution that does not match the driver. |
| C4 | Confirm FPS. | FPS is the pipeline's measured frame rate (rolling mean of real frame intervals), not a hardcoded 30. It should drop when the CPU is loaded and recover. | A constant FPS while processing clearly changes speed. |
| C5 | Check the camera LED / close-and-reopen test: click **Off** (camera on/off control), then try to open the same camera in another application. | The device is genuinely released (`camera_info.opened == false`, LED off) and the other application can open it. | LED stays on, or the second application reports "device in use" → the pipeline did not release the handle. |
| C6 | While the camera is **Off**, check the Dashboard. | `CAMERA OFF` is shown, FPS = 0, no gesture is ever produced, `Frames sent` does not move, and the camera preview stops (no fake frames). | A preview that keeps updating, or FPS that stays non-zero. |
| C7 | Turn the camera back **On**. | Capture reopens (same index), preview resumes, telemetry FPS recovers. | Camera stays off, or the error banner remains after a successful reopen. |
| C8 | Perform OPEN (four fingers), CLOSE (fist), PINCH/GRIP, POINT (index only) in front of the camera. | The Dashboard gesture readout changes with the gesture, and **exactly one** command per gesture change reaches the hardware (servos move once; `Frames sent` increases by one per stabilised command, not per frame). | Repeated identical frames per gesture, or a gesture shown in the UI that never reaches the ESP32. |
| C9 | Hold a STOP pose (open palm, thumb spread wide) **while armed** and again **while disarmed** (see §4 safety steps). | Both cases transmit `NG1|STOP` — STOP is deliberately always available (frozen semantics). | STOP recognised but never transmitted → safety-critical, stop testing and report. |
| C10 | Unplug the webcam while it is streaming, then try to turn capture on again. | An error banner appears with the pipeline's real message and `pipeline_state = CAMERA_OFF`/error; the UI never claims the camera is connected. Recovery requires re-selecting/re-enabling the camera (no automatic retry — see limitations). | The UI keeps showing a live-looking camera, or silently invents frames. |

## 2. SERIAL / ESP32

| # | Step | Expected result | Failure observation to record |
|---|---|---|---|
| S1 | Connect the ESP32 over USB. Click **Rescan ports**. | The list comes from `serial.list_ports` (pyserial inside the pipeline) and shows the real COM port (launchers default to **COM9**, `pc/config/default.yaml` defaults to COM3). | Ports listed that do not exist; real port missing. |
| S2 | Select the port and baud (115200 unless your firmware differs) and press **Connect**. | The button reads connected **only when the pipeline really holds the link** (`serial_info.connected == true`, `Serial mode = Real hardware`). | "Connected" while the port was never opened, or an error is swallowed. |
| S3 | In a separate shell, confirm the port is owned once: `mode COM9` / Device Manager / `python -c "import serial; serial.Serial('COM9')"` must **fail** while the pipeline holds it. | The second open fails → exactly one serial owner (the pipeline). | A second open succeeds → a duplicate serial path exists; report immediately. |
| S4 | Press **CALL** in *Test Command Dispatch*. | `NG1|CALL` is transmitted; firmware performs the CALL action; `Last TX` = `NG1|CALL`; `Frames sent` +1. | Nothing happens on the device; UI claims success (must not happen). |
| S5 | Press **OK**. | `NG1|OK` transmitted, device reacts, telemetry matches. | As above. |
| S6 | Press **GRIP**, **INDEX** (INDEX_FINGER), **FIST** (CLOSED_FIST). | Each transmits exactly one frame and moves the servos; `REST` transmits nothing (REST is idle-only and is rejected by the validator). | REST appearing on the wire; two frames for one press. |
| S7 | Press **STOP** three times in a row. | Three `NG1|STOP` frames — STOP is never de-duplicated (this is intentional: the operator must be able to repeat it). | STOP suppressed as a duplicate → safety-critical, report. |
| S8 | With the link connected, say/type a voice command (see §3) and simultaneously show the same gesture to the camera. | Exactly one frame for the utterance and the gesture is not re-sent after the pipeline already transmitted it (no renderer-side duplicate). | Two frames for one event. |
| S9 | Press **Disconnect**. | `serial_info.connected == false`; the UI shows the real port/baud but not "linked"; `Frames sent` stops increasing. | UI still claims connected. |
| S10 | With the link disconnected, press **CALL**. | The request fails visibly: a red/notice line with the pipeline's error (`serial link unavailable`) and **no** frame on the wire (`Last TX` unchanged, `Frames sent` unchanged). | A success message, or a frame arriving on the ESP32. |
| S11 | Press **Connect** again on the same port. | Reconnects; `Last TX` shows `NONE` for the fresh link; `Frames sent` restarts from 0; the next command transmits normally. | A stale `Last TX` from the previous link being presented as current. |
| S12 | Swap to a different COM port (or unplug/replug so Windows gives a new port) and connect. | The pipeline builds the new interface, releases the old one, and reports the new port; a command sent immediately after works. | Two ports open at once, or commands going to the old port. |
| S13 | While connected, issue a command that is not in the taxonomy (e.g. type `GRABBING` in a test path). | Rejected with a real error; nothing transmitted. | Any frame on the wire. |
| S14 | Leave everything connected for ~10 minutes with the camera running. | No duplicate writers, no unexplained transmissions, no `Frames sent` jumps, no UI freeze; the ESP32 receives only what the telemetry shows. | Any divergence between telemetry and the device. |

## 3. VOICE (real microphone)

Note: the spoken vocabulary is the frozen `intents` list. `open`, `release`, `palm`, `relax`
and `freeze` all map to **STOP**; `pinch` is **not** a recognised phrase and must produce no
command. Do not "fix" this during the test — record it.

| # | Step | Expected result | Failure observation to record |
|---|---|---|---|
| V1 | Grant microphone access, pick the real mic, press the mic button. | State becomes listening; the level meter follows your voice. | Permission silently denied, or a level meter that moves with no audio. |
| V2 | Say **"call"**. | Transcript appears; the pipeline reports intent `gesture` → `CALL`; **exactly one** `NG1|CALL` reaches the ESP32; the voice reply names CALL. | A transcript with no command; two commands for one utterance; a frame for a failed request. |
| V3 | Say **"okay"**. | One `NG1|OK` on the wire. | As above. |
| V4 | Say **"stop"**. | One `NG1|STOP`; the firmware stops. | As above. |
| V5 | Say **"close"**. | One `NG1|CLOSED_FIST`. | As above. |
| V6 | Say **"point"**. | One `NG1|INDEX_FINGER`. | As above. |
| V7 | Say **"open"**. | One `NG1|STOP` (this is the frozen vocabulary: "open" means release/stop, not a grip). | A different gesture, or any behaviour that differs from the documented mapping. |
| V8 | Say **"pinch"**, and then say something meaningless ("make me a sandwich"). | No command, no transmission; the voice log answers with the pipeline's reply/`Sorry, I don't have a command for that`. | Any frame on the wire for an unsupported phrase. |
| V9 | Disconnect the ESP32, then say **"call"**. | The command is refused with the real serial error shown in the voice log; `Frames sent` unchanged. | A reply that implies success. |
| V10 | Say the same command twice in a row ("close", "close"). | Two frames — an explicit voice request is an intentional retry, not a duplicate. | Second utterance suppressed. |

## 4. SAFETY (STOP arming / STOP transmission)

Semantics to verify (frozen — do not change during the test): the software arm flag
(`stop.set_armed`) is real backend state, is displayed by the UI, and **never gates STOP** —
STOP is available and transmissible whether armed or disarmed. The flag auto-clears after
`stop.arm_timeout_ms` (5 s by default) and does not itself move hardware.

| # | Step | Expected result | Failure observation to record |
|---|---|---|---|
| K1 | Open `Hardware`, find the **Safety** row. | It shows the backend's own state (`STOP DISARMED` / `STOP ARMED`) with a status dot; it never claims a state the pipeline does not have. | A hardcoded label, or a label that does not change when the backend changes. |
| K2 | Press **Arm STOP**. | The row switches to `STOP ARMED` (both the label and the dot), and the notice line says the pipeline reports ARMED. | A local-only flip with no round-trip; or an error hidden. |
| K3 | Wait ~6 s without touching anything. | The pipeline auto-disarms (`STOP DISARMED`) and the UI follows on its own. | UI stuck on ARMED → it is holding a local copy. |
| K4 | Press **Arm STOP** and then immediately **Disarm STOP**. | State flips back immediately; no queueing, no toggle ambiguity. | A toggle that ends in the wrong state. |
| K5 | Stop/kill the Python pipeline, then look at the Safety row. | The UI shows the backend as gone (no live state) and does not display "ARMED" from a local variable. | The UI keeps claiming an arming state with no backend. |
| K6 | Arm STOP, then press the manual **STOP** button. | `NG1|STOP` is transmitted once and the firmware stops. | Any suppression of STOP. |
| K7 | Disarm STOP, then perform the STOP gesture in front of the camera. | `NG1|STOP` still reaches the hardware (fail-safe semantics) while the UI reports DISARMED. | STOP blocked while disarmed, or the UI claiming ARMED. |
| K8 | During a STOP scenario, watch `Frames sent` and `Last TX`. | One STOP frame per STOP event; telemetry matches the device; no second frame appears later. | Extra STOP frames. |
| K9 | Restart the pipeline while the UI is running. | The UI re-attaches to the new pipeline and shows its real state (arming back to the default DISARMED). | Stale telemetry from the previous process. |

## 5. Evidence to collect

For any failure, capture: the UI screenshot, the Python console lines around the event
(`[CONTROL TX]`, `[MOCK SERIAL TX]`, validator messages), the exact `Last TX` / `Frames sent`
values before and after, and whether the ESP32 physically reacted. The report's §21 states
exactly what is verified in software — a physical mismatch is a new finding, not a known one.
