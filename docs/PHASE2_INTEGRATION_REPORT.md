# NeuroGrip Phase 2 — Real UI ↔ Backend Integration Report

Date: 2026-10-06
Branch: `arena/e7b2eaba-neurogrip-new` (base `bc082db`, Phase 1 at `3cb709d`)
Scope: make the existing UI genuinely control the **running Python pipeline**. No visual redesign.

Verification classes used throughout:

| Class | Meaning |
|---|---|
| **1** | REAL + VERIFIED — automated evidence covers the whole path |
| **2** | REAL, CODE-VERIFIED, HARDWARE TEST REQUIRED — path traced and unit/integration tested, but needs the physical camera / ESP32 / mic |
| **3** | NOT YET INTEGRATED |
| **4** | BROKEN |
| **5** | INTENTIONALLY DISPLAY-ONLY |

---

## 1. Architecture — final communication flow

```
┌──────────────────────────────── React renderer (src/) ────────────────────────────────┐
│  pages / components                                                                   │
│    ├── usePipelineState()  ──► pipelineStore ──► pipelineSource ──┐  (ONE IPC sub)     │
│    ├── useGestureCommand() ──► commandEmitter ──► SerialAdapter ──┼───────────────┐    │
│    ├── SystemContext (camera on/off, camera select, voice) ───────┤               │    │
│    └── pipelineControl.ts (typed control helpers) ────────────────┘               │    │
└───────────────────────────────────────────────────────────────────────────────────┼────┘
                                    │ window.neurogrip.*  (preload.cjs, contextIsolation)
                                    ▼
┌──────────────────────────── Electron main (electron/) ────────────────────────────────┐
│  main.cjs        IPC: onPipelineState/onCameraFrame/onBridgeStatus/control/serial:*    │
│  bridge.cjs      ONE TCP server on 127.0.0.1:8765  ──► accepts the Python pipeline     │
│                    Python → Electron : state snapshots, JPEG frames, control_results   │
│                    Electron → Python : control requests (camera/serial/command/...)    │
│  stt.cjs         Whisper (on-device) — speech-to-text only, never touches hardware     │
└───────────────────────────────────────────────────────────────────────────────────────┘
                                    ▲ ▼  newline-delimited JSON
┌────────────────────── Python pipeline process (pc/src/neurogrip/) ─────────────────────┐
│  app/cli.py → app/pipeline.py  (THE single owner of camera + serial)                   │
│    bridge/tcp_bridge.py  reader thread → bridge/control.py → app/control.py handlers   │
│    camera → detector → features(68-D) → recognizer → stabilizer → validator            │
│    validator → protocol (NG1|<CMD>\n) → communication/serial_port.py → ESP32           │
│    voice/intents.py  (transcript → intent → SAME validator + SAME serial owner)         │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

Key guarantees:

* **One pipeline, one camera owner, one serial owner.** Electron no longer spawns any Python
  process (`grep spawn electron/*.cjs` → only a comment). Every serial write goes through
  `NeuroGripPipeline.serial`.
* **One persistent connection.** State and control share the 8765 socket; the UI cannot act
  on a process other than the one producing the telemetry it displays.
* **One command path.** Detection, manual buttons, manual override and voice all end in
  `NeuroGripPipeline.dispatch_command()` / the detection branch → validator → encoder →
  the single serial interface.

Wire contract (unchanged framing, new direction):

```
Electron → Python   {"type":"control","id":"ui-7","action":"serial.connect","payload":{...}}
Python → Electron   {"type":"control_result","id":"ui-7","ok":true,"data":{...},"error":null}
```

Actions: `camera.list`, `camera.select`, `camera.set_enabled`, `serial.list_ports`,
`serial.connect`, `serial.disconnect`, `command.send`, `stop.set_armed`, `voice.set_state`,
`voice.transcript`, `pipeline.snapshot`.

---

## 2. Complete UI element → backend integration table

| UI element (file) | Action | Backend path | Class |
|---|---|---|---|
| Navbar tab bar *(`Navbar.tsx`)* | switch page | client-side only | 5 |
| Navbar "Backend Live" dot | read state | `bridge-status` (now emitted on socket **accept**) | 1 |
| Navbar hand/gesture dot | read state | `VisualizationState.command/hand_count` | 1 |
| Navbar serial dot | read state | `serial_info.connected` + `serial_info.mode` | 1 |
| Navbar fullscreen button | window state | Electron/BrowserWindow | 1 |
| Camera source picker *(`CameraSelector.tsx`)* | `camera.list` / `camera.select` | pipeline re-opens the real device | 1 (mock camera) / 2 (real camera) |
| "Rescan" in camera picker | `camera.list` | pipeline probes indices via `cv2.VideoCapture` | 1 / 2 |
| Live vision feed *(`CameraPanel.tsx`)* | read state | real JPEG from the Python camera (`send_frame`), throttled 10 Hz | 1 / 2 |
| Landmark overlay *(`CameraPanel.tsx`)* | read state | real MediaPipe points now serialised (`landmarks`) | 1 / 2 |
| Camera off / error banner | read state | `camera_info.error`, `pipeline_state=CAMERA_OFF` | 1 |
| Camera on/off (context) | `camera.set_enabled` | pipeline closes/reopens the capture | 1 / 2 |
| Detected gesture readout | read state | `command` / `hand_count` / `stabilizer_state` | 1 |
| Confidence bar + history | read state | `confidence`, `latency_ms` | 1 |
| FPS / latency HUD | read state | `fps`, `latency_ms` | 1 |
| Pipeline status (`Pipeline` row) | read state | `pipeline_state` (now includes `CAMERA_OFF`) | 1 |
| Telemetry panel *(`TelemetryPanel.tsx`)* | read state | full `VisualizationState` (10 previously-dropped fields now sent: `is_armed`, `is_tx_permitted`, `final_command`, `is_stabilized`, `stabilizer_state`, `stop_rule_*`, `hagrid_*`, `taxonomy_mapped`) | 1 |
| Manual gesture buttons *(`Dashboard.tsx`, 13 poses incl. STOP)* | `command.send` (source `override`) | validator → encoder → serial owner | 1 (mock) / 2 (ESP32) |
| "Release" override button | clears override | preview only; no command | 1 |
| Gesture Guide cards *(`GestureGuide.tsx`)* | local preview | 3D hand only, never transmits | 5 |
| STOP button (manual grid) | `command.send STOP` | same validator → `NG1\|STOP` (never de-duplicated) | 1 / 2 |
| STOP arming | `stop.set_armed` (available) | **no dedicated UI control exists**; backend state is reported | 3 |
| Safety row (telemetry/hardware) | read state | `safety_status` + `is_armed` from the pipeline | 1 |
| Hardware: Rescan ports | `serial.list_ports` | `pyserial.comports()` inside the pipeline | 1 |
| Hardware: COM port select | `serial.connect {port}` | pipeline rebuilds + reconnects its own interface | 1 / 2 |
| Hardware: baud select | `serial.connect {baud}` | same | 1 / 2 |
| Hardware: Connect | `serial.connect {mode:"real"}` | real open; failure is reported, no fallback | 2 |
| Hardware: Disconnect | `serial.disconnect` | real close; pipeline then refuses to transmit | 1 / 2 |
| Hardware: status/active port/baud/last TX | read state | `serial_info` (mode/port/baud/connected/frames_sent/last_tx) | 1 |
| Hardware: "Test Command Dispatch" buttons | `command.send` | same single path (CALL/OK included) | 1 / 2 |
| Hardware: transport / mock-vs-real row | read state | `serial_info.mode` | 1 |
| Voice: mic button *(`VoiceAssistant.tsx`)* | start/stop capture | local capture; state reported via `voice.set_state` | 2 |
| Voice: mic picker *(`MicSelector.tsx`)* | local device | browser/Electron capture device | 2 |
| Voice: transcript | `voice.transcript` | intent parsed **inside the pipeline** (`voice/intents.py`) | 1 (routing) / 2 (real mic) |
| Voice: command execution | same `command.send` path | validator → encoder → serial owner | 1 / 2 |
| Voice: reply / status intent | backend reply | `status_summary()` from real camera+serial state | 1 |
| Voice: typed text + chips | `voice.transcript` | same as speech | 1 |
| Voice output (TTS) toggle/volume | local | browser/OS speech synthesis | 5 |
| Voice: engine status | read state | Electron Whisper engine (`stt:*`) | 2 |
| "Start / stop / restart pipeline" | — | owned by the launcher scripts, not the UI | 3 |
| 3D hand, charts, servo/spec cards | display | static diagrams + real state where noted | 5 |

---

## 3. Files changed (18)

Backend (Python):

* `pc/src/neurogrip/app/pipeline.py` — runtime control surface (camera/serial/command/STOP/voice),
  `_control_lock`, camera-off + invalid-frame state publishing, real `last_tx` telemetry,
  wire-name reporting, one shared `_transport_status_str()`.
* `pc/src/neurogrip/bridge/tcp_bridge.py` — bidirectional bridge: reader thread, control
  dispatch, `dataclasses.fields`-based state serialisation (no more silently dropped fields),
  real landmarks, throttled preview frames, reconnect.
* `pc/src/neurogrip/communication/serial_port.py` — `list_serial_ports()`.

Electron:

* `electron/main.cjs` — uses `bridge.cjs`; new `neurogrip:control` channel; `serial:*`
  handlers re-implemented as control requests (no `python -c`, no second serial owner);
  `bridge-status` emitted on accept.
* `electron/preload.cjs` — additive `control()` + `onControlResult()`; `serialSend(cmd, source)`.

Frontend:

* `src/services/pipelineControl.ts` **(new)** — typed control channel + action vocabulary.
* `src/services/pipelineStore.ts` — derives camera/serial/voice/error/landmark state from the
  pipeline instead of hardcoded values.
* `src/services/pipelineMapping.ts` — raw `camera_info`/`serial_info`/`voice_info` types,
  real landmark parsing.
* `src/services/desktopSerial.ts` — `serial_info` as the source of truth; source tag passed
  through to the pipeline.
* `src/services/desktopVision.ts` — real landmarks, preview-frame tag preserved.
* `src/services/SystemContext.tsx` — camera on/off + camera select + voice interpretation now
  go to the pipeline; camera state mirrors backend reality.
* `src/services/localWhisperVoice.ts` — optional `interpret` hook (pipeline-owned intent).
* `src/services/adapters.ts` — additive `CommandSourceTag` + `VoiceInterpreter` types.
* `src/hooks/useCameraDevices.ts` — desktop cameras come from `camera.list` (real indices).
* `src/hooks/useGestureCommand.ts` — forwards the command source.
* `src/components/CameraPanel.tsx` — real landmark overlay + real camera-off/error states.
* `src/components/Navbar.tsx` — serial indicator from `serial_info`.
* `src/pages/Hardware.tsx` — connect/disconnect/test buttons report real pipeline results.
* `src/types/desktop.d.ts` — additive control-channel typings.

---

## 4. Files added (9)

| File | Purpose |
|---|---|
| `pc/src/neurogrip/bridge/control.py` | control protocol: action vocabulary, parse/encode, `ControlDispatcher` (never raises) |
| `pc/src/neurogrip/app/control.py` | binds every action to a method on the live pipeline |
| `pc/src/neurogrip/voice/__init__.py`, `voice/intents.py` | voice intent decided inside the pipeline (mirror of the renderer's `intents.ts`) |
| `pc/src/neurogrip/camera/devices.py` | best-effort camera name for a device index |
| `electron/bridge.cjs` | testable Electron side of the bridge (no Electron import) |
| `pc/tests/test_control_channel.py` | 44 integration tests incl. cross-language contract |
| `scripts/verify_frontend_phase2.mjs` | 42 frontend control-channel checks |
| `scripts/verify_bridge_control.mjs` | 15 Electron bridge checks |
| `docs/PHASE2_INTEGRATION_REPORT.md` | this report |

---

## 5. Persistent control-channel design

* **Transport:** the existing 8765 TCP socket. Python dials Electron (unchanged), so a UI
  reload never disturbs the pipeline.
* **Framing:** one JSON object per line, UTF-8, `\n` terminated. Requests carry a client-side
  `id`; results echo it. State snapshots and preview frames share the socket (already the case).
* **Node side (`electron/bridge.cjs`):** `request(action, payload)` → writes one line, resolves
  on the matching `control_result`. Failure modes all resolve: no pipeline → immediate
  `{ok:false}`, no answer → timeout (`CONTROL_TIMEOUT_MS=6000`), socket error / disconnect →
  every pending request is failed with a real message, uncorrelated results are surfaced.
* **Python side:** a daemon reader thread frames lines, `parse_request` validates the action
  against the vocabulary, `ControlDispatcher` routes to a handler and always answers with
  `{ok,data,error}` — a broken handler can never take the pipeline down.
* **Back-pressure:** nothing blocks the CV loop. Control handlers run on the reader thread and
  mutate pipeline state under `RLock`; state/frames are throttled to 10 Hz.
* **No hidden second owner:** `configure_serial()` builds and connects the *new* interface
  before releasing the old one, so at no point do two interfaces hold the same port.
* **Security model unchanged:** `contextIsolation: true`, `nodeIntegration: false`,
  `sandbox: true`, allow-listed permissions, no navigation, `control()` is a bounded
  action vocabulary — the renderer still cannot reach the OS directly.

---

## 6. Camera integration status

| Capability | Status | Evidence |
|---|---|---|
| Real frames on screen | REAL | pipeline `send_frame` (JPEG, 10 Hz) → `neurogrip:frame` → `<img>`; verified with the mock camera |
| Start / stop capture | REAL | `camera.set_enabled` closes/reopens the real capture; `pipeline_state=CAMERA_OFF` while stopped |
| Enabled state truthfulness | REAL | `camera_info.enabled/opened`; UI mirrors it (a camera stopped elsewhere shows stopped) |
| Device enumeration | REAL | `camera.list` probes indices with `cv2.VideoCapture` and hides unavailable ones |
| Device switching | REAL | `camera.select` closes the old device, opens the new one, reports failure |
| Resolution / error reporting | REAL | `camera_info.resolution/error` shown in the panel |
| Reconnect after failure | PARTIAL | re-select or re-enable re-opens; no automatic retry loop |
| Real USB/built-in camera | **HARDWARE TEST REQUIRED** | no camera in the verification environment |

Class: **1** with the mock camera, **2** for physical cameras.

## 7. CV / gesture command path status

`camera → detector → 68-D features → recognizer → stabilizer → validator → NG1 encoder → serial → ESP32`

* Verified end-to-end in `test_control_channel.py`: a detected `CALL`/`OK` produced exactly
  **one** `b"NG1|CALL\n"` / `b"NG1|OK\n"` on the pipeline's serial link, and the UI state
  reported `final_command`/`last_tx` accordingly.
* The 13-command taxonomy, `NEUROGRIP_TAXONOMY`, the `NG1|<CMD>\n` wire format and the 68-D
  feature contract are **unchanged**. `communication/protocol.py` was not modified.
* CALL/OK fix (Phase 1, still holding): `resolve_command` now resolves both vocabularies, so
  the validator accepts the canonical names and the encoder emits `NG1|CALL` / `NG1|OK`.
* Real MediaPipe landmarks now reach the UI overlay.
* Class **1** with a stubbed recognizer/detector; **2** with the real model bundle
  (`models/mediapipe/hand_landmarker.task` and the HaGRID checkpoint are absent here).

## 8. Voice assistant status

`mic → Whisper (Electron) → transcript → voice.transcript → pipeline intent → validator → encoder → serial owner`

* Intent resolution moved **into the pipeline** (`voice/intents.py`, a port of the renderer's
  `intents.ts`), so a transcript can only produce a command through the same validated path.
* The renderer no longer transmits voice commands itself; it sends the transcript and shows the
  backend's reply/command. Replies for "status" come from `status_summary()` (real state).
* Verified: `call`, `okay`, `close the hand`, `grip`, `thumbs up` each produced exactly one wire
  frame; unknown phrases transmitted nothing; a voice command with the serial link down failed
  with `serial link unavailable` (no silent success).
* Voice state (`idle/listening/processing/speaking`) is reported to the pipeline and echoed back.
* Class **1** for routing/validation/transmission; **2** for the real microphone + Whisper model
  (no audio device in this environment).

## 9. Manual command integration status

* `Dashboard` manual grid and `Hardware` test buttons → `command.send` → one validated
  transmission. Verified for all 13 taxonomy names plus a rejected name (`GRABBING` → no frame).
* Phase-1 double-send fix still holds: exactly one wire frame per override; the tagged preview
  frame is refused (`reentrant`/`unchanged`).
* An explicit manual request is allowed to retry after a *failed* transmission — a command that
  never reached the wire is not remembered as emitted (`validator.reset()` on failure).
* Class **1** (mock serial) / **2** (ESP32).

## 10. Serial / hardware integration status

| Capability | Status |
|---|---|
| Port enumeration | REAL — `serial.list_ports` via pyserial inside the pipeline |
| Connect / disconnect | REAL — the pipeline's own interface; disconnect leaves it genuinely disconnected |
| Baud + port reported to UI | REAL — `serial_info` |
| "Connected" semantics | REAL — true only when the pipeline holds a link (`mode` distinguishes real/mock) |
| TX telemetry | REAL — `frames_sent` and `last_tx` from the pipeline, no local guessing |
| Real ESP32 | **HARDWARE TEST REQUIRED** |

Class **1** (mock) / **2** (hardware).

## 11. STOP / safety integration status

* STOP is transmitted by the same authoritative path (`command.send STOP` → validator →
  `NG1|STOP`); the validator never de-duplicates STOP.
* Arming state (`is_armed`, `safety_status`) is now actually delivered (the old bridge silently
  dropped those fields) and displayed by the telemetry/hardware pages.
* `stop.set_armed` is implemented and tested end-to-end (explicit armed/disarmed, no toggle
  ambiguity; arming auto-expires after `stop.arm_timeout_ms`).
* **Gap (reported, not hidden):** there is no dedicated STOP-arm button in the existing UI, so
  arming is backend-only from the desktop app today (class 3). STOP *commands* are class 1/2.
* STOP semantics were not weakened or redesigned.

## 12. Real-time telemetry / state integration status

* One IPC subscription (`pipelineSource`) feeds one derived store (`pipelineStore`) consumed via
  `useSyncExternalStore`; the UI holds no parallel copy of pipeline state.
* The state payload now carries the whole `VisualizationState` (previously 10 fields were dropped)
  plus `camera_info`, `serial_info`, `voice_info`, `errors`, `manual_tx_count`, `landmarks`.
* Backend-originated changes propagate: a camera stopped by the pipeline shows as stopped, a
  frame transmitted by the pipeline appears as `lastTx`/`frames_sent`, and a detection made by
  the pipeline appears as the detected gesture.
* Class **1**.

## 13. CALL / OK end-to-end status

| Stage | Verified |
|---|---|
| Recognizer → stabilizer → validator | yes (stub recognizer emitting `CALL`/`OK`) |
| Validator accepts the canonical name | yes |
| Encoder produces `NG1\|CALL\n` / `NG1\|OK\n` | yes |
| Mock serial writes exactly one frame | yes |
| Real serial (pyserial mocked) writes the exact bytes | yes (Phase 1 suite) |
| UI shows the command and the wire frame | yes |
| Manual button path (CALL/OK present in the grid/Hardware buttons) | yes |
| Voice path ("call", "okay" → same frames) | yes |
| Physical ESP32 reception | **HARDWARE TEST REQUIRED** |

Remaining known gap from the Phase 1 review: `recognition/hybrid.py`'s
`EXTRATREES_TO_NEUROGRIP_MAP` / `str(cmd)` usage was not rewritten in this phase; the canonical
mapping used by the production path is unaffected, but the ExtraTrees label→command mapping
still deserves a dedicated review pass.

## 14. Automated test results

| Suite | Command | Result |
|---|---|---|
| Control channel (new) | `pytest pc/tests/test_control_channel.py` | **44 passed** |
| Full Python suite | `pytest pc/tests -q --ignore=pc/tests/test_gui_dashboard.py --ignore=pc/tests/test_gui_neutral_reset_cleanup.py` | **33 failed / 392 passed** — failure set **identical** to the Phase 1 baseline (33 / 348) |
| Frontend Phase 1 harness | `node scripts/verify_frontend_phase1.mjs` | **56/56 passed** |
| Frontend Phase 2 harness (new) | `node scripts/verify_frontend_phase2.mjs` | **42/42 passed** |
| Electron bridge harness (new) | `node scripts/verify_bridge_control.mjs` | **15/15 passed** |
| Type check | `npx tsc --noEmit` | **5 errors, all pre-existing unused-variable warnings** (baseline 20) |

What the new tests actually prove:

* control requests reach the pipeline and are correlated by id, out-of-order, with timeouts and
  disconnects always resolving (no UI hang);
* the renderer and pipeline action vocabularies cannot drift (contract test);
* camera enable/disable/select really open and close the backend device and report truthful state;
* `serial.connect` failure is reported and leaves the pipeline running;
* disconnect really disconnects and then refuses to transmit;
* CALL/OK (and every other taxonomy command) reach the serial link exactly once for manual,
  voice and detection paths;
* invalid commands transmit nothing; duplicate suppression works; a failed send can be retried;
* STOP arming round-trips and is visible to the UI;
* voice transcripts are interpreted inside the pipeline, unknown phrases transmit nothing;
* the 10 previously-dropped state fields and the new runtime sections arrive in the UI;
* the override preview frame never produces a second command.

**No ESP32, camera or microphone exists in this environment** — the tests above use the mock
camera, mock serial, and mocked pyserial boundaries. Physical verification is the user's step
(§17).

## 15. `npm run build` result

```
✓ 1622 modules transformed
dist/index.html  2,336.39 kB │ gzip: 896.70 kB
✓ built in ~11 s
```

(Phase 1 was 1620 modules / 2,330.91 kB / gzip 894.91 kB; `tsc` is green apart from the 5
pre-existing warnings.)

## 16. Remaining failures and known gaps

1. **33 pre-existing Python failures** — byte-identical failure set before and after Phase 1/2
   (verified by diffing the sorted `FAILED` lists). 21 are git-LFS pointer stubs in `archive/`
   (`UnpicklingError: invalid load key 'v'`); the rest are missing `archive/data/...` paths,
   stale taxonomy-count assertions (`13` vs `14`) and hybrid-recognizer label expectations
   (`GRABBING` vs `NO_COMMAND`).
2. **2 GUI test files cannot be collected** — `test_gui_dashboard.py` needs `neurogrip.frontend`
   (the package ships only `frontend.zip`) and `test_gui_neutral_reset_cleanup.py` needs
   `tkinter` (absent). Both are excluded in every run above.
3. **Landmark overlay** is wired to real data but was only exercised with synthetic landmarks.
4. **No automatic camera/serial retry** — a failed device needs a re-select/re-connect from the UI.
5. **STOP arming has no UI control** (see §11).
6. **Nothing starts/stops the Python process from the UI** — launchers still own process
   lifecycle; the camera on/off control is the UI-level equivalent.
7. The `setSource(deviceId)` path is only meaningful for the browser demo; desktop cameras are
   addressed by their real camera index (`camera.select`).

## 17. Hardware tests you must perform

1. **Camera** — select each real camera index in the picker; confirm the preview switches and
   the panel reports the real resolution; then power the camera off/on and confirm
   `CAMERA OFF` → live frames.
2. **ESP32 serial** — connect the board, pick the real COM port + baud, press Connect (the
   button must only read "connected" while the pipeline holds the port), then:
   * show CALL and OK to the camera; confirm the servos react and the telemetry row shows
     `NG1|CALL` / `NG1|OK` and `frames_sent` increments;
   * press the hardware **Test Dispatch** buttons (including CALL/OK);
   * press Disconnect and repeat a manual command — it must **fail visibly** and send nothing;
   * reconnect and confirm the same command transmits again.
3. **STOP** — press the manual STOP button and confirm `NG1|STOP` reaches the firmware; verify
   the safety row matches the backend's armed state.
4. **Microphone** — grant mic access, say "call", "okay", "close the hand", "grip"; confirm the
   transcript, the recognized command, exactly one frame per utterance, and that an unknown
   phrase transmits nothing. Then disconnect the ESP32 and say a command — the UI must report
   the serial failure instead of claiming success.
5. **Voice + camera together** — verify a camera detection and a voice command cannot both write
   the same frame twice.
6. **Long run** — leave it running ~10 minutes and confirm no duplicate serial writers, no UI
   freeze when the camera fails, and that a UI reload re-attaches to the same pipeline.

## 18. Intentionally left unimplemented

* **Visual UI redesign** — explicitly out of scope for Phase 2 (separate phase, awaits approval).
* **Starting/stopping the Python process from the UI** — launcher-owned today; adding process
  lifecycle management would widen the Electron security surface and was not requested.
* **A dedicated STOP-arm control** — would be a new UI element; the backend action and state are
  ready (`stop.set_armed`).
* **Mapping browser `deviceId`s to OpenCV indices** — not possible; desktop cameras are selected
  by real index instead.
* **Firmware-side features** (per-servo telemetry, uptime, ACK frames) — the ESP32 firmware is
  not in this repository and nothing reports them; those rows remain labelled as not reported.
* **Automatic camera/serial recovery loops** — manual re-select/re-connect only.
* **`recognition/hybrid.py` ExtraTrees label mapping cleanup** — flagged in §13, not changed.
