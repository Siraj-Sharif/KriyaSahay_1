# NeuroGrip Phase 2.1 — Final Functional Gap Closure & Hardware Readiness

Date: 2026-10-06
Branch: `arena/e7b2eaba-neurogrip-new` (base `bc082db`, Phase 1 `3cb709d`, Phase 2 `d62b217`)
Scope: close the remaining functional/integration gaps after Phase 2 and make the system
ready for physical verification. **No visual redesign** (theme, colours, layout, typography,
navigation and animations are untouched; the only new UI element reuses the existing button and
status styles on the existing Hardware page).

Verification classes used throughout:

| Class | Meaning |
|---|---|
| **1** | REAL + VERIFIED — automated evidence covers the whole path |
| **2** | REAL, CODE-VERIFIED, HARDWARE TEST REQUIRED — path traced and tested, needs real camera / ESP32 / mic |
| **3** | NOT YET INTEGRATED |
| **4** | BROKEN |
| **5** | INTENTIONALLY DISPLAY-ONLY |

---

## 1. Objective

Phase 2 made the UI talk to the running pipeline. Phase 2.1 had to (a) independently re-verify
that claim against the code, (b) close the gaps it left (a STOP ARM/DISARM control, the
ExtraTrees mapping review), and (c) prove that every command source still reaches hardware
through exactly one validated path, with failures visible and state truthful — before anyone
puts the ESP32 on the bench.

## 2. Initial gaps, verified against the code (not taken on trust)

The Phase 2 report was read first and then re-checked against the source. It was correct about
the architecture, but its claim set was incomplete in four ways; each was reproduced in code
before being fixed:

1. **`stop.set_armed` had no caller in the UI.** `src/services/pipelineControl.ts` exported
   `setStopArmed`, and `pc/src/neurogrip/app/control.py` implemented the action, but no
   component imported it (`grep setStopArmed src/` matched only the definition). Class 3 gap.
2. **Voice commands were transmitted twice in desktop mode (real defect, fixed).**
   `localWhisperVoice` sends the transcript to the pipeline (`voice.transcript`), the pipeline
   validates and writes the frame, and then the adapter called `onCommand(remote.gesture)`,
   which in `SystemContext` was `setOverrideGesture` → `useGestureCommand.setOverride` →
   `emit(gesture, "override")` → `desktopSerial.send` → a **second** `command.send` for the same
   utterance (the pipeline's transmission is invisible to the renderer's de-duplication state).
3. **Detected CV gestures were re-dispatched by the renderer (real defect, fixed).** The
   `vision.subscribe` handler always called `emitFromFrame(f)`. In desktop mode the pipeline
   already owns CV transmission, so the renderer was asking for a second dispatch of the *raw*
   frame — including frames the pipeline's stabilizer had **not** yet stabilised
   (`command` is published while the stabilizer is still `STABILIZING`), which let an unstable
   gesture bypass the stabilizer/validator window.
4. **`safety_status` was hardcoded to `"NORMAL"` on the 1-hand path (fixed).** Every other
   branch published `STOP ARMED`/`STOP DISARMED`, so the UI's safety row changed wording with
   hand presence instead of reporting the pipeline's arming state.

Two smaller findings were also closed: a fresh serial link kept the previous port's `Last TX`
(misleading after a port swap), and the state field list in the frontend types omitted
`is_armed` (the value was sent but not typed, so the UI could not use it).

The Phase 2 report's ExtraTrees note ("deserves a dedicated review pass") was correct and is
answered in §3; its STOP-arming gap is answered in §4.

## 3. ExtraTrees / hybrid recognition review (frozen taxonomy untouched)

**Outcome: no mapping bug found. The recognisers were left unchanged; the stale tests that
assert a 14-command taxonomy containing `GRABBING` were not "fixed" into the running code.**

Reviewed: `recognition/hybrid.py`, `recognition/ml_recognizer.py`, `recognition/rule_based.py`,
`recognition/base.py`, `hagrid/taxonomy.py`, `features/extractor.py`, `features/finger_state.py`,
`commands/definitions.py`, `commands/validator.py`, `communication/protocol.py`, plus the
training script that produced the bundle (`scripts/train_model_v2_2.py`).

| Check | Finding |
|---|---|
| Numeric → string step | `MLRecognizer.predict` takes `argmax(predict_proba)` and reads `self._classes[top]`, where `_classes = bundle["classes"]` is the same list the trainer took from `clf.classes_`. sklearn's `predict_proba` columns follow `classes_`, so the index → label step cannot be off by one. Pinned by a real joblib bundle on disk with a non-alphabetical class list and a classifier whose argmax column is known, for **every** class. |
| Bundle metadata | `_validate_bundle` refuses a mismatched `feature_version` (must equal `FeatureExtractor.FEATURE_VERSION`) or `feature_dim` (must equal 68, the frozen 68-D contract). |
| Hardcoded class lists | None in the prediction path; labels always come from the bundle. |
| Legacy names | `EXTRATREES_TO_NEUROGRIP_MAP` is a string→string table with a `NO_COMMAND` default: `INDEX`→`INDEX_FINGER`, `CLOSE`→`CLOSED_FIST`, `GRAB`→`GRIP`, `THUMB_ONLY`→`THUMBS_UP`, `MIDDLE`→`MIDDLE_FINGER`, `TWO_FINGER`→`TWO_FINGERS`, `THREE_FINGER`→`THREE_FINGERS`, `PINKY`→`PINKY`, `FOUR_FINGERS`→`FOUR_FINGERS`, `STOP`→`STOP`, `INDEX_PINKY`→`INDEX_PINKY`, `RING`/`REST`→`NO_COMMAND`. Unknown or non-string input → `NO_COMMAND`. |
| Taxonomy | 13 members, no `GRABBING`; the HaGRID table is the official 34-class order, index-addressed, and `map_hagrid_to_neurogrip` only ever returns a taxonomy member or `NO_COMMAND` (`grabbing` and the deliberately removed `one` → `NO_COMMAND`). |
| Routing gate | `_check_landmark_routing_gate` chooses INDEX_FINGER / TWO_FINGERS / PINKY from finger states **before** inference; only those postures consult ExtraTrees (skipping the CPU CNN), everything else falls through to HaGRID. The pinky rule deliberately requires the thumb to be CLOSED, so the CALL posture (thumb+little finger) cannot be mis-routed to the pinky model. Both directions are pinned by tests. |
| STOP / REST | The rule-based STOP check runs first and is authoritative; `REST` is idle-only and is rejected by the validator; `STOP` is never de-duplicated. |
| Gate-vs-model disagreement | If the gate says INDEX_FINGER but the model answers something else, the **model's** verdict is used (that is the documented routing contract) — the result's `reason` records both (`LANDMARK_GATE_EXTRA_TREES(INDEX_FINGER -> ET predicted CLOSE->CLOSED_FIST)`), so telemetry can audit it. Behaviour unchanged; documented here rather than altered. |
| Dead constant | `ROUTED_TARGET_COMMANDS` is defined and unused. Left in place (removing it is an unrelated refactor); recorded so it is not mistaken for live routing. |

Two related facts that are frequently misread and are now pinned by tests:
the 13-member `NeuroGripCommand` enum is the **dataset** vocabulary (a data contract), and the
runtime/hardware vocabulary is `NEUROGRIP_TAXONOMY`/`CanonicalCommand`; `resolve_command`
accepts both, which is what makes CALL/OK transmittable.

## 4. STOP / safety review — ARM / DISARM control added

**Existing semantics were verified, not redesigned.** The pipeline owns one boolean,
`_is_stop_armed`:

* it is reported in every state snapshot (`is_armed`, `safety_status`),
* it auto-clears after `stop.arm_timeout_ms` (5 s in `pc/config/default.yaml`),
* **it does not gate STOP**: `process_frame` passes `is_stop_armed=True` to the recognizer and
  the stabilizer, `RuleBasedRecognizer` never consults the flag for its STOP branch, and the
  validator has no arming check. STOP stays recognised and transmissible while disarmed —
  pinned by `test_disarmed_stop_is_still_transmitted_exactly_once`. This is the frozen
  fail-safe behaviour asserted by `test_pipeline_disarmed_stop_never_emits_serial_command`
  ("STOP-as-normal-command architecture"); the older `archive/scratch` spec that blocked
  disarmed STOP is superseded and was **not** resurrected.

Because the legacy dashboard exposes the same toggle (`[ TOGGLE STOP SAFETY (SPACE) ]`,
pre-armed on start) and the backend action already existed, the architecture clearly supports a
user-facing control, so the minimum one was implemented:

* `src/pages/Hardware.tsx` — one **Arm STOP / Disarm STOP** button in the existing serial card,
  using the existing `mono rounded-lg border …` button style, plus the existing `StatusDot` on
  the Safety row. The label is derived from `pipelineState.isArmed` and the request always asks
  for the **opposite of the backend's** state (`requestStopArmed(!pipelineState.isArmed)`).
* `src/services/pipelineStore.ts` — the derived state now carries `isArmed`, `isTxPermitted`
  and `finalCommand` straight from the snapshot (`derivePipelineState`), defaulting to
  *not armed / not transmitting* when an older bridge omits them. No local React state.
* `src/services/pipelineControl.ts` — `stopArmedFrom()` reads the pipelined `{armed}` echo.
* A rejected request shows the pipeline's real error in the card's notice line; the button never
  flips optimistically.

Class: **1** with the pipeline present, **2** for the physical STOP behaviour (see the checklist).

## 5. Camera hardware readiness

| Capability | Status | Evidence / notes |
|---|---|---|
| Discovery / enumeration | REAL | `camera.list` probes indices with `cv2.VideoCapture` inside the pipeline and hides what cannot be opened; labels come from DirectShow names when `pygrabber` is present. |
| Selection by index | REAL | `camera.select` closes the old device first and opens the new one; failure returns the real error and marks capture disabled. |
| Enable / disable | REAL | `camera.set_enabled` genuinely opens/releases the capture; disabling also resets the stabilizer and sends idle. |
| Frame delivery | REAL | Frames come from the pipeline's camera via `send_frame` (JPEG, 10 Hz); the renderer never fabricates a frame. |
| Processing stops when disabled | VERIFIED | `test_disabled_camera_stops_consuming_and_publishing_frames` asserts the camera is not read, no preview is published, `frame is None`, FPS 0, `CAMERA_OFF`. |
| Unavailable / invalid device | VERIFIED | `test_unavailable_camera_is_reported_and_never_faked` (OpenCV backend patched so nothing can open): `ok:false`, `camera_info.error = "camera unavailable"`, error present in `errors[]`, state `CAMERA_OFF`. Invalid frames set `camera_error` + `ERROR` state (Phase 2). |
| Timestamps / FPS | REAL | FPS is the rolling mean of measured frame intervals in the pipeline; `latency_ms` is measured per frame. No hardcoded FPS/status anywhere in the frontend (`deriveCameraStatus` is derived from `camera_info`). |
| Hidden assumptions found | Documented | (a) `list_cameras(probe_limit=5)` offers the configured index plus indices 0–4, so a camera at index ≥5 is only reachable if it is the configured one. (b) The OpenCV backend auto-scans 0–9 when the requested index fails and **mutates `config.index`** to the device it actually opened — the UI then displays the real (fallback) index, so it cannot show a camera that is not open, but the selection is not guaranteed to be the requested device. (c) `camera_info.resolution` falls back to the requested target when the driver reports 0×0. |
| Auto retry / recovery | NOT IMPLEMENTED | Deliberate: re-select or re-enable from the UI. Recorded as a limitation (§20). |

Class **1** with the mock camera, **2** for real USB/built-in devices.

## 6. Serial hardware readiness

| Capability | Status | Evidence / notes |
|---|---|---|
| Port discovery | REAL | `serial.list_ports` → `pyserial.comports()` inside the pipeline. |
| Connect / disconnect | REAL | `serial.connect` / `serial.disconnect` build and release the pipeline's own interface; `connected` is true only when the interface reports it. |
| Port / baud reporting | REAL | `serial_info` (mode, port, baud, state, frames_sent, last_tx) is the single source; verified by test including a live swap to `COM_TEST @ 57600`. |
| Exactly one owner | VERIFIED | `configure_serial()` builds the **new** interface, connects it, then disconnects the old one; the test asserts the old interface is released and the new one is the pipeline's only link. A static guard asserts `serial.Serial(` appears only in `communication/serial_port.py` and that `electron/*.cjs` contains no `child_process`/`spawn(`/`execFile(`/`execSync`/`serialport`. |
| Command while disconnected | VERIFIED | `test_command_after_disconnect_is_rejected_and_transmits_nothing`: `ok:false`, `sent:false`, error mentions serial, zero new bytes, `errors[]` contains `serial link unavailable`. Same for the voice path. |
| Last transmitted frame | REAL (fixed) | `last_tx` is the frame the pipeline actually encoded; a **new link now reports `NONE`** instead of the previous port's frame. |
| TX permission | REAL | `config.serial.enabled` + `is_connected`; the UI never opens a second connection to send (all sends go through `command.send`). |
| Real ESP32 | **HARDWARE TEST REQUIRED** | no board in this environment; see checklist §2. |

Class **1** with mock serial, **2** for the physical board.

## 7. Voice hardware readiness

Flow: `mic → Whisper (Electron `stt:*`) → transcript → voice.transcript → pipeline intent → validator → encoder → the pipeline's serial owner`.

| Case | Verified behaviour |
|---|---|
| `call`, `okay`, `stop`, `close`, `point`, `open` | Each produces **exactly one** frame (`NG1|CALL`, `NG1|OK`, `NG1|STOP`, `NG1|CLOSED_FIST`, `NG1|INDEX_FINGER`, `NG1|STOP`) with `sent:true` and the wire name echoed back. `open` → STOP is the frozen vocabulary (`intents.ts`/`intents.py`), documented rather than changed. |
| Unsupported phrase (`pinch`) | `intent:"unknown"`, no transmission — `pinch` is not in the vocabulary; adding it would be a taxonomy decision, so it is reported as a limitation. |
| Invalid/ambiguous speech | `parse_intent` returns `unknown` with a spoken reply; nothing is transmitted. |
| Serial down | The command fails with the real serial error (visible in the voice log); no frame. |
| Repetition | An explicit voice request is an intentional retry: `close` twice transmits twice (explicit sources bypass the detector latch). |
| Duplicate transmission | **Fixed in 2.1**: the renderer no longer re-sends a command the pipeline already wrote (preview-only voice callback). |
| Real microphone / Whisper model | **HARDWARE TEST REQUIRED** — no audio device in this environment. |

Class **1** for routing/validation/transmission, **2** for the physical microphone.

## 8. Command-source audit

| Source | Path | Classification |
|---|---|---|
| Camera detection | `process_frame` → stabilizer → validator → `protocol.py` → `pipeline.serial` | legitimate (single owner) |
| Manual buttons / Hardware test buttons | `command.send` → `dispatch_command` → same validator + serial | legitimate (single owner) |
| Voice | `voice.transcript` → `dispatch_command(source="voice")` → same path | legitimate (single owner) |
| `electron/*.cjs` `serial:*` wrappers | thin wrappers over the control channel | legitimate (no process, no port) |
| `python -c` one-shot helpers | removed in Phase 2; only a **comment** in `main.cjs` documents them | historical/dead |
| `serial.Serial(` | only `communication/serial_port.py` | the one serial owner |
| `subprocess` / `spawn` / `child_process` | none in `electron/*.cjs`; the launcher scripts own process start/stop (outside the app) | test-only / launcher-owned |
| Alternate emitters | none: `ProtocolEncoder.encode_command_string` is the only frame encoder, reached only through the validator | frozen protocol |

Two genuine bypasses were found and fixed (see §2 items 2 and 3). Everything else classified as
legitimate, test-only, initialisation-only or historical-dead.

## 9. Real-time state audit

One IPC subscription (`pipelineSource`) → one derived store (`pipelineStore`, read through
`useSyncExternalStore`) → components. Every field below arrives in the payload and is derived
from the backend; nothing is replaced by a hardcoded frontend value.

| Field | Payload key | Frontend |
|---|---|---|
| camera | `camera_info` (index, enabled, opened, state, backend, resolution, target_resolution, error) | `cameraStatus`, `cameraIndex`, `cameraEnabled`, `cameraError`, `cameraResolution` |
| serial | `serial_info` (mode, port, baud, connected, state, frames_sent, last_tx) | `serialConnected`, `serialMode`, `serialPort`, `baudRate`, `framesSent`, `lastTx` |
| voice | `voice_info` (state, transcript, command, reply, engine) | `voice` |
| errors | `errors[]` (camera + last error) | `errors`, and now rendered on the Hardware page ("Pipeline errors") |
| final command | `final_command` | `finalCommand` (new) |
| stabilizer | `stabilizer_state`, `is_stabilized` | raw payload (TelemetryPanel), typed in `RawPipelineState` |
| safety | `is_armed`, `safety_status`, `is_stop_active` | `isArmed` (new), `safetyStatus`, `isStopActive` |
| transmission | `is_tx_permitted`, `last_tx` | `isTxPermitted` (new), `lastTx` |
| STOP diagnostics | `stop_rule_triggered`, `stop_rule_metrics` | raw payload + TelemetryPanel |
| HaGRID diagnostics | `hagrid_raw`, `hagrid_conf`, `taxonomy_mapped` | raw payload + TelemetryPanel |
| FPS / latency / recognition | `fps`, `latency_ms`, `command`, `confidence`, `hand_count`, `pipeline_state`, `model_used`, `routing_str` | same-named derived fields |
| landmarks | `landmarks` (+`landmarks_count`) | `landmarks`, `landmarksCount` |

Intentional display-only fields: gesture descriptions, servo channel tables, static topology
cards, 3D hand and charts (`GestureGuide`, `AIPipelineDiagram`, servo specs) — all class 5.

## 10. Error / failure-visibility audit

| Failure | Where it surfaces | Test |
|---|---|---|
| Camera unavailable / cannot open | `camera.set_enabled` → `ok:false` + `camera_info.error`; state `CAMERA_OFF`; `errors[]` | `test_unavailable_camera_is_reported_and_never_faked` |
| Invalid camera frame | `_camera_error` + state `ERROR` (Phase 2) | Phase 2 suite |
| Serial disconnected | `ok:false` with the pipeline's error, `sent:false`, `errors[]` | `test_command_after_disconnect_is_rejected_and_transmits_nothing` |
| Command while not connected (voice) | same, visible in the voice log | `test_voice_command_is_refused_while_serial_is_down` |
| Invalid command | validator rejection (`GRABBING`, unknown names) → no frame | Phase 1/2 suites |
| Voice engine unavailable | `stt:*` engine status + failure message in the voice log; no fake success | `verify_frontend_phase2.mjs` (engine status) |
| Control-channel timeout/disconnect | every pending request resolves `{ok:false,error}` (6 s timeout) | `verify_bridge_control.mjs`, `test_control_channel.py` |
| Python pipeline gone | bridge status off; UI state stops updating (no fabricated state) | Phase 2 harness + checklist K5 |
| Any refused command | error string in `errors[]` + a notice line at the click site | same as above |

No fake success paths were found: an action that did not reach hardware always returns
`ok:false`/`sent:false` and shows the backend's own message.

## 11. UI integration table (retained 1–5 classification)

| UI element (file) | Action | Backend path | Class |
|---|---|---|---|
| Navbar tabs | switch page | client-side | 5 |
| Navbar "Backend Live" dot | read state | bridge-status on socket accept | 1 |
| Navbar hand/gesture dot, serial dot | read state | `command`/`hand_count`, `serial_info` | 1 |
| Navbar fullscreen | window state | Electron BrowserWindow | 1 |
| Camera picker + Rescan (`CameraSelector`) | `camera.list` / `camera.select` | pipeline probes/opens the device | 1 (mock) / 2 (real) |
| Live feed + landmark overlay (`CameraPanel`) | read state | pipeline JPEG frames + real landmarks | 1 / 2 |
| Camera on/off, camera-off/error banner | `camera.set_enabled`, read state | pipeline opens/releases capture; `camera_info.error` | 1 / 2 |
| Gesture readout, confidence, FPS/latency, pipeline status | read state | `command`, `confidence`, `fps`, `latency_ms`, `pipeline_state` | 1 |
| Telemetry panel | read state | full snapshot incl. safety/HaGRID fields | 1 |
| Dashboard manual gesture buttons (13, incl. STOP) | `command.send` (source `override`) | validator → encoder → serial owner | 1 / 2 |
| Dashboard "Release" | clear override | preview only | 1 (no command) |
| Dashboard override indicator | read state | local override state + real frame | 5 |
| Gesture Guide cards | local preview | 3D hand only | 5 |
| **Hardware: Arm/Disarm STOP (new)** | `stop.set_armed` | pipeline arming state; label follows `is_armed` | 1 / 2 |
| **Hardware: Safety row (updated)** | read state | `safety_status` + `is_armed` (dot) | 1 |
| **Hardware: Pipeline errors row (new)** | read state | `errors[]` | 1 |
| Hardware: Rescan / port / baud / Connect / Disconnect | `serial.list_ports`, `serial.connect`, `serial.disconnect` | the pipeline's own serial owner | 1 / 2 |
| Hardware: status, active port, baud, framing, last TX, transport, mode, frames sent | read state | `serial_info` | 1 |
| Hardware: Test Command Dispatch (STOP/GRIP/INDEX/FIST/REST) | `command.send` | same single path | 1 / 2 |
| Voice: mic button, mic picker, test meter, allow-access | local capture | browser/Electron device APIs; state reported via `voice.set_state` | 2 |
| Voice: transcript / typed text / chips | `voice.transcript` | pipeline intent → validator → serial owner | 1 / 2 |
| Voice: reply, status intent | backend reply | `status_summary()` from real state | 1 |
| Voice: TTS toggle, volume, voice select, replay | local | browser speech synthesis | 5 |
| Voice: engine status | read state | Electron Whisper (`stt:*`) | 2 |
| Start/stop/restart the Python process | — | owned by the launcher scripts | 3 (reported, unchanged) |

No element is class 4. The only class 3 item is process lifecycle (deliberate).

## 12. Files changed (7)

* `pc/src/neurogrip/app/pipeline.py` — truthful `safety_status` on the 1-hand path; `last_tx`
  reset when the serial link is rebuilt.
* `src/services/pipelineStore.ts` — `isArmed`, `isTxPermitted`, `finalCommand` derived from the
  backend snapshot.
* `src/services/pipelineMapping.ts` — `is_armed` / `stabilizer_state` typed in `RawPipelineState`.
* `src/services/pipelineControl.ts` — `stopArmedFrom()` helper.
* `src/services/SystemContext.tsx` — voice commands are preview-only in desktop mode; detector
  frames are no longer re-emitted when the pipeline owns transmission.
* `src/hooks/useGestureCommand.ts` — `previewOverride()` (visual only) split out of `setOverride()`.
* `src/pages/Hardware.tsx` — the Arm/Disarm STOP control and the pipeline-errors row (existing
  styles only).

## 13. Files added (2)

* `pc/tests/test_phase2_1_gaps.py` — 39 tests (§3–§11).
* `scripts/verify_frontend_phase2_1.mjs` — 30 frontend checks (arming control, derived safety
  state, duplicate-send guards, static wiring guards).

## 14. Tests added / updated

* ExtraTrees/HaGRID: bundle class-order resolution for every class, canonical mapping for every
  legacy label, taxonomy membership of every mapped value, HaGRID index table, routing gate
  (INDEX/TWO/PINKY, thumb-open pinky protected), gated path skips HaGRID, ungated path uses
  HaGRID as CALL.
* STOP: arm/disarm round-trip via the control channel, non-boolean rejection, disarmed STOP
  still transmits exactly once, STOP never de-duplicated, CV transmission waits for the
  stabilizer.
* Camera: unavailable device reported and never faked, disabled camera stops reads/preview.
* Serial: command after disconnect rejected with zero bytes, swap keeps one owner with fresh
  `last_tx`/`frames_sent`.
* Voice: one frame per recognised phrase (`call`, `okay`, `open`, `stop`, `close`, `point`),
  unsupported phrase transmits nothing, failure while disconnected, repeated command = retry.
* Cross-source: CV + manual + voice all reach the same serial object in order; static absence of
  duplicate hardware paths.
* State/errors: full required key set; a refused transmission is visible in `errors[]`.

## 15. Test results

| Suite | Command | Result |
|---|---|---|
| Phase 2.1 gaps (new) | `pytest pc/tests/test_phase2_1_gaps.py` | **39 passed** |
| Phase 1 + 2 + 2.1 integration | `pytest pc/tests/test_phase2_1_gaps.py pc/tests/test_control_channel.py pc/tests/test_call_ok_hardware_command.py` | **143 passed** (4.6 s) |
| Serial-integration file (includes pre-existing cases) | `pytest pc/tests/test_phase11_serial_integration.py` | 8 passed / 4 pre-existing LFS failures |
| Full Python suite | `pytest pc/tests -q --ignore=pc/tests/test_gui_dashboard.py --ignore=pc/tests/test_gui_neutral_reset_cleanup.py` | **33 failed / 431 passed** — failure set **identical** to the Phase 1/2 baseline (33/392 before the 39 new tests) |
| Frontend Phase 1 harness | `node scripts/verify_frontend_phase1.mjs` | **56/56** |
| Frontend Phase 2 harness | `node scripts/verify_frontend_phase2.mjs` | **42/42** |
| Electron bridge harness | `node scripts/verify_bridge_control.mjs` | **15/15** |
| Frontend Phase 2.1 harness (new) | `node scripts/verify_frontend_phase2_1.mjs` | **30/30** |
| Type check | `npx tsc --noEmit` | **5 errors, all pre-existing unused-variable errors** (`Navbar.tsx(28,9)`, `TelemetryPanel.tsx(28,5)`, `TelemetryPanel.tsx(40,5)`, `Dashboard.tsx(17,11)`, `Dashboard.tsx(17,18)`) — no new errors |

The new tests were also run in isolation repeatedly (`-p no:randomly`) to confirm they are not
order-dependent.

## 16. Build result

```
vite v7.3.2 building client environment for production...
✓ 1622 modules transformed
dist/index.html  2,338.72 kB │ gzip: 897.21 kB
✓ built in 10.03s
```

(Phase 2: 1622 modules / 2,336.39 kB / gzip 896.70 kB — the small delta is the new safety
control and helpers.)

## 17. Baseline failures (pre-existing, unchanged)

33 failures, byte-identical to the Phase 1 and Phase 2 baselines (`diff` of the sorted `FAILED`
lists is empty). Causes:

* **21** — `_pickle.UnpicklingError: invalid load key, 'v'`: git-LFS pointer stubs under
  `archive/` (including `pc/models/hagrid/ResNet18.pth`, `YOLOv10n_hands.pt`) loaded by
  `hagrid/model.py`. Data-availability failures, not code.
* **2** — stale taxonomy-count assertions (`assert 13 == 14`) and **2** list-length variants
  expecting `GRABBING` as a 14th command (`test_hagrid_model.py`,
  `test_hybrid_production_pipeline.py::test_1…`, `test_hagrid_realworld_eval.py`). The locked
  taxonomy has 13 members and no `GRABBING`; the tests are wrong, the code is right.
* **3** — `assert 1 == 0` and **3** missing `archive/data/...` paths (dataset artefacts absent).
* The remaining ones are the same archive/LFS-dependent pipeline and dataset scripts.

These were **not** hidden, skipped or "fixed" by loosening assertions.

## 18. New failures

**None.** No test that passed before Phase 2.1 fails now; the failure list is identical.

## 19. Hardware tests to perform

`docs/PHASE2_1_HARDWARE_TEST_CHECKLIST.md` — preconditions (P1–P3), CAMERA (C1–C10),
SERIAL (S1–S14), VOICE (V1–V10), SAFETY (K1–K9), each with the expected result and the failure
observation to record.

## 20. Remaining limitations (deliberate, not failures)

1. **Process lifecycle is launcher-owned.** Nothing in the UI starts/stops the Python pipeline;
   the launchers (`start_neurogrip.ps1/.bat`, hardcoded to `D:\NeuroGrip_Project_New`) do. The
   camera on/off control is the in-app equivalent.
2. **No automatic retry/recovery** for camera or serial: a device that disappears needs an
   explicit re-select/re-enable/re-connect. Adding reconnect loops was judged out of scope for a
   readiness pass.
3. **No hardware in the verification environment**: camera, microphone, ESP32 and COM ports are
   absent, so every physical claim is class 2 at best.
4. **`pinch` is not a voice phrase**; `open`/`release`/`freeze`/`relax`/`palm` intentionally map
   to STOP (frozen vocabulary, shared by `intents.ts` and `intents.py`).
5. **Camera index probing is 0–4** plus the configured index; the OpenCV fallback may open a
   different index than requested (the reported index is always the real one).
6. **The GUI test files remain uncollectable** here (`test_gui_dashboard.py` needs
   `neurogrip.frontend`, shipped only as `frontend.zip`; `test_gui_neutral_reset_cleanup.py`
   needs `tkinter`), so they are excluded from every run.
7. **33 pre-existing failures** remain (§17).
8. **`.venv`/`node_modules` are not part of the repository**; in the sandbox
   `opencv-python-headless` is required (no X11) and the npm lockfile points at a mirror.
9. **The arm flag is a status/reporting flag, not an interlock** — documented in §4 rather than
   changed, because gating STOP on it would weaken the frozen safety behaviour.

## 21. What is and is not physically verified

**Verified in software (automated, reproducible):** the control channel; camera enable/disable/
select and failure reporting against the real OpenCV code path with an unopenable device; serial
connect/disconnect/swap with exactly one owner; command refusal after disconnect with zero bytes;
STOP availability, non-de-duplication and single transmission; voice intent routing for the
supported phrases and zero transmission for unsupported input; CV + manual + voice sharing one
serial owner; the state payload's completeness; error visibility; the ExtraTrees/HaGRID mapping
(incl. class-order resolution from a real joblib bundle); the build and type check.

**NOT physically verified (must be done on the real rig):** ESP32 reception and servo behaviour;
the real webcam (discovery, resolution, real FPS, gesture recognition quality with real
lighting/hands); the real microphone and Whisper transcription; COM9/baud on the target PC;
sustained multi-minute hardware behaviour. **The system is therefore not "fully hardware
verified"** — it is code-verified and ready for the checklist above.
