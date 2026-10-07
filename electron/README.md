# Kriya Sahay Desktop

Kriya Sahay is an Electron robotics-control dashboard with a supervised Python CV backend and an optional on-device voice assistant. Python remains the single owner of the camera, CV pipeline, safety state, command validation and serial interface.

## One-time setup (Windows)

```bash
# From the repository root
npm install

# In electron/
cd electron
npm install
```

Install the Python dependencies in a virtual environment at the repository root or under `pc/`. Electron checks `NEUROGRIP_PYTHON`, `VIRTUAL_ENV`, `.venv`, then falls back to the system Python executable. The Python package must be available from `pc/src`.

## Run

```bash
cd electron
npm start
```

Electron starts one Python process, waits for the existing TCP bridge and confirms the pipeline snapshot before reporting the backend ready. Closing the desktop app requests orderly Python shutdown and releases devices; the Hardware view is where a discovered serial port can be selected and connected. No COM port or camera index is assumed to be the only available device.

The voice model is optional. On first use it can download once and is cached on the PC; model loading is shared between startup warm-up and transcription. The Voice tab reports microphone permission and model availability.

## Optional: prefetch the voice model

```bash
npm run prefetch-model
npm start
```

## Build an installer

```bash
npm run dist
```

## Safety and device reporting

The current NG1 serial protocol does not identify the downstream board or report servo-driver/actuator state. The dashboard therefore reports an open serial link and actual writes only; it does not infer that an ESP32, PCA9685 or servo is present from a port alone. Browser preview mode is explicitly simulated and cannot send commands to hardware.
