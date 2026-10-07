# NeuroGrip Desktop

Runs the NeuroGrip UI as a Windows desktop app with a fully on-device voice assistant
(Whisper speech recognition + your OS voices). No internet is needed once the model is on disk.

## One-time setup (on your Windows laptop)

```bash
# 1. in the project root
npm install

# 2. in this folder
cd electron
npm install
```

## Run it

```bash
cd electron
npm start          # builds the UI, then opens the desktop app
```

First launch: the voice model (~75 MB) downloads once and is cached on your PC.
The Voice tab shows the progress. After that it works offline.

## Make it 100 % offline from the very first launch

```bash
npm run prefetch-model   # downloads the model into electron/models
npm start                # now uses the bundled copy, never touches the network
```

## Build an installer (.exe)

```bash
npm run dist             # → electron/release/  (installer + portable .exe)
```
This bundles the model inside the installer.

## How the voice assistant works here

mic you selected → records until you pause → Whisper (runs locally) → command parser → hand + serial → spoken reply

- Works with any microphone Windows lists: laptop array, USB, Bluetooth, headset, audio interface.
- Needs *Windows Settings → Privacy & security → Microphone → "Let desktop apps access your microphone"* turned on.
- Change the model with an environment variable, e.g. `NEUROGRIP_STT_MODEL=Xenova/whisper-small.en`
  (more accurate, slower). Run `npm run prefetch-model` again after changing it.
