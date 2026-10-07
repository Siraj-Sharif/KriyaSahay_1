"""
neurogrip.voice
---------------
Voice-intent parsing for the pipeline.

Audio capture and speech-to-text stay on the UI side (Electron Whisper / browser Speech
API); the **transcript** is handed to the running pipeline, which parses the intent,
validates the resulting command and transmits it through the one authoritative serial
path. This module is that parser.
"""
from neurogrip.voice.intents import Intent, parse_intent

__all__ = ["Intent", "parse_intent"]
