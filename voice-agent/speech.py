"""Rozpoznawanie mowy (faster-whisper, lokalnie) i synteza (edge-tts, polski głos neuronowy).

Ciężkie biblioteki są importowane leniwie, żeby serwer startował szybko, a testy
nie wymagały ich instalacji.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from functools import lru_cache
from pathlib import Path

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")      # tiny/base/small/medium/large-v3
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")      # cpu lub cuda
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE", "int8")   # int8 na CPU, float16 na GPU
TTS_VOICE = os.environ.get("TTS_VOICE", "pl-PL-ZofiaNeural")  # albo pl-PL-MarekNeural
TTS_RATE = os.environ.get("TTS_RATE", "+5%")

# Rozszerzenie pliku dobieramy po typie MIME, bo iPhone nagrywa audio/mp4, a Chrome audio/webm.
_EXT_BY_MIME = {
    "audio/mp4": ".m4a", "audio/x-m4a": ".m4a", "audio/aac": ".aac",
    "audio/webm": ".webm", "audio/ogg": ".ogg", "audio/wav": ".wav",
    "audio/wave": ".wav", "audio/x-wav": ".wav", "audio/mpeg": ".mp3",
}


def extension_for(mime: str | None) -> str:
    if not mime:
        return ".bin"
    base = mime.split(";")[0].strip().lower()
    return _EXT_BY_MIME.get(base, ".bin")


@lru_cache(maxsize=1)
def _whisper():
    from faster_whisper import WhisperModel  # import leniwy

    print(f"[speech] Ładuję model Whisper '{WHISPER_MODEL}' ({WHISPER_DEVICE}, {WHISPER_COMPUTE})...")
    model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
    print("[speech] Model Whisper gotowy.")
    return model


def warm_up() -> None:
    """Pobierz i załaduj model przy starcie serwera, a nie przy pierwszym nagraniu."""
    _whisper()


def _transcribe_sync(audio_bytes: bytes, mime: str | None) -> str:
    with tempfile.NamedTemporaryFile(suffix=extension_for(mime), delete=False) as tmp:
        tmp.write(audio_bytes)
        path = tmp.name
    try:
        segments, _info = _whisper().transcribe(
            path, language="pl", beam_size=5, vad_filter=True,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
    finally:
        Path(path).unlink(missing_ok=True)


async def transcribe(audio_bytes: bytes, mime: str | None) -> str:
    """Zamień nagranie na tekst, nie blokując pętli zdarzeń serwera."""
    return await asyncio.to_thread(_transcribe_sync, audio_bytes, mime)


async def synthesize(text: str) -> bytes:
    """Zamień tekst na MP3 (edge-tts korzysta z usługi Microsoftu, bez klucza)."""
    import edge_tts  # import leniwy

    if not text.strip():
        return b""
    communicate = edge_tts.Communicate(text, voice=TTS_VOICE, rate=TTS_RATE)
    chunks: list[bytes] = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks)
