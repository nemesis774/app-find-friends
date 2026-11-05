"""Serwer agenta głosowego: iPhone (Safari) <-> ten serwer <-> Claude Code.

Przepływ jednej wypowiedzi:
  nagranie z przeglądarki -> faster-whisper (tekst) -> `claude -p` -> edge-tts (MP3) -> przeglądarka

Uruchomienie:  uvicorn server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import base64
import os
import secrets
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

import speech
from claude_runner import ClaudeConfig, ask_claude, clean_for_speech

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

# Token chroni dostęp do Claude Code na Twoim komputerze. Jeśli nie podasz własnego
# (VOICE_AGENT_TOKEN), generowany jest losowy i wypisywany przy starcie.
TOKEN = os.environ.get("VOICE_AGENT_TOKEN") or secrets.token_urlsafe(12)
WARM_UP = os.environ.get("WHISPER_WARM_UP", "1") != "0"

CONFIG = ClaudeConfig.from_env()
app = FastAPI(title="Agent głosowy Claude Code")

# sesje rozmów: id klienta (z przeglądarki) -> session_id Claude Code
_sessions: dict[str, str | None] = {}


@app.on_event("startup")
async def _startup() -> None:
    print("=" * 60)
    print("Agent głosowy Claude Code")
    print(f"  katalog roboczy Claude: {CONFIG.workdir or os.getcwd()}")
    print(f"  model: {CONFIG.model or 'domyślny'}  tryb uprawnień: {CONFIG.permission_mode or 'domyślny'}")
    print(f"  token dostępu: {TOKEN}")
    print("  otwórz na telefonie:  https://<adres-serwera>/?token=" + TOKEN)
    print("=" * 60)
    if WARM_UP:
        import asyncio
        await asyncio.to_thread(speech.warm_up)


def _check_token(request: Request) -> None:
    sent = request.headers.get("x-voice-token") or request.query_params.get("token")
    if not sent or not secrets.compare_digest(sent, TOKEN):
        raise HTTPException(status_code=401, detail="Nieprawidłowy token dostępu.")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/app.js")
async def app_js() -> FileResponse:
    return FileResponse(STATIC_DIR / "app.js", media_type="application/javascript")


@app.get("/api/health")
async def health(_: None = Depends(_check_token)) -> dict:
    return {"ok": True, "workdir": CONFIG.workdir or os.getcwd(), "voice": speech.TTS_VOICE}


async def _respond(client_id: str, user_text: str, want_audio: bool) -> dict:
    session_id = _sessions.get(client_id)
    reply = await ask_claude(CONFIG, user_text, session_id)
    _sessions[client_id] = reply.session_id
    spoken = clean_for_speech(reply.text)
    audio_b64 = ""
    if want_audio and spoken:
        try:
            audio_b64 = base64.b64encode(await speech.synthesize(spoken)).decode("ascii")
        except Exception as exc:  # TTS nie może blokować odpowiedzi tekstowej
            print(f"[tts] błąd syntezy: {exc!r}")
    return {
        "transcript": user_text,
        "reply": reply.text,
        "spoken": spoken,
        "audio_mp3_base64": audio_b64,
        "is_error": reply.is_error,
        "session_id": reply.session_id,
    }


@app.post("/api/talk")
async def talk(
    audio: UploadFile = File(...),
    client_id: str = Form(...),
    want_audio: bool = Form(True),
    _: None = Depends(_check_token),
) -> JSONResponse:
    data = await audio.read()
    if len(data) < 1000:
        raise HTTPException(status_code=400, detail="Nagranie jest puste lub za krótkie.")
    text = await speech.transcribe(data, audio.content_type)
    if not text:
        return JSONResponse({"transcript": "", "reply": "", "spoken": "", "audio_mp3_base64": "",
                             "is_error": False, "nothing_heard": True})
    return JSONResponse(await _respond(client_id, text, want_audio))


@app.post("/api/text")
async def text_message(
    text: str = Form(...),
    client_id: str = Form(...),
    want_audio: bool = Form(True),
    _: None = Depends(_check_token),
) -> JSONResponse:
    if not text.strip():
        raise HTTPException(status_code=400, detail="Pusta wiadomość.")
    return JSONResponse(await _respond(client_id, text.strip(), want_audio))


@app.post("/api/reset")
async def reset(client_id: str = Form(...), _: None = Depends(_check_token)) -> dict:
    _sessions.pop(client_id, None)
    return {"ok": True, "client_id": client_id or str(uuid.uuid4())}
