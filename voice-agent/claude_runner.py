"""Wywołanie Claude Code w trybie headless (`claude -p`) i czyszczenie odpowiedzi pod TTS.

Moduł nie ma żadnych zależności poza biblioteką standardową, więc można go
testować bez instalowania serwera.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from dataclasses import dataclass, field


DEFAULT_SYSTEM_PROMPT = (
    "Rozmawiasz z użytkownikiem głosem, po polsku. Twoja odpowiedź zostanie "
    "odczytana przez syntezator mowy, dlatego: odpowiadaj zwięźle (zwykle 1-4 zdania), "
    "pełnymi zdaniami, bez formatowania Markdown, bez list punktowanych, bez bloków kodu, "
    "bez adresów URL i bez emoji. Liczby i skróty zapisuj tak, jak się je wymawia. "
    "Jeśli wykonujesz dłuższe zadanie, na końcu krótko powiedz, co zrobiłeś. "
    "Jeśli czegoś nie rozumiesz, zadaj jedno krótkie pytanie doprecyzowujące."
)


@dataclass
class ClaudeConfig:
    """Ustawienia uruchamiania Claude Code, czytane ze zmiennych środowiskowych."""

    binary: str = "claude"
    workdir: str | None = None
    model: str | None = None
    permission_mode: str | None = None
    allowed_tools: list[str] = field(default_factory=list)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    timeout_s: float = 300.0

    @classmethod
    def from_env(cls) -> "ClaudeConfig":
        tools = os.environ.get("CLAUDE_ALLOWED_TOOLS", "").strip()
        return cls(
            binary=os.environ.get("CLAUDE_BIN", "claude"),
            workdir=os.environ.get("CLAUDE_WORKDIR") or None,
            model=os.environ.get("CLAUDE_MODEL") or None,
            permission_mode=os.environ.get("CLAUDE_PERMISSION_MODE") or None,
            allowed_tools=[t.strip() for t in tools.split(",") if t.strip()],
            system_prompt=os.environ.get("CLAUDE_SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT,
            timeout_s=float(os.environ.get("CLAUDE_TIMEOUT_S", "300")),
        )


@dataclass
class ClaudeReply:
    text: str
    session_id: str | None
    is_error: bool
    raw: dict


def build_command(config: ClaudeConfig, prompt: str, session_id: str | None) -> list[str]:
    """Zbuduj argumenty `claude -p`. Pierwsza tura tworzy sesję, kolejne ją wznawiają."""
    cmd = [config.binary, "-p", prompt, "--output-format", "json",
           "--append-system-prompt", config.system_prompt]
    if session_id:
        cmd += ["--resume", session_id]
    if config.model:
        cmd += ["--model", config.model]
    if config.permission_mode:
        cmd += ["--permission-mode", config.permission_mode]
    if config.allowed_tools:
        cmd += ["--allowedTools", *config.allowed_tools]
    return cmd


def parse_output(stdout: str) -> ClaudeReply:
    """Odczytaj wynik `--output-format json`.

    Claude Code wypisuje jeden obiekt JSON z polami m.in. `result`, `session_id`,
    `is_error`. Gdyby na stdout trafiło coś przed JSON-em, bierzemy ostatnią linię
    zaczynającą się od nawiasu klamrowego.
    """
    data: dict | None = None
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    data = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
    if not isinstance(data, dict):
        return ClaudeReply(text=stdout.strip(), session_id=None, is_error=True, raw={})

    text = data.get("result")
    if not isinstance(text, str) or not text.strip():
        # Odmowa, błąd API albo pusta odpowiedź – pokaż co wiemy.
        text = data.get("error") or data.get("subtype") or "Brak odpowiedzi od Claude Code."
    return ClaudeReply(
        text=str(text),
        session_id=data.get("session_id"),
        is_error=bool(data.get("is_error", False)),
        raw=data,
    )


async def ask_claude(config: ClaudeConfig, prompt: str, session_id: str | None) -> ClaudeReply:
    """Uruchom Claude Code dla jednej wypowiedzi użytkownika."""
    binary = shutil.which(config.binary)
    if not binary:
        raise FileNotFoundError(
            f"Nie znaleziono programu '{config.binary}'. Zainstaluj Claude Code "
            "(npm install -g @anthropic-ai/claude-code) i zaloguj się poleceniem `claude`."
        )
    cmd = build_command(config, prompt, session_id)
    cmd[0] = binary
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=config.workdir or None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.DEVNULL,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=config.timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return ClaudeReply(
            text="Claude Code nie odpowiedział w wyznaczonym czasie.",
            session_id=session_id, is_error=True, raw={},
        )
    stdout = out.decode("utf-8", errors="replace")
    reply = parse_output(stdout)
    if proc.returncode != 0 and not reply.raw:
        detail = err.decode("utf-8", errors="replace").strip() or stdout.strip()
        reply = ClaudeReply(
            text=f"Claude Code zakończył się błędem: {detail[:400]}",
            session_id=session_id, is_error=True, raw={},
        )
    if reply.session_id is None:
        reply.session_id = session_id
    return reply


_MD_PATTERNS = [
    (re.compile(r"```.*?```", re.S), " "),          # bloki kodu
    (re.compile(r"`([^`]*)`"), r"\1"),               # kod w linii
    (re.compile(r"!\[[^\]]*\]\([^)]*\)"), " "),      # obrazki
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),   # linki -> sam tekst
    (re.compile(r"https?://\S+"), " "),              # gołe adresy
    (re.compile(r"^\s{0,3}#{1,6}\s*", re.M), ""),    # nagłówki
    (re.compile(r"^\s*[-*+]\s+", re.M), ""),         # punktory
    (re.compile(r"^\s*\d+[.)]\s+", re.M), ""),       # numeracja
    (re.compile(r"(\*\*|__)(.*?)\1", re.S), r"\2"),  # pogrubienie
    (re.compile(r"(?<!\w)[*_](.+?)[*_](?!\w)", re.S), r"\1"),  # kursywa
    (re.compile(r"^\s*>\s?", re.M), ""),             # cytaty
    (re.compile(r"^\s*\|.*\|\s*$", re.M), " "),      # wiersze tabel
    (re.compile(r"<[^>]+>"), " "),                   # tagi HTML/XML
]


def clean_for_speech(text: str) -> str:
    """Usuń Markdown i inne elementy, które źle brzmią czytane na głos."""
    out = text
    for pattern, repl in _MD_PATTERNS:
        out = pattern.sub(repl, out)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{2,}", "\n", out)
    return out.strip()
