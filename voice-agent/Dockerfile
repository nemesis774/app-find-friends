# Obraz z Claude Code (Node 22) + Python 3.11 dla serwera agenta głosowego.
FROM node:22-bookworm-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv python3-pip ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/* \
 && npm install -g @anthropic-ai/claude-code

# Użytkownik bez uprawnień roota; konfiguracja Claude i modele Whisper trafiają na wolumin /data.
RUN useradd -m -s /bin/bash agent && mkdir -p /data /work /app && chown -R agent:agent /data /work /app
ENV CLAUDE_CONFIG_DIR=/data/claude \
    HF_HOME=/data/huggingface \
    CLAUDE_WORKDIR=/work \
    DISABLE_AUTOUPDATER=1 \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app
COPY --chown=agent:agent requirements.txt .
USER agent
RUN python3 -m venv .venv && pip install --no-cache-dir -r requirements.txt
COPY --chown=agent:agent . .

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
