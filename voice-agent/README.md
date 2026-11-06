# Agent głosowy po polsku na Claude Code (z iPhone'a)

Rozmawiasz głosem z Claude Code przez Safari na iPhonie. „Mózgiem” jest Claude Code
uruchomiony na Twoim komputerze (tryb `claude -p`, działa na Twoim loginie, bez
osobnego klucza API). Rozpoznawanie mowy działa lokalnie (faster-whisper), głos
polski daje edge-tts (bez klucza, wymaga internetu).

```
iPhone (Safari)  --HTTPS-->  komputer: server.py
   nagranie           ->  faster-whisper  (mowa -> tekst, lokalnie)
                      ->  claude -p ...   (Claude Code, Twoja subskrypcja)
                      ->  edge-tts        (tekst -> MP3, głos pl-PL-ZofiaNeural)
   odtworzenie   <--  JSON: transkrypcja + odpowiedź + MP3
```

## Wymagania na komputerze

- Python 3.10–3.12 (na 3.13 faster-whisper może nie mieć gotowych paczek),
- Node.js i Claude Code: `npm install -g @anthropic-ai/claude-code`, potem `claude` i logowanie,
- ok. 500 MB miejsca na model Whisper `small` (pobiera się przy pierwszym starcie).

## Instalacja i start

```bash
cd voice-agent
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

W konsoli pojawi się token i adres w stylu `https://<adres-serwera>/?token=XXXX`.
Token chroni dostęp do Claude Code na Twoim komputerze – nie udostępniaj go.

## Połączenie z iPhone'a (ważne: musi być HTTPS)

Safari pozwala użyć mikrofonu tylko na stronie otwartej przez HTTPS. Dwie proste drogi:

1. **Tailscale (polecane, prywatnie).** Zainstaluj Tailscale na komputerze i na iPhonie
   (to samo konto). Na komputerze: `tailscale serve --bg 8000`. Dostaniesz adres
   `https://<nazwa-komputera>.<tailnet>.ts.net` z poprawnym certyfikatem. Dodaj na końcu
   `/?token=XXXX`.
2. **Cloudflare quick tunnel (publiczny adres, bez konta).** `cloudflared tunnel --url http://localhost:8000`
   wypisze adres `https://....trycloudflare.com`. Adres jest publiczny, więc token jest
   obowiązkowy; ustaw własny: `VOICE_AGENT_TOKEN=twoje-tajne-haslo uvicorn ...`.

Na iPhonie otwórz adres z tokenem w Safari, zgódź się na mikrofon. Możesz dodać stronę
do ekranu początkowego (Udostępnij → Do ekranu początkowego) – token zapisuje się w
przeglądarce.

## Jak rozmawiać

- Dotknij mikrofonu, powiedz polecenie, dotknij ponownie – nagranie leci do serwera.
- Odpowiedź pojawia się w czacie i jest czytana na głos (przycisk ⏹ przerywa).
- „Nowa rozmowa” zaczyna nową sesję Claude Code; w jednej sesji Claude pamięta kontekst.
- Pole tekstowe pozwala pisać zamiast mówić (przydatne do testów).

## Konfiguracja (zmienne środowiskowe)

| Zmienna | Domyślnie | Znaczenie |
|---|---|---|
| `CLAUDE_WORKDIR` | katalog uruchomienia | folder, w którym pracuje Claude Code (np. Twój projekt) |
| `CLAUDE_MODEL` | domyślny Claude Code | np. `sonnet` dla szybszych odpowiedzi |
| `CLAUDE_PERMISSION_MODE` | domyślny | `acceptEdits` pozwala edytować pliki bez pytania; `plan` tylko planuje |
| `CLAUDE_ALLOWED_TOOLS` | brak | np. `Read,Grep,Bash(git status)` – narzędzia bez pytania o zgodę |
| `CLAUDE_SYSTEM_PROMPT` | wbudowany | instrukcja „mów po polsku, krótko, bez Markdown” |
| `CLAUDE_TIMEOUT_S` | 300 | limit czasu jednej odpowiedzi |
| `WHISPER_MODEL` | `small` | `base` szybciej, `medium` dokładniej po polsku |
| `TTS_VOICE` | `pl-PL-ZofiaNeural` | albo `pl-PL-MarekNeural` |
| `VOICE_AGENT_TOKEN` | losowy | stały token dostępu |

W trybie `-p` Claude Code nie może zadać pytania o zgodę, więc narzędzie wymagające
zgody jest odrzucane. Jeśli chcesz, by agent zmieniał pliki głosem, ustaw
`CLAUDE_PERMISSION_MODE=acceptEdits` i ogranicz `CLAUDE_WORKDIR` do jednego projektu.

## Testy

```bash
python -m unittest tests.test_claude_runner -v
```

## Typowe problemy

- **„Brak dostępu do mikrofonu”** – strona otwarta przez `http://`, potrzebny HTTPS (patrz wyżej).
- **Nic nie słychać** – sprawdź przełącznik wyciszenia iPhone'a; Safari odtwarza dźwięk dopiero po dotknięciu strony.
- **Długa pierwsza odpowiedź** – model Whisper ładuje się przy starcie (`WHISPER_WARM_UP=0` to wyłącza).
- **„Nie znaleziono programu claude”** – zainstaluj Claude Code lub podaj ścieżkę w `CLAUDE_BIN`.

## Uruchomienie w chmurze (VPS) – krok po kroku

Zamiast własnego komputera agent może działać na tanim serwerze w chmurze. Wtedy działa
cały czas, a iPhone łączy się z nim z dowolnego miejsca.

**Czego potrzebujesz:** serwer VPS z Ubuntu, 2 vCPU i 4 GB RAM (np. Hetzner, OVH,
Mikr.us), ok. 20–30 zł miesięcznie. Przy zakupie dostaniesz publiczny adres IP i dostęp
SSH (hasło albo klucz).

**1. Połącz się z serwerem** (z Maca: aplikacja Terminal, z Windows: PowerShell):

```bash
ssh root@TWOJ_ADRES_IP
```

**2. Zainstaluj Dockera i pobierz projekt:**

```bash
curl -fsSL https://get.docker.com | sh
apt-get install -y git
git clone -b feat/voice-agent https://github.com/nemesis774/app-find-friends.git
cd app-find-friends/voice-agent
```

**3. Ustaw adres i hasło dostępu:**

```bash
cp .env.example .env
nano .env
```

W pliku wpisz `DOMAIN` (bez własnej domeny: IP z myślnikami zamiast kropek plus `.sslip.io`,
np. `203-0-113-10.sslip.io`) i własne `VOICE_AGENT_TOKEN`. Zapisz: Ctrl+O, Enter, Ctrl+X.

**4. Zbuduj i uruchom** (pierwszy raz kilka minut):

```bash
docker compose up -d --build
docker compose logs -f agent      # poczekaj na "Model Whisper gotowy", wyjście: Ctrl+C
```

**5. Zaloguj Claude Code na serwerze** swoim kontem (subskrypcją):

```bash
docker compose exec agent claude
```

Wybierz logowanie przez konto Claude. Claude Code wypisze link – otwórz go na telefonie
lub komputerze, zaloguj się, skopiuj kod i wklej w terminalu. Potem wpisz `/exit`.
Login zapisuje się na woluminie, więc przetrwa restarty. Alternatywa bez logowania:
wpisz `ANTHROPIC_API_KEY` w `.env` (płatność za użycie) i zrób `docker compose up -d`.

**6. Otwórz na iPhonie:** `https://TWOJA_DOMENA/?token=TWOJ_TOKEN`, np.
`https://203-0-113-10.sslip.io/?token=...`. Zgódź się na mikrofon i mów.

**Aktualizacja po zmianach w kodzie:**

```bash
cd app-find-friends/voice-agent && git pull && docker compose up -d --build
```

**Uwagi:**
- Claude Code na serwerze pracuje w katalogu `work/` obok projektu, nie widzi plików z Twojego komputera.
- Adres jest publiczny, więc token jest jedyną ochroną. Używaj długiego hasła i nie włączaj trybu `bypassPermissions`.
- Porty 80 i 443 muszą być otwarte w firewallu serwera (u większości dostawców są domyślnie).
