// Logika strony: nagrywanie (MediaRecorder), wysyłka do serwera, odtwarzanie MP3.
(() => {
  const $ = (id) => document.getElementById(id);
  const log = $("log"), status = $("status"), mic = $("mic"), player = $("player");

  // Token z adresu (?token=...) zapisujemy, żeby móc dodać stronę do ekranu początkowego.
  const params = new URLSearchParams(location.search);
  if (params.get("token")) {
    localStorage.setItem("voiceToken", params.get("token"));
    history.replaceState(null, "", location.pathname);
  }
  const token = localStorage.getItem("voiceToken") || "";
  let clientId = localStorage.getItem("voiceClient");
  if (!clientId) { clientId = crypto.randomUUID(); localStorage.setItem("voiceClient", clientId); }

  const add = (cls, text) => {
    const el = document.createElement("div");
    el.className = "msg " + cls; el.textContent = text;
    log.appendChild(el); log.scrollTop = log.scrollHeight; return el;
  };
  const setStatus = (t) => { status.textContent = t; };

  // Safari pozwala odtwarzać dźwięk tylko po geście użytkownika. Podczas dotknięcia
  // "odblokowujemy" element audio cichym plikiem; później można ustawić inne źródło.
  const SILENCE = "data:audio/mp3;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA//tQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABJbmZvAAAADwAAAAIAAAGyAGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBg////////////////////////////////////////////////////////////////AAAAAExhdmM1OC4xMwAAAAAAAAAAAAAAACQCgAAAAAAAAAGy6jN9gAAAAAAAAAAAAAAAAAAAAP/7EGQAAAHdAE1VAAAAAAD/8AAAAAAA==";
  let unlocked = false;
  const unlockAudio = () => {
    if (unlocked) return;
    player.src = SILENCE;
    player.play().then(() => { unlocked = true; }).catch(() => {});
  };
  const playBase64 = (b64) => {
    if (!b64 || !$("speak").checked) return;
    player.src = "data:audio/mpeg;base64," + b64;
    player.play().catch((e) => add("sys err", "Nie udało się odtworzyć dźwięku: " + e.message));
  };
  $("stop").onclick = () => { player.pause(); player.currentTime = 0; };

  const api = async (path, form) => {
    form.append("client_id", clientId);
    form.append("want_audio", $("speak").checked ? "true" : "false");
    const res = await fetch(path, { method: "POST", body: form, headers: { "x-voice-token": token } });
    if (res.status === 401) throw new Error("Brak lub zły token. Otwórz adres z ?token=… z konsoli serwera.");
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || ("Błąd serwera " + res.status));
    return res.json();
  };

  const handleReply = (data) => {
    if (data.nothing_heard) { setStatus("Nic nie usłyszałem, spróbuj jeszcze raz."); return; }
    if (data.transcript && !data._typed) add("me", data.transcript);
    add(data.is_error ? "ai err" : "ai", data.reply);
    playBase64(data.audio_mp3_base64);
    setStatus("Dotknij mikrofonu i mów.");
  };

  // ---- nagrywanie ----
  let recorder = null, chunks = [], stream = null;
  const mimeType = ["audio/mp4", "audio/webm;codecs=opus", "audio/webm", "audio/ogg"]
    .find((m) => window.MediaRecorder && MediaRecorder.isTypeSupported(m)) || "";

  const startRec = async () => {
    unlockAudio();
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e) {
      add("sys err", "Brak dostępu do mikrofonu: " + e.message + " (strona musi być otwarta przez HTTPS).");
      return;
    }
    chunks = [];
    recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    recorder.onstop = sendRecording;
    recorder.start();
    mic.classList.add("rec"); setStatus("Słucham… dotknij ponownie, aby wysłać.");
  };

  const stopRec = () => {
    unlockAudio();
    if (recorder && recorder.state !== "inactive") recorder.stop();
    if (stream) stream.getTracks().forEach((t) => t.stop());
    mic.classList.remove("rec");
  };

  const sendRecording = async () => {
    const blob = new Blob(chunks, { type: recorder.mimeType || mimeType || "audio/webm" });
    mic.disabled = true; setStatus("Rozpoznaję mowę i pytam Claude…");
    try {
      const form = new FormData();
      form.append("audio", blob, "nagranie" + (blob.type.includes("mp4") ? ".m4a" : ".webm"));
      handleReply(await api("/api/talk", form));
    } catch (e) {
      add("sys err", e.message); setStatus("Spróbuj ponownie.");
    } finally { mic.disabled = false; }
  };

  mic.onclick = () => (recorder && recorder.state === "recording") ? stopRec() : startRec();

  // ---- wiadomość tekstowa ----
  const sendText = async () => {
    const text = $("text").value.trim();
    if (!text) return;
    unlockAudio();
    $("text").value = ""; add("me", text); setStatus("Pytam Claude…");
    try {
      const form = new FormData(); form.append("text", text);
      const data = await api("/api/text", form); data._typed = true; handleReply(data);
    } catch (e) { add("sys err", e.message); setStatus(""); }
  };
  $("send").onclick = sendText;
  $("text").addEventListener("keydown", (e) => { if (e.key === "Enter") sendText(); });

  $("reset").onclick = async () => {
    try { await api("/api/reset", new FormData()); } catch (e) { /* ignoruj */ }
    log.innerHTML = ""; add("sys", "Nowa rozmowa.");
  };

  if (!token) add("sys err", "Brak tokenu. Otwórz adres z ?token=… wypisany przez serwer.");
  if (!window.MediaRecorder) add("sys err", "Ta przeglądarka nie obsługuje nagrywania (MediaRecorder).");
  add("sys", "Gotowy. Rozmowa trafia do Claude Code na Twoim komputerze.");
})();
