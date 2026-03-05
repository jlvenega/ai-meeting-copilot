import os
import json
import base64
import threading
import queue
import asyncio
from typing import Dict, Any, List, Optional

import numpy as np
import sounddevice as sd
import websocket

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

from dotenv import load_dotenv
from openai import OpenAI

# Load .env
load_dotenv()

# =========================
# Config
# =========================
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

RATE = 24000
CHUNK_MS = 10
CHUNK_SAMPLES = int(RATE * CHUNK_MS / 1000)

REALTIME_URL = "wss://api.openai.com/v1/realtime?intent=transcription"
TRANSCRIBE_MODEL = "gpt-4o-mini-transcribe"
TEXT_MODEL = "gpt-5.2"

audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=2000)
translation_jobs: "queue.Queue[tuple[int, str]]" = queue.Queue(maxsize=2000)

app = FastAPI()


# =========================
# Helpers
# =========================
def safe_json_parse(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        return None


def list_input_devices() -> List[Dict[str, Any]]:
    devs = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            devs.append({"i": i, "name": d["name"]})
    return devs


def pick_blackhole_device_index() -> int:
    devs = list_input_devices()
    for d in devs:
        if "BlackHole" in d["name"]:
            return d["i"]
    if not devs:
        raise RuntimeError("No input audio devices found.")
    return devs[0]["i"]


def pcm16_b64(x_float32: np.ndarray) -> str:
    x = np.clip(x_float32, -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16).tobytes()
    return base64.b64encode(pcm).decode("ascii")


def normalize_newlines(s: str) -> str:
    # Some payloads can contain literal "\n"
    return (s or "").replace("\\n", "\n")


def translate_text(text_src: str, translate_to: str, meeting_context: str) -> str:
    prompt = (
        "You translate meeting transcript snippets.\n"
        "The MEETING CONTEXT may be in any language; use it only as background.\n"
        f"Translate the INPUT to {translate_to}. Keep meaning. Preserve line breaks if present.\n"
        "Return ONLY the translation, no extra text.\n\n"
        f"MEETING CONTEXT:\n{meeting_context or '(none)'}\n\n"
        f"INPUT:\n{text_src}\n"
    )
    r = client.responses.create(
        model=TEXT_MODEL,
        input=prompt,
        store=False,
    )
    return normalize_newlines((r.output_text or "").strip())


def summarize_turns(turns: List[Dict[str, str]], summary_lang: str, meeting_context: str) -> str:
    trimmed = turns[-220:]
    transcript = "\n".join([f"- {t.get('src','').strip()}" for t in trimmed if t.get("src", "").strip()])

    prompt = (
        "You summarize meeting transcripts.\n"
        "IMPORTANT:\n"
        "- Do NOT guess who said what.\n"
        "- If an owner is unclear, write 'Owner: Unknown'.\n"
        "- Focus on decisions, action items, risks, and open questions.\n"
        f"- Write in {summary_lang}.\n\n"
        "Return markdown with sections:\n"
        "## Summary\n"
        "## Decisions\n"
        "## Action items (Owner / Due)\n"
        "## Risks / Blockers\n"
        "## Open questions\n\n"
        f"MEETING CONTEXT:\n{meeting_context or '(none)'}\n\n"
        f"Transcript bullets:\n{transcript}\n"
    )
    r = client.responses.create(
        model=TEXT_MODEL,
        input=prompt,
        store=False,
    )
    return normalize_newlines((r.output_text or "").strip())


# =========================
# UI
# =========================
INDEX_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Meeting Copilot (Paired History)</title>
  <style>
    body { font-family: -apple-system, system-ui, Arial; margin: 18px; }
    h2 { margin: 0; font-size: 18px; }
    .muted { color:#666; font-size: 12px; }

    .card { border:1px solid #ddd; border-radius:12px; padding:12px; width: 100%; margin-top: 12px; }
    .pill { display:inline-block; padding:2px 8px; border:1px solid #ddd; border-radius:999px; font-size:12px; }
    button { padding:7px 10px; border-radius:10px; border:1px solid #ccc; background:#fff; cursor:pointer; font-size: 12px; }
    select { margin-left: 6px; font-size: 12px; }
    textarea { width: 100%; font-size: 12px; }
    pre { white-space: pre-wrap; font-family: inherit; margin: 0;}

    .controls { display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-top:8px; }

    /* Paired history table */
    .history {
      height: 66vh;
      overflow: auto;
      border: 1px solid #eee;
      border-radius: 10px;
      background: #fafafa;
    }
    .headerRow, .entryRow {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      padding: 10px;
      border-bottom: 1px solid #eee;
    }
    .headerRow {
      position: sticky;
      top: 0;
      background: #fff;
      z-index: 10;
      font-size: 12px;
      color: #444;
      font-weight: 600;
    }
    .cell {
      white-space: pre-wrap;
      font-size: 12px;
      line-height: 1.25;
      padding: 6px 8px;
      border-radius: 10px;
      background: #fff;
      border: 1px solid #eee;
      min-height: 18px;
    }
    .placeholder {
      color: #888;
      font-style: italic;
    }

    /* Summary full-width */
    .summaryBox {
      font-family: inherit;
      font-size: 12px;
      border: 1px dashed #ddd;
      border-radius: 10px;
      padding: 10px;
      background: #fff;
      line-height: 1.25;
      min-height: 120px;
      white-space: pre-wrap;
    }
  </style>
</head>
<body>
  <h2>Meeting Copilot</h2>
  <div class="muted">Each turn becomes a paired row: transcript (left) + translation (right). Summary is full-width below.</div>
  <div id="status" class="muted" style="margin-top:6px;"></div>

  <div class="card">
    <div class="muted">Settings</div>
    <div class="controls">
      <label class="pill">ASR:
        <select id="asrLang">
          <option value="auto" selected>auto</option>
          <option value="es">es</option>
          <option value="en">en</option>
        </select>
      </label>

      <label class="pill">Translate to:
        <select id="translateTo">
          <option value="es" selected>es</option>
          <option value="en">en</option>
        </select>
      </label>

      <label class="pill">Summary:
        <select id="summaryLang">
          <option value="en" selected>en</option>
          <option value="es">es</option>
        </select>
      </label>

      <button onclick="applySettings()">Apply</button>
      <button onclick="generateSummary()">Generate summary</button>
      <button onclick="clearLogs()">Clear</button>
      <button onclick="copyTranscript()">Copy transcript</button>
    </div>

    <div style="margin-top:10px;">
      <div class="muted">Meeting context (optional; ES/EN ok)</div>
      <textarea id="meetingContext" style="height:62px;" placeholder="Ej: Reunión semanal de plataforma de datos..."></textarea>
      <div class="controls" style="margin-top:8px;">
        <button onclick="setContext()">Set context</button>
        <span class="muted">Steers translation & summary.</span>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="muted">Paired history</div>
    <div id="history" class="history" style="margin-top:8px;">
      <div class="headerRow">
        <div>Transcript</div>
        <div>Translation</div>
      </div>
      <div id="rows"></div>
    </div>
  </div>

  <div class="card">
    <div class="muted">Summary (full width)</div>
    <pre id="summary" class="summaryBox" style="margin-top:8px;"></pre>
  </div>

<script>
  const ws = new WebSocket(`ws://${location.host}/ws`);

  function normalizeNewlines(s){
    return (s || "").replace(/\\n/g, "\n");
  }

  const state = {
    order: [],
    srcById: {},
    trById: {},
    rowEls: {} // id -> {srcEl, trEl}
  };

  function makeRow(id) {
    const row = document.createElement("div");
    row.className = "entryRow";

    const src = document.createElement("div");
    src.className = "cell";
    src.textContent = state.srcById[id] || "";

    const tr = document.createElement("div");
    tr.className = "cell placeholder";
    tr.textContent = state.trById[id] || "…";

    row.appendChild(src);
    row.appendChild(tr);

    state.rowEls[id] = { srcEl: src, trEl: tr };

    return row;
  }

  function scrollToBottom() {
    const history = document.getElementById("history");
    history.scrollTop = history.scrollHeight;
  }

  ws.onopen = () => {
    document.getElementById("status").textContent = "Connected.";
    applySettings();
  };

  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);

    if (msg.type === "status") {
      document.getElementById("status").textContent = msg.text || "";
      return;
    }

    if (msg.type === "turn") {
      const id  = msg.id;
      const src = normalizeNewlines(msg.src || "");

      state.order.push(id);
      state.srcById[id] = src;
      state.trById[id] = "…";

      const rows = document.getElementById("rows");
      rows.appendChild(makeRow(id));
      scrollToBottom();
      return;
    }

    if (msg.type === "translation") {
      const id = msg.id;
      const tr = normalizeNewlines(msg.translation || "");
      state.trById[id] = tr;

      const el = state.rowEls[id];
      if (el && el.trEl) {
        el.trEl.classList.remove("placeholder");
        el.trEl.textContent = tr;
      }
      return;
    }

    if (msg.type === "summary") {
      document.getElementById("summary").textContent = normalizeNewlines(msg.text || "");
      return;
    }

    if (msg.type === "clear_ack") {
      state.order = [];
      state.srcById = {};
      state.trById = {};
      state.rowEls = {};
      document.getElementById("rows").innerHTML = "";
      document.getElementById("summary").textContent = "";
      return;
    }
  };

  function applySettings() {
    const settings = {
      asrLang: document.getElementById("asrLang").value,
      translateTo: document.getElementById("translateTo").value,
      summaryLang: document.getElementById("summaryLang").value
    };
    ws.send(JSON.stringify({ type: "settings", settings }));
  }

  function setContext() {
    const ctx = document.getElementById("meetingContext").value || "";
    ws.send(JSON.stringify({ type: "context", context: ctx }));
  }

  function generateSummary() {
    ws.send(JSON.stringify({ type: "summary_request" }));
  }

  function clearLogs() {
    ws.send(JSON.stringify({ type: "clear" }));
  }

  function copyTranscript() {
    const txt = state.order.map(id => state.srcById[id] || "").join("\n\n");
    navigator.clipboard.writeText(txt);
  }
</script>
</body>
</html>
"""


@app.get("/")
def home():
    return HTMLResponse(INDEX_HTML)


@app.get("/devices")
def devices():
    return {"devices": list_input_devices()}


# =========================
# WS Endpoint
# =========================
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()

    state: Dict[str, Any] = {
        "settings": {
            "asrLang": "auto",
            "translateTo": "es",
            "summaryLang": "en",
        },
        "meeting_context": "",
        "turns": [],  # [{id, src, translation}]
    }

    # server -> browser queue
    send_q: asyncio.Queue[str] = asyncio.Queue()

    async def sender():
        while True:
            msg = await send_q.get()
            await ws.send_text(msg)

    sender_task = asyncio.create_task(sender())

    def send_from_thread(payload: dict):
        s = json.dumps(payload, ensure_ascii=False)
        asyncio.run_coroutine_threadsafe(send_q.put(s), loop)

    # pick audio input device (BlackHole preferred)
    try:
        device_index = pick_blackhole_device_index()
    except Exception as e:
        await ws.send_text(json.dumps({"type": "status", "text": f"Audio device error: {e}"}))
        sender_task.cancel()
        return

    await ws.send_text(json.dumps({
        "type": "status",
        "text": f"Using input device index {device_index}. Route Meet/Teams audio to Multi-Output incl. BlackHole."
    }))

    # Turn IDs
    turn_counter_box = {"v": 0}
    counter_lock = threading.Lock()

    # Translation worker (async)
    def translation_worker():
        while True:
            tid, txt = translation_jobs.get()
            try:
                tr = translate_text(
                    text_src=txt,
                    translate_to=state["settings"].get("translateTo", "es"),
                    meeting_context=state["meeting_context"]
                )
            except Exception as e:
                tr = f"[Translation error: {e}]"

            for t in reversed(state["turns"]):
                if t.get("id") == tid:
                    t["translation"] = tr
                    break

            send_from_thread({"type": "translation", "id": tid, "translation": tr})

    threading.Thread(target=translation_worker, daemon=True).start()

    # Realtime transcription websocket
    headers = [f"Authorization: Bearer {OPENAI_API_KEY}"]
    current_item = {"item_id": None, "text": ""}

    def rt_send(rtws, obj):
        rtws.send(json.dumps(obj))

    def configure_transcription_session(rtws):
        lang = state["settings"].get("asrLang", "auto")
        transcription_obj = {"model": TRANSCRIBE_MODEL}
        if lang != "auto":
            transcription_obj["language"] = lang

        rt_send(rtws, {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": RATE},
                        "transcription": transcription_obj,
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.5,
                            "prefix_padding_ms": 250,
                            "silence_duration_ms": 300
                        }
                    }
                }
            }
        })

    def on_open(rtws):
        configure_transcription_session(rtws)

        def audio_callback(indata, frames, time_info, status):
            mono = indata[:, 0].astype(np.float32)
            for start in range(0, len(mono), CHUNK_SAMPLES):
                chunk = mono[start:start + CHUNK_SAMPLES]
                if len(chunk) < CHUNK_SAMPLES:
                    break
                try:
                    audio_q.put_nowait(chunk)
                except queue.Full:
                    pass

        def audio_thread():
            with sd.InputStream(
                device=device_index,
                channels=1,
                samplerate=RATE,
                blocksize=CHUNK_SAMPLES,
                callback=audio_callback,
            ):
                while True:
                    chunk = audio_q.get()
                    rt_send(rtws, {"type": "input_audio_buffer.append", "audio": pcm16_b64(chunk)})

        threading.Thread(target=audio_thread, daemon=True).start()

    def on_message(rtws, message):
        data = json.loads(message)
        t = data.get("type", "")

        if t == "conversation.item.input_audio_transcription.delta":
            # We no longer render "live" in UI; do nothing
            return

        if t == "conversation.item.input_audio_transcription.completed":
            transcript = normalize_newlines((data.get("transcript") or "").strip())
            if not transcript:
                return

            with counter_lock:
                turn_counter_box["v"] += 1
                tid = turn_counter_box["v"]

            # Send turn immediately (translation will follow)
            state["turns"].append({"id": tid, "src": transcript, "translation": ""})
            send_from_thread({"type": "turn", "id": tid, "src": transcript, "translation": ""})

            try:
                translation_jobs.put_nowait((tid, transcript))
            except queue.Full:
                send_from_thread({"type": "translation", "id": tid, "translation": "[Translation queue full]"})


    def on_error(rtws, err):
        send_from_thread({"type": "status", "text": f"Realtime error: {err}"})


    rtws = websocket.WebSocketApp(
        REALTIME_URL,
        header=headers,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
    )
    threading.Thread(target=rtws.run_forever, daemon=True).start()

    # UI commands
    try:
        while True:
            incoming = await ws.receive_text()
            msg = safe_json_parse(incoming) or {}
            mtype = msg.get("type")

            if mtype == "settings":
                new = msg.get("settings", {})
                if isinstance(new, dict):
                    for k in ["asrLang", "translateTo", "summaryLang"]:
                        if k in new:
                            state["settings"][k] = str(new[k])

                    await ws.send_text(json.dumps({"type": "status", "text": f"Settings updated: {state['settings']}"}))
                    try:
                        configure_transcription_session(rtws)
                    except Exception:
                        pass

            elif mtype == "context":
                ctx = msg.get("context", "")
                if isinstance(ctx, str):
                    state["meeting_context"] = ctx.strip()
                    await ws.send_text(json.dumps({"type": "status", "text": "Context set ✅ (ES/EN ok)"}))

            elif mtype == "summary_request":
                if not state["turns"]:
                    await ws.send_text(json.dumps({"type": "summary", "text": "No turns captured yet."}))
                else:
                    try:
                        out = summarize_turns(
                            turns=state["turns"],
                            summary_lang=state["settings"].get("summaryLang", "en"),
                            meeting_context=state["meeting_context"]
                        )
                        await ws.send_text(json.dumps({"type": "summary", "text": out}))
                    except Exception as e:
                        await ws.send_text(json.dumps({"type": "summary", "text": f"Summary error: {e}"}))

            elif mtype == "clear":
                state["turns"] = []
                current_item["item_id"] = None
                current_item["text"] = ""

                # Clear queued translation jobs (best effort)
                try:
                    while True:
                        translation_jobs.get_nowait()
                except queue.Empty:
                    pass

                send_from_thread({"type": "clear_ack"})

    except Exception:
        pass
    finally:
        try:
            sender_task.cancel()
        except Exception:
            pass