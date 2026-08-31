#!/usr/bin/env python3
"""
server.py — Whisper (faster-whisper) WebSocket transcription server
            with live CPU/RAM/latency metrics broadcast to dashboard.

Two kinds of clients connect to this same server:
1. The KWS trigger's audio streamer (stream_to_vosk.py) — sends raw PCM bytes.
2. The browser dashboard (dashboard.html) — sends "register_dashboard" on connect,
   then listens for broadcast events.

Setup:
    pip install faster-whisper websockets psutil --break-system-packages

The Whisper model is downloaded automatically on first run (~150MB for base.en).

Run:
    python3 server.py
Then open dashboard.html in a browser.
"""

import asyncio
import json
import os
import sys
import importlib
import glob
import ctypes

# Auto-resolve and pre-load CUDA shared libraries from pip packages for faster-whisper/ctranslate2
try:
    nvidia_cublas = importlib.import_module("nvidia.cublas")
    nvidia_cudnn = importlib.import_module("nvidia.cudnn")
    cublas_dir = os.path.join(list(nvidia_cublas.__path__)[0], "lib")
    cudnn_dir = os.path.join(list(nvidia_cudnn.__path__)[0], "lib")
    os.environ["LD_LIBRARY_PATH"] = f"{cublas_dir}:{cudnn_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"

    # Pre-load shared libraries into process memory so ctranslate2 dlopen finds libcublas.so.12
    for path in glob.glob(os.path.join(cublas_dir, "*.so*")) + glob.glob(os.path.join(cudnn_dir, "*.so*")):
        try:
            ctypes.CDLL(path)
        except Exception:
            pass
except (ImportError, IndexError, AttributeError):
    pass

import time
import threading
import numpy as np

try:
    import websockets
except ImportError:
    print("Missing: pip install websockets --break-system-packages")
    sys.exit(1)

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    print("[warn] psutil not available — CPU/RAM metrics disabled.")

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Missing: pip install faster-whisper --break-system-packages")
    sys.exit(1)

# ── Config ─────────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = "medium"       # base (multilingual) / small / medium
WHISPER_DEVICE     = "cuda"         # "cuda" for GPU, "cpu" for CPU-only
WHISPER_COMPUTE    = "float16"      # float16 (GPU) / int8 (CPU)
SAMPLE_RATE        = 16000          # Hz — must match stream_to_vosk.py
HOST               = "0.0.0.0"
PORT               = 8765
METRICS_INTERVAL   = 2.0            # seconds between metric broadcasts

LANG_MAP = {
    "en": "English 🇬🇧",
    "hi": "Hindi 🇮🇳",
    "ta": "Tamil 🇮🇳",
    "ml": "Malayalam 🇮🇳",
    "kn": "Kannada 🇮🇳",
}

LANGUAGE_CORRECTIONS = {
    "ms": "ml",  # Malay -> Malayalam
    "mr": "hi",  # Marathi -> Hindi
    "ne": "hi",  # Nepali -> Hindi
    "sa": "hi",  # Sanskrit -> Hindi
}
# ───────────────────────────────────────────────────────────────────────────

dashboard_clients = set()

# Latency tracking (trigger → first transcript result)
_trigger_ts      = None   # float or None
_last_latency_ms = None   # float or None


def load_model():
    print(f"Loading Whisper '{WHISPER_MODEL_SIZE}' on {WHISPER_DEVICE} …")
    t0 = time.time()
    try:
        m = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE,
                         compute_type=WHISPER_COMPUTE)
        # Warm-up dummy inference to verify GPU execution works
        dummy_audio = np.zeros(16000, dtype=np.float32)
        list(m.transcribe(dummy_audio, language="en", vad_filter=False)[0])
    except Exception as e:
        print(f"[warn] CUDA initialization or test failed ({e}), falling back to CPU/int8")
        m = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    elapsed = time.time() - t0
    print(f"Whisper model ready in {elapsed:.1f}s")
    return m


# ── Broadcast helper ────────────────────────────────────────────────────────
async def broadcast(payload: dict):
    if not dashboard_clients:
        return
    msg = json.dumps(payload)
    dead = set()
    for client in set(dashboard_clients):   # snapshot — avoid size-change race
        try:
            await client.send(msg)
        except websockets.exceptions.ConnectionClosed:
            dead.add(client)
    dashboard_clients.difference_update(dead)


# ── Metrics broadcaster ─────────────────────────────────────────────────────
async def metrics_broadcaster():
    """Pushes CPU %, RAM, and last ASR latency every METRICS_INTERVAL seconds."""
    if _HAS_PSUTIL:
        psutil.cpu_percent(interval=None)   # warm-up; first call always 0.0
    while True:
        await asyncio.sleep(METRICS_INTERVAL)
        if not dashboard_clients:
            continue
        if _HAS_PSUTIL:
            cpu  = psutil.cpu_percent(interval=None)
            vm   = psutil.virtual_memory()
            ram_used_mb  = vm.used  / 1024 / 1024
            ram_total_mb = vm.total / 1024 / 1024
            ram_pct      = vm.percent
        else:
            cpu = ram_used_mb = ram_total_mb = ram_pct = None

        await broadcast({
            "type":         "metrics",
            "cpu_pct":      cpu,
            "ram_used_mb":  round(ram_used_mb,  1) if ram_used_mb  is not None else None,
            "ram_total_mb": round(ram_total_mb, 1) if ram_total_mb is not None else None,
            "ram_pct":      ram_pct,
            "latency_ms":   round(_last_latency_ms, 1) if _last_latency_ms is not None else None,
            "ts":           time.time(),
        })


# ── Whisper transcription (runs in thread pool) ─────────────────────────────
def _transcribe(audio_np: np.ndarray) -> tuple[str, str]:
    """Blocking call — run via loop.run_in_executor so it doesn't block asyncio."""
    if audio_np.size == 0:
        return "", "Unknown"
    try:
        segments, info = whisper_model.transcribe(
            audio_np,
            beam_size=5,
            condition_on_previous_text=False,
            no_speech_threshold=0.8,
            log_prob_threshold=-0.8,
            vad_filter=True,           # skip silent sections
            vad_parameters=dict(min_silence_duration_ms=300),
        )
        text = " ".join(s.text.strip() for s in segments).strip()

        # If vad_filter dropped quiet spoken text, fallback to vad_filter=False
        if not text and audio_np.size > 16000 * 0.5:
            segments, info = whisper_model.transcribe(
                audio_np,
                beam_size=5,
                condition_on_previous_text=False,
                no_speech_threshold=0.8,
                log_prob_threshold=-0.8,
                vad_filter=False,
            )
            text = " ".join(s.text.strip() for s in segments).strip()

        lang_code = info.language
        lang_code = LANGUAGE_CORRECTIONS.get(lang_code, lang_code)

        if lang_code not in ["en", "hi", "ta", "ml", "kn"]:
            print(f"[server] Auto-detected '{lang_code}' (outside target). Retrying as English...")
            try:
                segments, info = whisper_model.transcribe(
                    audio_np,
                    language="en",
                    beam_size=5,
                    condition_on_previous_text=False,
                    no_speech_threshold=0.8,
                    log_prob_threshold=-0.8,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=300),
                )
                text = " ".join(s.text.strip() for s in segments).strip()
            except Exception:
                pass
            lang_code = "en"

        lang_name = LANG_MAP.get(lang_code, lang_code.upper())
        print(f"[server] Detected language: {lang_name} (prob: {info.language_probability:.2f})")
        return text, lang_name
    except Exception as e:
        print(f"[server] Transcription error: {e}")
        return "", "Unknown"


# ── Connection handler ───────────────────────────────────────────────────────
async def handle_connection(websocket):
    global _trigger_ts, _last_latency_ms

    client_addr = websocket.remote_address

    # First message determines role
    try:
        first_message = await asyncio.wait_for(websocket.recv(), timeout=5)
    except asyncio.TimeoutError:
        first_message = None

    # ── Dashboard client ──────────────────────────────────────────────────
    if first_message == "register_dashboard":
        print(f"[server] dashboard connected: {client_addr}")
        dashboard_clients.add(websocket)
        await broadcast({"type": "status", "text": "online", "ts": time.time()})
        try:
            async for _ in websocket:
                pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            dashboard_clients.discard(websocket)
            print(f"[server] dashboard disconnected: {client_addr}")
        return

    # ── Audio-streaming (KWS trigger) client ─────────────────────────────
    print(f"[server] audio client connected: {client_addr}")
    _trigger_ts      = time.time()
    _last_latency_ms = None
    await broadcast({"type": "status", "text": "listening", "ts": time.time()})

    # Send a "processing" pseudo-partial so dashboard shows activity immediately
    await broadcast({"type": "partial", "text": "🎙️ Listening…", "ts": time.time()})

    # Collect all raw PCM int16 bytes into a list of chunks
    pcm_chunks = []

    if isinstance(first_message, (bytes, bytearray)):
        pcm_chunks.append(first_message)

    try:
        async for message in websocket:
            if isinstance(message, (bytes, bytearray)):
                pcm_chunks.append(message)
    except websockets.exceptions.ConnectionClosed:
        pass

    # ── Transcribe after stream ends ──────────────────────────────────────
    if pcm_chunks:
        await broadcast({"type": "partial", "text": "⏳ Transcribing…", "ts": time.time()})

        # Convert raw int16 PCM → float32 normalised [-1, 1]
        raw = b"".join(pcm_chunks)
        audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        loop = asyncio.get_event_loop()
        t0   = time.perf_counter()
        text, lang = await loop.run_in_executor(None, _transcribe, audio_np)
        transcribe_ms = (time.perf_counter() - t0) * 1000

        if _trigger_ts is not None:
            _last_latency_ms = (time.time() - _trigger_ts) * 1000
            print(f"[server] end-to-end latency: {_last_latency_ms:.0f} ms  "
                  f"(whisper inference: {transcribe_ms:.0f} ms)")

        if text:
            print(f"[FINAL] [{lang}] {text}")
            await broadcast({"type": "final", "text": text, "language": lang, "ts": time.time()})
        else:
            print("[server] empty transcript (silence or no speech detected)")
            await broadcast({"type": "partial", "text": "", "ts": time.time()})

    print(f"[server] audio client disconnected: {client_addr}")
    _trigger_ts = None
    await broadcast({"type": "status", "text": "idle", "ts": time.time()})


# ── Main ─────────────────────────────────────────────────────────────────────
async def main():
    print(f"NOVA Whisper WebSocket server on ws://{HOST}:{PORT}")
    async with websockets.serve(handle_connection, HOST, PORT):
        asyncio.create_task(metrics_broadcaster())
        print("Ready. Waiting for KWS trigger and/or dashboard to connect…")
        await asyncio.Future()   # run forever


if __name__ == "__main__":
    whisper_model = load_model()
    asyncio.run(main())
