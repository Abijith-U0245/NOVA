<![CDATA[<div align="center">

# 🛰️ NOVA

### **Edge-First Voice Activation System**

*Low-latency Keyword Spotting on ESP32-S3 + Cloud ASR Pipeline*

[![ISRO Smart Automation](https://img.shields.io/badge/ISRO-Smart%20Automation-orange?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJ3aGl0ZSI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiLz48L3N2Zz4=)](https://www.isro.gov.in/)
[![Problem Statement](https://img.shields.io/badge/PS-26172-blue?style=for-the-badge)]()
[![Platform](https://img.shields.io/badge/ESP32--S3-TFLite%20Micro-green?style=for-the-badge)]()
[![ASR](https://img.shields.io/badge/Whisper-medium-purple?style=for-the-badge)]()
[![Edge Impulse](https://img.shields.io/badge/Edge%20Impulse-KWS%20Model-red?style=for-the-badge)](https://studio.edgeimpulse.com/)

---

**NOVA** is a two-stage voice activation pipeline that runs custom **Keyword Spotting (KWS)** on a resource-constrained **ESP32-S3** microcontroller, then offloads full speech-to-text to **OpenAI Whisper** on a host PC — all in real time. Built for the **ISRO Smart Automation Challenge**, Problem Statement **#26172**: *Low-Latency Edge Voice Activator*.

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Challenge Constraints](#-challenge-constraints)
- [Hardware Requirements](#-hardware-requirements)
- [Software Requirements](#-software-requirements)
- [Repository Structure](#-repository-structure)
- [Setup & Installation](#-setup--installation)
  - [1. ESP32-S3 Firmware](#1-esp32-s3-firmware)
  - [2. PC Server & Dashboard](#2-pc-server--dashboard)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Dashboard](#-dashboard)
- [Configuration](#%EF%B8%8F-configuration)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

---

## 🔭 Overview

NOVA implements a **wake-word → ASR** pipeline optimised for edge deployment:

1. **Edge KWS** — An ESP32-S3 continuously listens via an INMP441 I2S MEMS microphone and runs a quantised TFLite Micro model (trained on [Edge Impulse](https://studio.edgeimpulse.com/)) to detect the wake word **"NOVA"** in real time.
2. **Cloud ASR** — Upon detection, the ESP32 sends an HTTP trigger to a host PC, which captures audio from its own microphone and transcribes it using **faster-whisper** (a CTranslate2-accelerated Whisper implementation).
3. **Live Dashboard** — A browser-based dashboard connects via WebSocket to display live transcripts, ESP32 telemetry (CPU, RAM, PSRAM, inference time), and end-to-end latency metrics.

The system supports **multilingual transcription** — English, Hindi, Tamil, Malayalam, and Kannada — with automatic language detection.

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          NOVA System Architecture                        │
│                                                                          │
│  ┌─────────────────────────┐          ┌─────────────────────────────┐    │
│  │     ESP32-S3 (Edge)     │  HTTP    │        Host PC (Server)     │    │
│  │                         │ trigger  │                             │    │
│  │  INMP441 Mic ─► I2S DMA│────────►│  server.py (WebSocket +     │    │
│  │       │                 │          │           HTTP listener)    │    │
│  │       ▼                 │ metrics  │       │                     │    │
│  │  TFLite Micro KWS      │────────►│       ▼                     │    │
│  │  (Edge Impulse model)   │          │  stream_to_vosk.py         │    │
│  │       │                 │          │  (PC mic capture)          │    │
│  │       ▼                 │          │       │                     │    │
│  │  Vote Window Filter     │          │       ▼                     │    │
│  │  (confidence + silence  │          │  faster-whisper (Whisper)   │    │
│  │   gating)               │          │  GPU/CPU transcription     │    │
│  │       │                 │          │       │                     │    │
│  │       ▼                 │          │       ▼                     │    │
│  │  HTTP POST /trigger ────┼──────►  │  WebSocket broadcast       │    │
│  │                         │          │       │                     │    │
│  └─────────────────────────┘          └───────┼─────────────────────┘    │
│                                               │                          │
│                                               ▼                          │
│                                  ┌─────────────────────────┐             │
│                                  │   dashboard.html         │             │
│                                  │                          │             │
│                                  │  • Live transcript       │             │
│                                  │  • ESP32 CPU/RAM gauges  │             │
│                                  │  • ASR latency chart     │             │
│                                  │  • Constraint pass/fail  │             │
│                                  └─────────────────────────┘             │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Challenge Constraints

NOVA was designed to meet the strict requirements of **ISRO Smart Automation Challenge PS-26172**:

| Constraint | Requirement | NOVA's Status |
|:---|:---|:---|
| **RAM Usage** | < 256 KB internal SRAM | ✅ ~200–220 KB (SRAM only; mic buffers offloaded to PSRAM) |
| **CPU Idle** | < 10% CPU usage at idle | ✅ KWS inference fits within each audio slice window |
| **Open Source** | All components must be open-source | ✅ Edge Impulse SDK, TFLite Micro, faster-whisper, Vosk |
| **Custom KWS** | Custom-trained keyword spotting model | ✅ Trained on Edge Impulse with custom "NOVA" dataset |

---

## 🔧 Hardware Requirements

| Component | Details |
|:---|:---|
| **Microcontroller** | ESP32-S3 (with PSRAM — e.g., ESP32-S3-DevKitC-1 N8R8/N16R8) |
| **Microphone** | INMP441 I2S MEMS digital microphone |
| **Host PC** | Linux/macOS/Windows with a microphone and Python 3.8+ |
| **GPU (optional)** | NVIDIA GPU with CUDA for accelerated Whisper inference |

### Wiring — INMP441 to ESP32-S3

| ESP32-S3 GPIO | INMP441 Pin | Function |
|:---|:---|:---|
| GPIO 15 | SCK | Bit Clock (BCLK) |
| GPIO 16 | WS | Word Select (LRCK) |
| GPIO 17 | SD | Serial Data (DOUT) |
| 3.3V | VDD | Power |
| GND | GND & L/R | Ground + Channel select (Left) |

> **Note:** Tie the INMP441 **L/R** pin to **GND** to select the left channel.

---

## 💻 Software Requirements

### ESP32 (Arduino IDE)

- **Arduino IDE** 2.x or PlatformIO
- **ESP32 Arduino Core** ≥ 2.0.4
- **Board Setting:** Tools → PSRAM → **OPI PSRAM** (required for your board variant)
- **Edge Impulse Library:** `nova_esp_inferencing` (included in this repo — add as ZIP library)

### Host PC (Python)

```
Python 3.8+
```

| Package | Install Command |
|:---|:---|
| `faster-whisper` | `pip install faster-whisper` |
| `websockets` | `pip install websockets` |
| `aiohttp` | `pip install aiohttp` |
| `psutil` | `pip install psutil` |
| `numpy` | `pip install numpy` |
| `sounddevice` | `pip install sounddevice` |

**One-liner:**
```bash
pip install faster-whisper websockets aiohttp psutil numpy sounddevice
```

> For NVIDIA GPU acceleration, also install `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` (pip packages). The server auto-detects and pre-loads CUDA libraries.

---

## 📁 Repository Structure

```
NOVA/
├── README.md                          # This file
├── server.py                          # WebSocket + HTTP server (Whisper ASR)
├── dashboard.html                     # Real-time monitoring dashboard
│
├── nova_trigger_esp32/                # ESP32-S3 wake-word detector (main firmware)
│   └── nova_trigger_esp32.ino         # Arduino sketch: I2S mic → KWS → HTTP trigger
│
├── nova_esp_inferencing/              # Edge Impulse Arduino library (KWS model)
│   ├── library.properties
│   ├── src/                           # Inferencing SDK + TFLite model
│   └── examples/                      # Platform-specific examples (ESP32, etc.)
│
├── nova-kws/                          # Edge Impulse C++ library export (KWS model)
│   ├── edge-impulse-sdk/              # Inference SDK
│   ├── model-parameters/              # Model config
│   └── tflite-model/                  # Quantised TFLite model file
│
├── example-standalone-inferencing/    # PC-side standalone KWS (C++ / Linux)
│   ├── source/                        # C++ main for Edge Impulse inference
│   ├── stream_to_vosk.py              # Mic → WebSocket audio streamer
│   ├── Makefile                       # Build system for C++ demo
│   └── build.sh / build.bat           # Build scripts (Linux / Windows)
│
├── nova_demo/                         # C++ live-mic KWS demo (ALSA + vote logic)
│   └── main.cpp                       # ALSA mic capture → EI classifier → trigger
│
├── inmp441_test/                      # INMP441 microphone hardware test utility
│   └── inmp441_test.ino               # Arduino sketch: VU meter + channel diagnostics
│
├── debug_mic.wav                      # Last captured audio segment (for debugging)
└── .gitignore
```

---

## 🚀 Setup & Installation

### 1. ESP32-S3 Firmware

1. **Open Arduino IDE** and install the **ESP32 board package** (Espressif Systems) from the Board Manager.

2. **Install the Edge Impulse library:**
   - In Arduino IDE: *Sketch → Include Library → Add .ZIP Library*
   - Select the `nova_esp_inferencing/` folder (or zip it first).

3. **Open** `nova_trigger_esp32/nova_trigger_esp32.ino`.

4. **Configure your network** (at the top of the file):
   ```cpp
   #define WIFI_SSID        "YourWiFiSSID"
   #define WIFI_PASSWORD    "YourWiFiPassword"
   #define PC_IP            "192.168.x.x"       // Your PC's IP on the same network
   #define PC_TRIGGER_PORT  8766
   #define TRIGGER_URL      "http://192.168.x.x:8766/trigger"
   #define METRICS_URL      "http://192.168.x.x:8766/esp32_metrics"
   ```

5. **Board settings** in Arduino IDE:
   - Board: **ESP32S3 Dev Module**
   - PSRAM: **OPI PSRAM**
   - Flash Size: **8MB** (or your board's flash size)
   - Partition Scheme: **Huge APP**

6. **Upload** the sketch to your ESP32-S3.

### 2. PC Server & Dashboard

1. **Install Python dependencies:**
   ```bash
   pip install faster-whisper websockets aiohttp psutil numpy sounddevice
   ```

2. **Start the server:**
   ```bash
   python3 server.py
   ```
   The server will:
   - Download the Whisper model on first run (~1.5 GB for `medium`)
   - Start a WebSocket server on `ws://0.0.0.0:8765`
   - Start an HTTP trigger endpoint on `http://0.0.0.0:8766`

3. **Open the dashboard:**
   - Open `dashboard.html` in any modern browser.
   - It auto-connects to `ws://127.0.0.1:8765`.

---

## 🎙 Usage

1. **Power on the ESP32-S3** — it connects to WiFi and begins listening.
2. **Ensure `server.py` is running** on your PC.
3. **Open `dashboard.html`** in a browser to monitor the system.
4. **Say "NOVA"** — the ESP32 detects the wake word and triggers the PC.
5. **Speak your command** — the PC captures audio from its microphone and transcribes it with Whisper.
6. **View the transcript** on the dashboard in real time.

### Pipeline Flow

```
You say "NOVA"  →  ESP32 KWS detects it  →  HTTP POST to PC
                                                    │
You speak a command  ←──────────────────────────────┘
        │
PC mic captures audio  →  Whisper transcribes  →  Dashboard shows result
```

---

## ⚙ How It Works

### 1. Wake-Word Detection (ESP32-S3)

- The **INMP441** microphone feeds 16 kHz, 16-bit PCM audio via I2S DMA.
- Audio is sliced into overlapping windows and fed to a **TFLite Micro** model (trained on Edge Impulse).
- A **sliding vote window** (5 slices, 1 vote required at ≥ 50% confidence) with a **mic amplitude gate** (peak > 1500) reduces false positives.
- A **3-second cooldown** prevents rapid re-triggers.
- On positive detection, an **HTTP POST** is sent to `server.py` on the host PC.

### 2. Memory Allocation Strategy

The ESP32-S3 has limited internal SRAM (~320 KB). NOVA uses a careful allocation strategy:

- **Tensor arena** → Allocated in PSRAM (512 KB — too large for SRAM). ESP-NN SIMD kernels are **disabled** because they require internal SRAM operands; TFLite reference kernels run correctly from PSRAM.
- **Mic ring buffers** → Allocated in PSRAM (no SIMD alignment requirement).
- **Model initialization** → Runs **before** WiFi connects to minimise SRAM fragmentation.

### 3. Audio Streaming & ASR (Host PC)

- `server.py` receives the trigger and launches `stream_to_vosk.py`.
- `stream_to_vosk.py` opens the PC's default microphone, streams raw PCM chunks over WebSocket.
- Streaming stops after **1 second of silence** or a **6-second maximum**.
- The server transcribes the collected audio with **faster-whisper** (CTranslate2 backend).
- Automatic language detection with corrections (e.g., Malay → Malayalam, Marathi → Hindi).
- Results are broadcast to all connected dashboard clients via WebSocket.

### 4. Telemetry

The ESP32 sends periodic JSON telemetry to `server.py`:
- Dual-core CPU load (estimated from inference-time-to-slice-time ratio)
- Internal SRAM usage (used/total KB)
- PSRAM usage (used/total KB)
- Inference time per audio slice (ms)

---

## 📊 Dashboard

The dashboard (`dashboard.html`) is a single-page, self-contained HTML file with:

| Feature | Description |
|:---|:---|
| **Mission Banner** | ISRO challenge context and constraint pills |
| **Status Indicator** | Real-time connection status (Connected / Streaming / Idle) |
| **Constraint Cards** | Live pass/fail indicators for RAM, CPU, and latency constraints |
| **CPU Gauge** | ESP32 dual-core CPU usage with color-coded thresholds |
| **RAM Gauge** | ESP32 internal SRAM usage with constraint limit overlay |
| **Latency Gauge** | End-to-end trigger → transcript latency (ms) |
| **Live Transcript** | Current partial/final transcript with language badge |
| **History Log** | Timestamped list of all recognised utterances |
| **Latency Sparkline** | Bar chart of recent latency samples with best/avg stats |
| **Starfield Animation** | Animated background for visual polish |

The dashboard falls back to **PC metrics** (via `psutil`) if the ESP32 is offline for > 8 seconds.

---

## ⚙️ Configuration

### ESP32 Firmware (`nova_trigger_esp32.ino`)

| Parameter | Default | Description |
|:---|:---|:---|
| `WIFI_SSID` | — | WiFi network name |
| `WIFI_PASSWORD` | — | WiFi password |
| `PC_IP` | — | Host PC IP address |
| `CONFIDENCE_THRESHOLD` | `0.50` | Min KWS confidence to count as a vote |
| `VOTE_WINDOW` | `5` | Number of inference slices in voting window |
| `VOTE_REQUIRED` | `1` | Votes needed to trigger |
| `MIC_AMPLITUDE_GATE` | `1500` | Min peak amplitude to consider a slice non-silent |
| `COOLDOWN_MS` | `3000` | Cooldown between triggers (ms) |

### Server (`server.py`)

| Parameter | Default | Description |
|:---|:---|:---|
| `WHISPER_MODEL_SIZE` | `"medium"` | Whisper model size (`base`, `small`, `medium`, `large-v3`) |
| `WHISPER_DEVICE` | `"cuda"` | Compute device (`cuda` / `cpu`) |
| `WHISPER_COMPUTE` | `"float16"` | Precision (`float16` for GPU, `int8` for CPU) |
| `SAMPLE_RATE` | `16000` | Audio sample rate (Hz) |
| `PORT` | `8765` | WebSocket server port |
| `HTTP_TRIGGER_PORT` | `8766` | HTTP trigger/telemetry port |
| `METRICS_INTERVAL` | `2.0` | Dashboard metrics broadcast interval (s) |

### Audio Streamer (`stream_to_vosk.py`)

| Parameter | Default | Description |
|:---|:---|:---|
| `SILENCE_AMPLITUDE_THRESHOLD` | `2500` | Peak amplitude below which counts as silence |
| `SILENCE_CUTOFF_SECONDS` | `1.0` | Seconds of silence before stopping |
| `MAX_STREAM_SECONDS` | `6.0` | Maximum recording duration |

---

## 🐛 Troubleshooting

| Issue | Solution |
|:---|:---|
| **ESP32 crashes on boot / "Failed to allocate"** | Ensure PSRAM is enabled in Arduino IDE (Tools → PSRAM → OPI PSRAM). Check the allocation strategy notes in the firmware source. |
| **KWS gives frozen/wrong output** | Do **not** redirect `ei_malloc` to PSRAM. ESP-NN is intentionally disabled; the reference kernels must be used with a PSRAM arena. |
| **WiFi won't connect** | Verify `WIFI_SSID` / `WIFI_PASSWORD`. The ESP32 restarts after 15 seconds if it can't connect. |
| **HTTP trigger fails (error -1)** | Check that `server.py` is running and `PC_IP` / `PC_TRIGGER_PORT` match the server. Ensure ESP32 and PC are on the same network. |
| **No audio / empty transcripts** | Run `inmp441_test.ino` to verify the INMP441 wiring. Check that the PC mic is accessible (try `arecord -l` on Linux). |
| **Whisper falls back to CPU** | Install `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` for CUDA. The server auto-retries on CPU if GPU fails. |
| **Dashboard shows "Disconnected"** | Ensure `server.py` is running on `ws://127.0.0.1:8765`. Check firewall rules. |
| **Wrong language detected** | Whisper may confuse similar languages. The server applies corrections (Malay→Malayalam, Marathi→Hindi) and retries unsupported languages as English. |

---

## 🧪 Utilities

### INMP441 Hardware Test

Upload `inmp441_test/inmp441_test.ino` to verify your microphone wiring:
- Prints live **left/right channel peak-to-peak** values on Serial Monitor (115200 baud).
- Use Arduino **Serial Plotter** for a real-time audio waveform.
- If both channels show near-zero values, check your wiring and L/R pin.

### PC-Side Standalone KWS Demo

The `nova_demo/main.cpp` runs the same Edge Impulse KWS model on a Linux PC using ALSA:
```bash
cd example-standalone-inferencing
make -j
./build/app
```
Useful for testing the model accuracy without the ESP32.

---

## 🙏 Acknowledgements

- **[Edge Impulse](https://edgeimpulse.com/)** — KWS model training & TFLite Micro SDK
- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** — CTranslate2-based Whisper implementation
- **[Espressif](https://www.espressif.com/)** — ESP32-S3 platform & Arduino core

---

## 📄 License

This project was built for the **ISRO Smart Automation Challenge** (Problem Statement #26172).  
The Edge Impulse SDK components are licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

---

<div align="center">

**Built with ❤️ for the ISRO Smart Automation Challenge**

*Say "NOVA" to activate* 🎙️

</div>
]]>
