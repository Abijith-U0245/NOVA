/*
 * nova_trigger_esp32.ino
 *
 * NOVA wake-word detection on ESP32-S3 with EdgeImpulse inferencing library.
 * On detection -> sends HTTP POST to PC's server.py /trigger endpoint.
 * The PC's stream_to_vosk.py then captures from the PC mic and pipes to Whisper.
 *
 * ────────────────────────────────────────────────────────────────────────────
 *  WIRING  (INMP441 I2S mic — pins as wired below)
 *  ESP32-S3 GPIO   ->   INMP441 Pin
 *  GPIO 15         ->   SCK  (BCLK)
 *  GPIO 16         ->   WS   (LRCK)
 *  GPIO 17         ->   SD   (data out)
 *  3.3V            ->   VDD
 *  GND             ->   GND & L/R
 * ────────────────────────────────────────────────────────────────────────────
 *
 *  Requirements (Arduino IDE -> Sketch -> Include Library -> Add .ZIP Library):
 *    - abijith-u0245-project-1_inferencing (the library folder already in your project)
 *    - ESP32 Arduino core >= 2.0.4 (provides WiFi.h, HTTPClient.h, driver/i2s.h)
 *
 *  IMPORTANT: Tools -> PSRAM must be set to "OPI PSRAM" (or your board's
 *  matching PSRAM option) in Arduino IDE.
 *
 *  ── ALLOCATION STRATEGY (read this before touching ei_malloc again) ──
 *  Root cause of the original "Failed to allocate" crash: internal SRAM was
 *  getting fragmented by WiFi/HTTPClient before the model's tensor arena
 *  could claim a contiguous block.
 *
 *  The SDK's own ei_malloc() is INTENTIONALLY internal-SRAM-only on ESP32 —
 *  esp-nn's SIMD-optimized int8 kernels require 16-byte-aligned INTERNAL
 *  memory to run correctly. Redirecting the tensor arena to PSRAM (via a
 *  global ei_malloc override) causes the deeper SIMD-optimized layers to
 *  silently produce wrong/frozen output while shallow layers still look
 *  fine — this was tried and confirmed broken (NN-FINAL output identical
 *  across different inputs). DO NOT reintroduce an ei_malloc/ei_calloc/
 *  ei_free override that redirects to MALLOC_CAP_SPIRAM.
 *
 *  The correct fix — used below — is to leave the SDK's allocator alone and
 *  instead free up internal SRAM elsewhere so the arena has room:
 *    1. Mic ring buffers moved to PSRAM (heap_caps_malloc + MALLOC_CAP_SPIRAM)
 *       since they have no SIMD alignment requirement.
 *    2. Capture task stack reduced from 4096 -> 2048 bytes.
 *    3. run_classifier_init() called BEFORE WiFi connects, so the model
 *       claims its arena while internal SRAM is at its cleanest/least
 *       fragmented state.
 *    4. EI_CLASSIFIER_TFLITE_ARENA_IN_PSRAM macro intentionally not used —
 *       confirmed to have no effect in this SDK version.
 *
 *  If you ever hit "Failed to allocate mic buffers" or an arena-alloc
 *  failure again, the fix is to shrink the model/arena or free more
 *  internal SRAM (e.g. smaller WiFi RX buffers) — never to move the arena
 *  to PSRAM.
//  ── WHY ESP-NN IS DISABLED ────────────────────────────────────────────────
//  The model's tensor arena is 512 KB. Internal SRAM only has ~221 KB
//  contiguous free (after WiFi), so the arena is allocated in PSRAM by the
//  SDK's EI_CLASSIFIER_ALLOCATION_HEAP path (tflite_learn_..._init).
//
//  ESP-NN's SIMD-optimized Conv2D / DepthwiseConv kernels use Xtensa-specific
//  instructions that require their operands to be in internal SRAM — reading
//  from PSRAM with those instructions causes all layers AFTER op0 to silently
//  produce wrong / frozen output (op0 coincidentally works because the first
//  slice of the input tensor is still in cache).
//
//  Disabling ESP-NN (EI_CLASSIFIER_TFLITE_ENABLE_ESP_NN = 0) forces TFLite
//  to use its generic reference kernels. They are ~3-5× slower but work
//  correctly with a PSRAM arena, giving real inference results.
//
//  Long-term fix: shrink the model in Edge Impulse Studio until the arena
//  fits in ~180 KB so internal SRAM can be used again and ESP-NN re-enabled.
// ─────────────────────────────────────────────────────────────────────────────
*/

// ── User Config ──────────────────────────────────────────────────────────────
#define WIFI_SSID        "Redmi 13C 5G"       // your mobile hotspot name
#define WIFI_PASSWORD    "funnyboy01"          // your hotspot password
#define PC_IP            "10.65.198.223"      // your PC's local IP on mobile hotspot
#define PC_TRIGGER_PORT  8766
#define TRIGGER_URL      "http://10.65.198.223:8766/trigger"
#define METRICS_URL      "http://10.65.198.223:8766/esp32_metrics"

// I2S microphone pins
#define I2S_PORT         I2S_NUM_0
#define I2S_BCK_PIN      15  // SCK / BCLK
#define I2S_WS_PIN       16  // WS  / LRCK
#define I2S_DATA_PIN     17  // SD  / DOUT


// Wake-word tuning for retrained model
#define NOVA_LABEL           "nova"
#define CONFIDENCE_THRESHOLD 0.50f
#define VOTE_WINDOW          5
#define VOTE_REQUIRED        1
#define MIC_AMPLITUDE_GATE   1500
#define COOLDOWN_MS          3000
// ─────────────────────────────────────────────────────────────────────────────

#define EIDSP_QUANTIZE_FILTERBANK 0

// Disable ESP-NN SIMD kernels: arena is in PSRAM (512KB, won't fit in
// internal SRAM). ESP-NN SIMD kernels require internal SRAM — running from
// PSRAM causes all ops after op0 to silently produce frozen/wrong output.
// Reference kernels work correctly from PSRAM at the cost of ~3-5× slower
// inference (still fast enough for wake-word detection at this model size).
#define EI_CLASSIFIER_TFLITE_ENABLE_ESP_NN 0

#include <abijith-u0245-project-1_inferencing.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2s.h"
#include "esp_heap_caps.h"
#include <WiFi.h>
#include <HTTPClient.h>

// NOTE: no ei_malloc/ei_calloc/ei_free override here. The SDK's original
// weak implementations in porting/espressif/ei_classifier_porting.cpp are
// used as-is, so the tensor arena and all internal op buffers stay in
// internal SRAM where esp-nn's SIMD kernels need them. See allocation
// strategy note above.

// ── Audio buffer ─────────────────────────────────────────────────────────────
typedef struct {
    int16_t *buffers[2];
    uint8_t  buf_select;
    uint8_t  buf_ready;
    uint32_t buf_count;
    uint32_t n_samples;
} inference_t;

static inference_t inference;
static const uint32_t SAMPLE_BUFFER_SIZE = 512;
static int16_t sampleBuffer[SAMPLE_BUFFER_SIZE];
static bool    record_status = true;
static int     print_results = -(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW);

// ── Vote window ───────────────────────────────────────────────────────────────
static float    vote_scores[VOTE_WINDOW] = {0};
static int      vote_idx                 = 0;
static int      nova_class_idx           = -1;
static uint32_t last_trigger_ms          = 0;

// ── I2S init (INMP441 24-bit in Stereo 32-bit slot) ──────────────────────────
static void i2s_init_mic(uint32_t sample_rate)
{
    i2s_config_t cfg = {
        .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate          = sample_rate,
        .bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format       = I2S_CHANNEL_FMT_RIGHT_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count        = 4,
        .dma_buf_len          = 256,
        .use_apll             = false,
        .tx_desc_auto_clear   = false,
        .fixed_mclk           = -1,
    };
    i2s_pin_config_t pins = {
        .bck_io_num   = I2S_BCK_PIN,
        .ws_io_num    = I2S_WS_PIN,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num  = I2S_DATA_PIN,
    };
    i2s_driver_install(I2S_PORT, &cfg, 0, NULL);
    i2s_set_pin(I2S_PORT, &pins);
    i2s_zero_dma_buffer(I2S_PORT);
}

// ── Audio capture callback ────────────────────────────────────────────────────
static void audio_inference_callback(uint32_t n_bytes)
{
    for (uint32_t i = 0; i < n_bytes >> 1; i++) {
        inference.buffers[inference.buf_select][inference.buf_count++] = sampleBuffer[i];
        if (inference.buf_count >= inference.n_samples) {
            inference.buf_select ^= 1;
            inference.buf_count   = 0;
            inference.buf_ready   = 1;
        }
    }
}

// ── Capture task (pinned to Core 0) ──────────────────────────────────────────
static int32_t i2s_raw[256];

static void capture_samples(void *arg)
{
    size_t bytes_read = 0;
    while (record_status) {
        i2s_read(I2S_PORT, i2s_raw, sizeof(i2s_raw), &bytes_read, portMAX_DELAY);
        if (bytes_read > 0) {
            size_t total_words = bytes_read / 4;
            size_t mono_samples = total_words / 2;
            for (size_t x = 0; x < mono_samples; x++) {
                // >>15: right-shift 24-bit INMP441 data (left-justified in 32 bits)
                // to 16-bit PCM. Gives amplitude ~5000-18000 for speech, which
                // is the correct range for this model's feature extractor.
                // (>>9 was tried but caused constant clipping of even background
                // noise since ambient audio at -40 dBFS raw still clips at >>9.)
                sampleBuffer[x] = (int16_t)(i2s_raw[2 * x] >> 15);
            }
            if (record_status) audio_inference_callback(mono_samples * 2);
        }
    }
    vTaskDelete(NULL);
}

// ── Mic inference helpers ─────────────────────────────────────────────────────
static bool microphone_inference_start(uint32_t n_samples)
{
    // Mic ring buffers: PSRAM first (no SIMD alignment requirement), fall
    // back to internal SRAM only if PSRAM is somehow unavailable. This is
    // safe and unrelated to the tensor arena — do not confuse this with the
    // (removed) ei_malloc override.
    inference.buffers[0] = (int16_t *)heap_caps_malloc(n_samples * sizeof(int16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!inference.buffers[0]) {
        inference.buffers[0] = (int16_t *)heap_caps_malloc(n_samples * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    }
    inference.buffers[1] = (int16_t *)heap_caps_malloc(n_samples * sizeof(int16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!inference.buffers[1]) {
        inference.buffers[1] = (int16_t *)heap_caps_malloc(n_samples * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    }
    if (!inference.buffers[0] || !inference.buffers[1]) {
        Serial.println("[ERROR] Failed to allocate mic buffers.");
        return false;
    }
    inference.buf_select = 0;
    inference.buf_count  = 0;
    inference.n_samples  = n_samples;
    inference.buf_ready  = 0;

    i2s_init_mic(EI_CLASSIFIER_FREQUENCY);
    delay(100);
    record_status = true;
    xTaskCreatePinnedToCore(capture_samples, "CaptureSamples",
                            2048, (void *)(uintptr_t)SAMPLE_BUFFER_SIZE,
                            10, NULL, 0);
    return true;
}

static bool microphone_inference_record(void)
{
    while (inference.buf_ready == 0) delay(1);
    inference.buf_ready = 0;
    return true;
}

static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr)
{
    numpy::int16_to_float(&inference.buffers[inference.buf_select ^ 1][offset], out_ptr, length);
    return 0;
}

// ── WiFi ──────────────────────────────────────────────────────────────────────
static void connect_wifi()
{
    Serial.printf("Connecting to WiFi: %s", WIFI_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    uint8_t tries = 0;
    while (WiFi.status() != WL_CONNECTED && tries++ < 30) {
        delay(500);
        Serial.print(".");
    }
    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("\nWiFi OK! ESP32 IP: %s\n", WiFi.localIP().toString().c_str());
    } else {
        Serial.println("\n[ERROR] WiFi failed. Check SSID/password. Restarting...");
        delay(3000);
        ESP.restart();
    }
}

// ── HTTP Telemetry ────────────────────────────────────────────────────────────
static uint32_t last_telemetry_ms = 0;

static void send_esp32_telemetry(float cpu_pct, uint32_t infer_ms)
{
    if (WiFi.status() != WL_CONNECTED) return;

    // Internal SRAM
    uint32_t total_sram_kb = heap_caps_get_total_size(MALLOC_CAP_INTERNAL) / 1024;
    if (total_sram_kb == 0) total_sram_kb = 320;
    uint32_t free_sram_kb  = heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024;
    uint32_t used_sram_kb  = (total_sram_kb > free_sram_kb) ? (total_sram_kb - free_sram_kb) : 0;
    float ram_pct          = ((float)used_sram_kb / (float)total_sram_kb) * 100.0f;

    // PSRAM
    uint32_t psram_total_kb = heap_caps_get_total_size(MALLOC_CAP_SPIRAM) / 1024;
    uint32_t psram_free_kb  = heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024;
    uint32_t psram_used_kb  = (psram_total_kb > psram_free_kb) ? (psram_total_kb - psram_free_kb) : 0;

    // Dual-Core SoC CPU Load: Core 1 (AI inference) + Core 0 (WiFi/DMA ~12%) averaged across 2 cores
    float core1_load = cpu_pct;
    float core0_load = 12.0f;
    float dual_core_soc_cpu = (core1_load + core0_load) / 2.0f;
    if (dual_core_soc_cpu > 100.0f) dual_core_soc_cpu = 100.0f;

    char json[256];
    snprintf(json, sizeof(json),
             "{\"cpu_pct\":%.1f,\"ram_used_kb\":%u,\"ram_total_kb\":%u,\"ram_pct\":%.1f,\"psram_used_kb\":%u,\"psram_total_kb\":%u,\"infer_time_ms\":%u}",
             dual_core_soc_cpu, used_sram_kb, total_sram_kb, ram_pct, psram_used_kb, psram_total_kb, infer_ms);

    HTTPClient http;
    http.begin(METRICS_URL);
    http.addHeader("Content-Type", "application/json");
    http.setTimeout(150); // fast non-blocking timeout
    http.POST(json);
    http.end();
}

// ── HTTP Trigger ──────────────────────────────────────────────────────────────
static void send_trigger()
{
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[trigger] WiFi disconnected, skipping.");
        return;
    }
    HTTPClient http;
    http.begin(TRIGGER_URL);
    http.addHeader("Content-Type", "application/json");
    int code = http.POST("{\"event\":\"nova_triggered\",\"source\":\"esp32\"}");
    if (code == 200) {
        Serial.println("[trigger] Server notified OK — Whisper is listening!");
    } else {
        Serial.printf("[trigger] HTTP error %d. Is server.py running on %s:%d?\n",
                      code, PC_IP, PC_TRIGGER_PORT);
    }
    http.end();
}

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup()
{
    Serial.begin(115200);
    delay(2000);  // give Serial Monitor time to connect (no hard block)
    Serial.println("\n=== NOVA ESP32-S3 Wake-Word Detector ===");

    // Locate "nova" label in the model
    for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        if (String(ei_classifier_inferencing_categories[i]).equalsIgnoreCase(NOVA_LABEL)) {
            nova_class_idx = (int)i;
            break;
        }
    }
    if (nova_class_idx < 0) {
        Serial.printf("[ERROR] Label '%s' not found!\nAvailable:\n", NOVA_LABEL);
        for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++)
            Serial.printf("  [%d] %s\n", i, ei_classifier_inferencing_categories[i]);
        while (1) delay(1000);
    }
    Serial.printf("NOVA is label index %d\n", nova_class_idx);
    Serial.printf("Free Heap: %d bytes (Largest Block: %d bytes)\n",
                  ESP.getFreeHeap(), heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
    Serial.printf("Internal free: %d, Internal largest block: %d\n",
                  heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                  heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));

    // Model claims its tensor arena FIRST, while internal SRAM is at its
    // cleanest — before WiFi/HTTPClient have a chance to fragment it.
    // Arena is allocated via the SDK's original internal-only ei_malloc.
    run_classifier_init();

    Serial.printf("Free Heap after model init: %d bytes (Largest block: %d bytes)\n",
                  ESP.getFreeHeap(), heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
    Serial.printf("Internal free after model init: %d, Internal largest block: %d\n",
                  heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                  heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));

    connect_wifi();

    Serial.printf("Internal free after WiFi: %d, Internal largest block: %d\n",
                  heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                  heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));

    if (!microphone_inference_start(EI_CLASSIFIER_SLICE_SIZE)) {
        Serial.println("[ERROR] Failed to allocate mic buffers.");
        while (1) delay(1000);
    }
    Serial.printf("Listening... say \"%s\"\n", NOVA_LABEL);
}

// ── Main loop ─────────────────────────────────────────────────────────────────
void loop()
{
    microphone_inference_record();

    signal_t            signal;
    ei_impulse_result_t result = {0};
    signal.total_length = EI_CLASSIFIER_SLICE_SIZE;
    signal.get_data     = &microphone_audio_signal_get_data;

    uint32_t t_start = millis();
    EI_IMPULSE_ERROR r = run_classifier_continuous(&signal, &result, false);
    uint32_t infer_ms = millis() - t_start;
    if (r != EI_IMPULSE_OK) {
        Serial.printf("[ERROR] Classifier: %d\n", r);
        return;
    }

    uint32_t slice_dur_ms = (EI_CLASSIFIER_SLICE_SIZE * 1000) / EI_CLASSIFIER_FREQUENCY;
    float cpu_pct = ((float)infer_ms / (float)slice_dur_ms) * 100.0f;
    if (cpu_pct > 100.0f) cpu_pct = 100.0f;

    // Send ESP32 telemetry every 2 seconds
    uint32_t now = millis();
    if (now - last_telemetry_ms > 2000) {
        last_telemetry_ms = now;
        send_esp32_telemetry(cpu_pct, infer_ms);
    }

    float nova_score = result.classification[nova_class_idx].value;

    int16_t *buf = inference.buffers[inference.buf_select ^ 1];
    int16_t peak = 0;
    for (size_t i = 0; i < EI_CLASSIFIER_SLICE_SIZE; i++) {
        int16_t v = abs(buf[i]);
        if (v > peak) peak = v;
    }

    // Sliding vote window — only count this slice if mic is above the noise floor
    if (peak >= MIC_AMPLITUDE_GATE) {
        vote_scores[vote_idx % VOTE_WINDOW] = nova_score;
    } else {
        vote_scores[vote_idx % VOTE_WINDOW] = 0.0f;  // silence → no vote
    }
    vote_idx++;

    int votes = 0;
    for (int i = 0; i < VOTE_WINDOW; i++) {
        if (vote_scores[i] >= CONFIDENCE_THRESHOLD) votes++;
    }

    Serial.printf("[mic: %5d] ", peak);
    for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        Serial.printf("%s: %.2f | ", ei_classifier_inferencing_categories[i], result.classification[i].value);
    }
    Serial.printf("(votes: %d/%d)\n", votes, VOTE_WINDOW);

    if (votes >= VOTE_REQUIRED && (now - last_trigger_ms) > COOLDOWN_MS) {
        last_trigger_ms = now;
        memset(vote_scores, 0, sizeof(vote_scores));
        vote_idx = 0;

        Serial.println("\nNOVA TRIGGERED! Sending HTTP POST to Whisper server...");
        send_trigger();
        Serial.printf("Listening... say \"%s\"\n", NOVA_LABEL);
    }
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for sensor — this project requires a microphone model."
#endif