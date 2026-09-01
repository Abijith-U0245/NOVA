/*
 * nova_trigger_esp32.ino
 *
 * NOVA wake-word detection on ESP32-S3 with EdgeImpulse inferencing library.
 * On detection -> sends HTTP POST to PC's server.py /trigger endpoint.
 * The PC's stream_to_vosk.py then captures from the PC mic and pipes to Whisper.
 *
 * ────────────────────────────────────────────────────────────────────────────
 *  WIRING  (INMP441 I2S mic — default pins below)
 *  ESP32-S3 GPIO   ->   INMP441 Pin
 *  GPIO  4         ->   SCK  (BCLK)
 *  GPIO  5         ->   WS   (LRCK)
 *  GPIO  6         ->   SD   (data out)
 *  3.3V            ->   VDD
 *  GND             ->   GND & L/R
 * ────────────────────────────────────────────────────────────────────────────
 *
 *  Requirements (Arduino IDE -> Sketch -> Include Library -> Add .ZIP Library):
 *    - nova_esp_inferencing  (the library folder already in your project)
 *    - ESP32 Arduino core >= 2.0.4 (provides WiFi.h, HTTPClient.h, driver/i2s.h)
 */

// ── User Config ──────────────────────────────────────────────────────────────
#define WIFI_SSID        "Abirami"            // 2.4GHz network — ESP32 does NOT support 5GHz!
#define WIFI_PASSWORD    "37@ab102"           // your WiFi password
#define PC_IP            "192.168.29.196"     // your PC's local IP
#define PC_TRIGGER_PORT  8766
#define TRIGGER_URL      "http://192.168.29.196:8766/trigger"

// I2S microphone pins — adjust if your wiring differs
#define I2S_PORT         I2S_NUM_0
#define I2S_BCK_PIN      4   // SCK / BCLK
#define I2S_WS_PIN       5   // WS  / LRCLK  (NOTE: must be different from BCK, kept as 5)
#define I2S_DATA_PIN     6   // SD  / data in

// Wake-word tuning
#define NOVA_LABEL           "nova"
#define CONFIDENCE_THRESHOLD 0.55f
#define VOTE_WINDOW          3
#define VOTE_REQUIRED        1
#define COOLDOWN_MS          1500     // min ms between triggers
// ─────────────────────────────────────────────────────────────────────────────

#define EIDSP_QUANTIZE_FILTERBANK 0

#include <abijith-u0245-project-1_inferencing.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2s.h"
#include <WiFi.h>
#include <HTTPClient.h>

// ── Audio buffer ─────────────────────────────────────────────────────────────
typedef struct {
    int16_t *buffers[2];
    uint8_t  buf_select;
    uint8_t  buf_ready;
    uint32_t buf_count;
    uint32_t n_samples;
} inference_t;

static inference_t inference;
static const uint32_t SAMPLE_BUFFER_SIZE = 2048;
static int16_t sampleBuffer[SAMPLE_BUFFER_SIZE];
static bool    record_status = true;
static int     print_results = -(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW);

// ── Vote window ───────────────────────────────────────────────────────────────
static float    vote_scores[VOTE_WINDOW] = {0};
static int      vote_idx                 = 0;
static int      nova_class_idx           = -1;
static uint32_t last_trigger_ms          = 0;

// ── I2S init ──────────────────────────────────────────────────────────────────
static void i2s_init_mic(uint32_t sample_rate)
{
    i2s_config_t cfg = {
        .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate          = sample_rate,
        .bits_per_sample      = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count        = 8,
        .dma_buf_len          = 512,
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
static void capture_samples(void *arg)
{
    const size_t bytes_to_read = (size_t)(uintptr_t)arg;
    size_t       bytes_read    = bytes_to_read;

    while (record_status) {
        i2s_read(I2S_PORT, sampleBuffer, bytes_to_read, &bytes_read, portMAX_DELAY);
        if (bytes_read > 0) {
            // Amplify quiet INMP441 output
            for (size_t x = 0; x < bytes_read / 2; x++) {
                sampleBuffer[x] = (int16_t)((int32_t)sampleBuffer[x] * 8);
            }
            if (record_status) audio_inference_callback(bytes_to_read);
        }
    }
    vTaskDelete(NULL);
}

// ── Mic inference helpers ─────────────────────────────────────────────────────
static bool microphone_inference_start(uint32_t n_samples)
{
    inference.buffers[0] = (int16_t *)malloc(n_samples * sizeof(int16_t));
    if (!inference.buffers[0]) return false;
    inference.buffers[1] = (int16_t *)malloc(n_samples * sizeof(int16_t));
    if (!inference.buffers[1]) { free(inference.buffers[0]); return false; }

    inference.buf_select = 0;
    inference.buf_count  = 0;
    inference.n_samples  = n_samples;
    inference.buf_ready  = 0;

    i2s_init_mic(EI_CLASSIFIER_FREQUENCY);
    delay(100);
    record_status = true;
    xTaskCreatePinnedToCore(capture_samples, "CaptureSamples",
                            1024 * 32, (void *)(uintptr_t)SAMPLE_BUFFER_SIZE,
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

    connect_wifi();
    run_classifier_init();

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

    EI_IMPULSE_ERROR r = run_classifier_continuous(&signal, &result, false);
    if (r != EI_IMPULSE_OK) {
        Serial.printf("[ERROR] Classifier: %d\n", r);
        return;
    }

    float nova_score = result.classification[nova_class_idx].value;

    // Sliding vote window
    vote_scores[vote_idx % VOTE_WINDOW] = nova_score;
    vote_idx++;

    int votes = 0;
    for (int i = 0; i < VOTE_WINDOW; i++) {
        if (vote_scores[i] >= CONFIDENCE_THRESHOLD) votes++;
    }

    Serial.printf("NOVA: %.2f   votes: %d/%d\n", nova_score, votes, VOTE_WINDOW);

    uint32_t now = millis();
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
