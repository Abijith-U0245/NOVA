/*
 * ============================================================================
 * INMP441 I2S Microphone Hardware Test Sketch
 * ============================================================================
 *
 * Wiring for ESP32-S3:
 *   ESP32-S3 Pin   ->   INMP441 Pin
 *   --------------------------------
 *   GPIO 4         ->   SCK (BCLK)
 *   GPIO 5         ->   WS  (LRCK / Word Select)
 *   GPIO 6         ->   SD  (Serial Data / DOUT)
 *   3.3V           ->   VDD
 *   GND            ->   GND
 *   GND            ->   L/R (Ties L/R to Left channel)
 *
 * How to test:
 *   1. Upload this sketch.
 *   2. Open Serial Monitor (115200 baud) to see the live ASCII VU meter & peak values.
 *   3. OR open Tools -> "Serial Plotter" in Arduino IDE to see the real-time audio waveform.
 *   4. Clap, tap near the mic, or speak into it to see the values react.
 * ============================================================================
 */

#include <Arduino.h>
#include "driver/i2s.h"

// ── Pin Configuration ────────────────────────────────────────────────────────
#define I2S_PORT         I2S_NUM_0
#define I2S_BCK_PIN      15  // SCK / BCLK
#define I2S_WS_PIN       16  // WS  / LRCK
#define I2S_DATA_PIN     17  // SD  / DOUT

#define SAMPLE_RATE      16000
#define BUFFER_SAMPLES   256

static int32_t raw_samples[BUFFER_SAMPLES];

void setup() {
    Serial.begin(115200);
    delay(1500);

    Serial.println("\n==============================================");
    Serial.println("     INMP441 I2S Microphone Test Utility      ");
    Serial.println("==============================================");
    Serial.printf("Pins: SCK=%d, WS=%d, SD=%d\n", I2S_BCK_PIN, I2S_WS_PIN, I2S_DATA_PIN);
    Serial.println("Make sure INMP441 L/R pin is connected to GND!");
    Serial.println("----------------------------------------------\n");

    // Standard I2S configuration reading stereo to inspect Left vs Right
    i2s_config_t i2s_config = {
        .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate          = SAMPLE_RATE,
        .bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format       = I2S_CHANNEL_FMT_RIGHT_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count        = 4,
        .dma_buf_len          = BUFFER_SAMPLES,
        .use_apll             = false,
        .tx_desc_auto_clear   = false,
        .fixed_mclk           = -1,
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num   = I2S_BCK_PIN,
        .ws_io_num    = I2S_WS_PIN,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num  = I2S_DATA_PIN,
    };

    esp_err_t err = i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    if (err != ESP_OK) {
        Serial.printf("[ERROR] Failed to install I2S driver: %d\n", err);
        while (1) delay(1000);
    }

    err = i2s_set_pin(I2S_PORT, &pin_config);
    if (err != ESP_OK) {
        Serial.printf("[ERROR] Failed to set I2S pins: %d\n", err);
        while (1) delay(1000);
    }

    // Enable pull-down on SD pin to prevent floating noise if mic is not driving the bus
    gpio_pulldown_en((gpio_num_t)I2S_DATA_PIN);
    gpio_pullup_dis((gpio_num_t)I2S_DATA_PIN);

    i2s_zero_dma_buffer(I2S_PORT);
    Serial.println("I2S Stereo Driver Initialized. Comparing Left vs Right channel...\n");
}

void loop() {
    size_t bytes_read = 0;
    esp_err_t res = i2s_read(I2S_PORT, raw_samples, sizeof(raw_samples), &bytes_read, portMAX_DELAY);

    if (res != ESP_OK || bytes_read == 0) {
        Serial.println("[WARN] i2s_read failed or 0 bytes read");
        delay(100);
        return;
    }

    size_t total_samples = bytes_read / sizeof(int32_t);
    int16_t max_left = 0, min_left = 0;
    int16_t max_right = 0, min_right = 0;

    for (size_t i = 0; i < total_samples; i += 2) {
        // Left channel is index i, Right channel is index i+1
        int16_t left_s  = (int16_t)(raw_samples[i] >> 14);
        int16_t right_s = (int16_t)(raw_samples[i+1] >> 14);

        if (left_s < min_left) min_left = left_s;
        if (left_s > max_left) max_left = left_s;

        if (right_s < min_right) min_right = right_s;
        if (right_s > max_right) max_right = right_s;
    }

    int32_t p2p_left  = max_left - min_left;
    int32_t p2p_right = max_right - min_right;

    Serial.printf("LEFT P2P: %5d (val: %6d) | RIGHT P2P: %5d (val: %6d)\n",
                  p2p_left, (int16_t)(raw_samples[0] >> 14),
                  p2p_right, (int16_t)(raw_samples[1] >> 14));

    delay(60);
}
