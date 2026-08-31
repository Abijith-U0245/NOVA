// main.cpp — live mic capture -> Edge Impulse KWS classifier -> NOVA vote trigger
// Build inside the merged example-standalone-inferencing repo
// Requires: libasound2-dev (ALSA) — already linked via Makefile LDFLAGS += -lasound

#include <alsa/asoundlib.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <deque>

#include "edge-impulse-sdk/classifier/ei_run_classifier.h"

#define SAMPLE_RATE 16000
#define WINDOW_SAMPLES EI_CLASSIFIER_RAW_SAMPLE_COUNT   // 16000 = 1.0s window
#define STEP_SAMPLES (SAMPLE_RATE / 4)                  // 0.25s hop -> 75% overlap

#define NOVA_LABEL "nova"
#define CONFIDENCE_THRESHOLD 0.55f
#define VOTE_WINDOW 3
#define VOTE_REQUIRED 1
#define COOLDOWN_HOPS 4   // ~1s cooldown after a trigger (4 hops at 0.25s each)

// rolling raw-sample ring buffer (float; DSP block expects raw int16-range values)
static std::vector<float> ring(WINDOW_SAMPLES, 0.0f);

// EI callback: copies from our ring buffer into the classifier's internal working buffer
static int get_signal_data(size_t offset, size_t length, float *out_ptr) {
    memcpy(out_ptr, ring.data() + offset, length * sizeof(float));
    return 0;
}

int find_label_index(const char *name) {
    for (int i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        if (strcmp(ei_classifier_inferencing_categories[i], name) == 0) return i;
    }
    return -1;
}

int main() {
    int nova_idx = find_label_index(NOVA_LABEL);
    if (nova_idx < 0) {
        printf("ERROR: could not find label '%s' in model categories\n", NOVA_LABEL);
        return 1;
    }
    printf("NOVA is class index %d\n", nova_idx);

    // ---- ALSA mic setup ----
    snd_pcm_t *capture_handle;
    snd_pcm_hw_params_t *hw_params;
    int err;

    if ((err = snd_pcm_open(&capture_handle, "default", SND_PCM_STREAM_CAPTURE, 0)) < 0) {
        printf("ERROR: cannot open audio device (%s)\n", snd_strerror(err));
        return 1;
    }
    snd_pcm_hw_params_alloca(&hw_params);
    snd_pcm_hw_params_any(capture_handle, hw_params);
    snd_pcm_hw_params_set_access(capture_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED);
    snd_pcm_hw_params_set_format(capture_handle, hw_params, SND_PCM_FORMAT_S16_LE);
    snd_pcm_hw_params_set_channels(capture_handle, hw_params, 1);
    unsigned int rate = SAMPLE_RATE;
    snd_pcm_hw_params_set_rate_near(capture_handle, hw_params, &rate, 0);
    snd_pcm_hw_params(capture_handle, hw_params);
    snd_pcm_prepare(capture_handle);

    printf("Listening... say \"NOVA\"\n");

    std::vector<int16_t> chunk(STEP_SAMPLES);
    std::deque<bool> votes;
    int cooldown_frames = 0;

    while (true) {
        // read one hop of new audio (0.25s)
        int frames_read = snd_pcm_readi(capture_handle, chunk.data(), STEP_SAMPLES);
        if (frames_read < 0) {
            snd_pcm_prepare(capture_handle); // recover from overrun
            continue;
        }

        // slide ring buffer left by STEP_SAMPLES, append new hop at the end
        memmove(ring.data(), ring.data() + STEP_SAMPLES, (WINDOW_SAMPLES - STEP_SAMPLES) * sizeof(float));
        for (int i = 0; i < STEP_SAMPLES; i++) {
            ring[WINDOW_SAMPLES - STEP_SAMPLES + i] = (float)chunk[i];
        }

        // ---- run classifier ----
        signal_t signal;
        signal.total_length = WINDOW_SAMPLES;
        signal.get_data = &get_signal_data;

        ei_impulse_result_t result = { 0 };
        EI_IMPULSE_ERROR res = run_classifier(&signal, &result, false);
        if (res != EI_IMPULSE_OK) {
            printf("run_classifier failed (%d)\n", res);
            continue;
        }

        float nova_score = result.classification[nova_idx].value;

        // ---- vote logic ----
        votes.push_back(nova_score > CONFIDENCE_THRESHOLD);
        if ((int)votes.size() > VOTE_WINDOW) votes.pop_front();
        int vote_count = 0;
        for (bool v : votes) if (v) vote_count++;

        printf("NOVA: %.2f   votes: %d/%d\n", nova_score, vote_count, (int)votes.size());

        if (cooldown_frames > 0) {
            cooldown_frames--;
        } else if (vote_count >= VOTE_REQUIRED) {
            printf(">>> NOVA CONFIRMED — launching VOSK stream <<<\n");
            votes.clear();
            cooldown_frames = COOLDOWN_HOPS;
            // Fire-and-forget launch of the Python VOSK streaming client.
            // '&' backgrounds it so this loop keeps listening for the next "NOVA"
            // immediately, rather than blocking on the ASR call.
            system("python3 stream_to_vosk.py &");
        }
    }

    snd_pcm_close(capture_handle);
    return 0;
}
