#pragma once

#include <cstddef>

namespace dataset {

constexpr int NUM_SAMPLES = 4935;
constexpr int NUM_FEATURES = 4;

// Embedded float array stored in flash (RODATA section).
// Layout: inputs[NUM_SAMPLES][NUM_FEATURES] (row-major) followed by targets[NUM_SAMPLES].
extern const float blob[];
extern const size_t blob_size;

inline float get_input(int sample, int feature) {
    return blob[sample * NUM_FEATURES + feature];
}

inline float get_target(int sample) {
    return blob[NUM_SAMPLES * NUM_FEATURES + sample];
}

}  // namespace dataset
