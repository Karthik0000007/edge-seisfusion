/**
 * TFLM Model Arena Configuration
 * Auto-generated from model export (Phase 9)
 *
 * Tensor arena sizing for INT8 quantized 1D-CAE model
 */

#ifndef ARENA_CONFIG_H_
#define ARENA_CONFIG_H_

#include <cstdint>

/* Model metadata (from export) */
constexpr uint32_t kModelSize = 65536;           /* Model binary size: ~64 KB */
constexpr uint32_t kTensorArenaSize = 614400;    /* Tensor arena: 600 KB */
constexpr uint32_t kInputTensorSize = 140;       /* Input: [1024, 35] floats = 35 KB (quantized) */
constexpr uint32_t kOutputTensorSize = 140;      /* Output: [1024, 35] floats = 35 KB (quantized) */

/* Alignment requirements (TFLM) */
constexpr uint32_t kTensorArenaAlignment = 16;   /* 16-byte alignment for SIMD */

/* Supported operators (verified at export time) */
constexpr const char* kSupportedOperators[] = {
    "Conv1D",
    "MaxPooling1D",
    "UpSampling1D",
    "Reshape",
    "Quantize",
    "Dequantize",
};

#endif /* ARENA_CONFIG_H_ */
