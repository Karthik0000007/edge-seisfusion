/**
 * EdgeSeisFusion Firmware Parameters
 * MCU-specific constants for Spresense deployment
 * Generated from config/firmware_params.h
 */

#ifndef FIRMWARE_PARAMS_H_
#define FIRMWARE_PARAMS_H_

#ifdef __cplusplus
extern "C" {
#endif

/* Memory Configuration (SRAM: 1.5 MB total) */
#define TENSOR_ARENA_SIZE_BYTES (614400U)      /* 600 KB tensor arena */
#define DSP_WORKSPACE_SIZE_BYTES (102400U)     /* 100 KB DSP workspace */
#define RING_BUFFER_SIZE_BYTES (204800U)       /* 200 KB ring buffers */
#define INFERENCE_SCRATCH_SIZE_BYTES (307200U) /* 300 KB inference scratch */
#define STATE_HEAP_SIZE_BYTES (204800U)        /* 200 KB state + allocator */

/* Sampling Configuration */
#define SAMPLING_RATE_HZ (1000U)               /* 1 kHz */
#define WINDOW_SIZE_SAMPLES (1024U)            /* 1 second per window */
#define NUM_CHANNELS (5U)                      /* Accel-X/Y/Z, Strain, Acoustic */

/* Anomaly Detection Thresholds */
#define ANOMALY_UP_THRESHOLD (2.5f)            /* Score > 2.5 → anomaly */
#define ANOMALY_DOWN_THRESHOLD (1.5f)          /* Score < 1.5 → healthy (hysteresis) */
#define COOLDOWN_SECONDS (300U)                /* 300 sec between consecutive alerts */
#define CONFIDENCE_THRESHOLD (0.3f)            /* Min confidence for valid decision */

/* Timing Budgets (milliseconds) */
#define DMA_CAPTURE_TIME_MS (1024U)            /* Async, doesn't block */
#define DSP_PROCESSING_TIME_MS (100U)          /* Feature extraction + Kalman */
#define INFERENCE_TIME_MS (300U)               /* INT8 inference on Cortex-M4F */
#define ANOMALY_SCORING_TIME_MS (10U)          /* State machine decision */
#define PAYLOAD_ENCODING_TIME_MS (2U)          /* Telemetry encoding */
#define RADIO_TX_TIME_MS (100U)                /* LoRaWAN transmission burst */
#define TOTAL_ACTIVE_TIME_BUDGET_MS (400U)     /* Total active per window */

/* Power Management */
#define ACTIVE_POWER_MW (45U)                  /* Active: 45 mW */
#define SLEEP_POWER_MW (0U)                    /* Sleep: <1 mW */
#define TARGET_DUTY_CYCLE_PERCENT (1U)         /* <1% active duty */
#define WATCHDOG_TIMEOUT_SEC (30U)             /* Watchdog timer: 30 seconds */
#define LOW_BATTERY_THRESHOLD_PERCENT (20U)    /* Alert at 20% battery */

/* Device Parameters */
#define DEVICE_ID_HEX (0x26041234UL)           /* Unique device identifier */
#define FIRMWARE_VERSION "1.0.0"
#define BOARD_TYPE "SPRESENSE"

/* Sensor Configuration */
#define ACCELEROMETER_RANGE_MG (2000U)         /* ±2000 mg full-scale */
#define ACCELEROMETER_RESOLUTION_BITS (16U)    /* 16-bit ADC */
#define STRAIN_GAUGE_RANGE_MV (5000U)          /* 0–5000 mV */
#define ACOUSTIC_RANGE_DB (120U)               /* 0–120 dB */

/* Calibration */
#define TEMPERATURE_SENSOR_ENABLED (1)
#define BATTERY_ADC_ENABLED (1)
#define ACCEL_CALIBRATION_SAMPLES (100U)
#define CALIBRATION_CHECK_INTERVAL_SEC (3600U)

/* Telemetry */
#define MAX_PAYLOAD_BYTES (20U)                /* Compact alert payload */
#define UPLINK_RETRY_MAX (3U)                  /* Max retries on TX failure */
#define UPLINK_RETRY_BACKOFF_MS (100U)         /* Exponential backoff base */
#define UPLINK_QUEUE_SIZE_ALERTS (10U)         /* Queue up to 10 unsent alerts */

/* Debug Configuration */
#define DEBUG_MODE_ENABLED (0)                 /* Set 1 for verbose logging */
#define PROFILING_ENABLED (0)                  /* Measure per-stage latency */
#define LOG_TO_FLASH_ENABLED (0)               /* Save logs to EEPROM */

#ifdef __cplusplus
}
#endif

#endif /* FIRMWARE_PARAMS_H_ */
