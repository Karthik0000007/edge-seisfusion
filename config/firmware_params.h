# Firmware Configuration Parameters
# MCU-specific constants for Spresense deployment
# Phase 10+: Embedded system tuning

[MEMORY]
TENSOR_ARENA_SIZE_BYTES = 614400        # 600 KB tensor arena (fits in SRAM)
DSP_WORKSPACE_SIZE_BYTES = 102400       # 100 KB DSP workspace
RING_BUFFER_SIZE_BYTES = 204800         # 200 KB ring buffers (5 channels × 1024 samples × 8 bytes × 5)
INFERENCE_SCRATCH_SIZE_BYTES = 307200   # 300 KB inference scratch space
STATE_HEAP_SIZE_BYTES = 204800          # 200 KB for state + allocator

[SAMPLING]
SAMPLING_RATE_HZ = 1000                 # 1 kHz deterministic sampling
WINDOW_SIZE_SAMPLES = 1024              # 1 second per window
NUM_CHANNELS = 5                        # Accel-X/Y/Z, Strain, Acoustic

[ANOMALY_DETECTION]
ANOMALY_UP_THRESHOLD = 2.5              # Score > 2.5 → anomaly
ANOMALY_DOWN_THRESHOLD = 1.5            # Score < 1.5 → healthy (hysteresis)
COOLDOWN_SECONDS = 300                  # 300 sec between consecutive alerts
CONFIDENCE_THRESHOLD = 0.3              # Min confidence for valid decision

[TIMING_BUDGETS_MS]
DMA_CAPTURE_TIME_MS = 1024              # Async, doesn't block main loop
DSP_PROCESSING_TIME_MS = 100            # Feature extraction + Kalman
INFERENCE_TIME_MS = 300                 # INT8 inference on Cortex-M4F
ANOMALY_SCORING_TIME_MS = 10            # State machine decision
PAYLOAD_ENCODING_TIME_MS = 2            # Telemetry encoding
RADIO_TX_TIME_MS = 100                  # LoRaWAN transmission burst
TOTAL_ACTIVE_TIME_BUDGET_MS = 400       # Total active per window (leaves 600 ms for sleep)

[POWER_MANAGEMENT]
ACTIVE_POWER_MW = 45                    # Active: 45 mW (10 mA @ 4.5V)
SLEEP_POWER_MW = 0.5                    # Sleep: 0.5 mW (100 μA @ 4.5V)
TARGET_DUTY_CYCLE_PERCENT = 1.0         # <1% active duty
WATCHDOG_TIMEOUT_SEC = 30               # Watchdog timer: 30 seconds
LOW_BATTERY_THRESHOLD_PERCENT = 20      # Alert at 20% battery capacity

[DEVICE_PARAMETERS]
DEVICE_ID_HEX = "0x26041234"            # Unique device identifier (6-byte)
FIRMWARE_VERSION = "1.0.0"              # Semantic versioning
BOARD_TYPE = "SPRESENSE"                # Sony Spresense ARM Cortex-M4F @ 156 MHz

[SENSOR_CONFIG]
ACCELEROMETER_RANGE_MG = 2000           # ±2000 mg full-scale
ACCELEROMETER_RESOLUTION_BITS = 16      # 16-bit ADC
STRAIN_GAUGE_RANGE_MV = 5000            # 0–5000 mV
ACOUSTIC_RANGE_DB = 120                 # 0–120 dB

[CALIBRATION]
TEMPERATURE_SENSOR_ENABLED = true       # Monitor on-device temperature
BATTERY_ADC_ENABLED = true              # Monitor battery voltage
ACCEL_CALIBRATION_SAMPLES = 100         # Calibration window
CALIBRATION_CHECK_INTERVAL_SEC = 3600   # Check calibration every hour

[TELEMETRY]
MAX_PAYLOAD_BYTES = 20                  # Compact alert payload
UPLINK_RETRY_MAX = 3                    # Max retries on TX failure
UPLINK_RETRY_BACKOFF_MS = 100           # Exponential backoff base: 100 ms
UPLINK_QUEUE_SIZE_ALERTS = 10           # Queue up to 10 unsent alerts in flash

[DEBUG]
DEBUG_MODE_ENABLED = false              # Set true for verbose logging
PROFILING_ENABLED = false               # Measure per-stage latency
LOG_TO_FLASH_ENABLED = false            # Save logs to EEPROM
