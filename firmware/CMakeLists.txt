cmake_minimum_required(VERSION 3.16)
project(EdgeSeisFusion_Firmware C CXX ASM)

# Set language standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Toolchain: ARM embedded
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

# Compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard")

# Source files
set(SOURCES
    main.cpp
    acquisition.cpp
    dsp_embedded.cpp
    inference.cpp
    anomaly_state_machine.cpp
    telemetry.cpp
)

# Model artifacts (TFLM exported)
list(APPEND SOURCES models/model.cc)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/config
    ${CMAKE_CURRENT_SOURCE_DIR}/models
    # Sony Spresense SDK paths (set in CMAKE_TOOLCHAIN_FILE or environment)
)

# TFLM library (TensorFlow Lite for Microcontrollers)
# Assuming TFLM prebuilt or included in Spresense SDK
find_package(TFLM REQUIRED)

# CMSIS-DSP library
find_package(CMSIS-DSP REQUIRED)

# Create executable
add_executable(firmware.elf ${SOURCES})

# Link libraries
target_link_libraries(firmware.elf
    PUBLIC TFLM::tflm
    PUBLIC CMSIS-DSP::cmsis_dsp
    m  # Math library
)

# Optional: UART/Serial library for debugging
find_package(Serial QUIET)
if(Serial_FOUND)
    target_link_libraries(firmware.elf PUBLIC Serial::Serial)
endif()

# Generate binary artifact
add_custom_command(TARGET firmware.elf POST_BUILD
    COMMAND arm-none-eabi-objcopy -O binary firmware.elf firmware.bin
    COMMENT "Generating binary artifact: firmware.bin"
)

# Optional: Generate hex artifact for some programmers
add_custom_command(TARGET firmware.elf POST_BUILD
    COMMAND arm-none-eabi-objcopy -O ihex firmware.elf firmware.hex
    COMMENT "Generating hex artifact: firmware.hex"
)

# Optional: Print size analysis
add_custom_command(TARGET firmware.elf POST_BUILD
    COMMAND arm-none-eabi-size firmware.elf
    COMMENT "Firmware size analysis"
)

# Set output directory
set_target_properties(firmware.elf PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
