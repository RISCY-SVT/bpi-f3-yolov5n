#!/usr/bin/env bash
# ======================================================================
# env.sh - Environment configuration for BPI-F3 YOLOv5n project
# Updated for SpaceMIT toolchain and CSI-NN2 C920V2 configuration
# ======================================================================



# Model configuration
export CPU_MODEL="c920v2"
export ONNX_MODEL="yolov5n_out3.onnx"
export MODEL_INPUT="images"
export MODEL_OUTPUT="/model.24/m.0/Conv_output_0;/model.24/m.1/Conv_output_0;/model.24/m.2/Conv_output_0"
export MODEL_INPUT_SHAPE="1 3 384 640"  # CRITICAL: 384x640, not 640x640!
export CALIBRATION_DIR="./calibration_images"
export QUANTIZATION_SCHEME="float16"
export OUTPUT_DIR="cpu_model"

# Toolchain configuration - USE SPACEMIT ONLY!
export TOOLS_PREFIX="riscv64-unknown-linux-gnu"
export RISCV="/opt/spacemit"                    # SpaceMIT toolchain path
export CROSS_PREFIX="${RISCV}/bin/${TOOLS_PREFIX}"
export SYSROOT="${RISCV}/sysroot"               # Target rootfs with GLIBC 2.39
export PREFIX="${SYSROOT}/usr"                  # Where libs/headers are

# CSI-NN2 installation path
export INSTALL_NN2_PREFIX="/opt/csi-nn2/install_nn2"

# ISA configuration for SpaceMIT K1X (RVV 1.0)
export MARCH_C920V2="rv64gcv_zvfh"              # RVV 1.0 with FP16 support
export ISA_FLAGS="-march=${MARCH_C920V2} -mabi=lp64d"

# Optimization flags
export OMP_NUM_THREADS=4                        # Use cores 0-3 for NN
COMMON_CFLAGS="${ISA_FLAGS} -O3 -ffast-math -pipe -ftree-vectorize -funroll-loops"
export CFLAGS="${COMMON_CFLAGS} -ffunction-sections -fdata-sections"
export CXXFLAGS="${COMMON_CFLAGS} -std=c++17"
export LDFLAGS="${ISA_FLAGS} -Wl,--gc-sections"

# Include paths for CSI-NN2
export RISCV_INCLUDES="-I${INSTALL_NN2_PREFIX}/${CPU_MODEL}/include \
-I${INSTALL_NN2_PREFIX}/${CPU_MODEL}/include/shl_public \
-I${INSTALL_NN2_PREFIX}/${CPU_MODEL}/include/csinn \
-I${OUTPUT_DIR}"

# Library paths and libraries
# export RISCV_LIBS="-L${INSTALL_NN2_PREFIX}/${CPU_MODEL}/lib -lshl \
# -L${SYSROOT}/usr/lib/riscv64-linux-gnu \
# -lopencv_core -lopencv_imgproc \
# -lavformat -lavcodec -lavutil -lswscale \
# -lSDL2 \
# -lpthread -ldl -lm -latomic \
# -static-libgcc -static-libstdc++"

# !!! NB! Linker flags are managed by Makefile. Do not use RISCV_LIBS in this project.
export RISCV_LIBS=""

# PKG-CONFIG setup for cross-compilation (prefer sysroot pkg-config dirs)
export PATH="$(dirname "${CROSS_PREFIX}"):$PATH"
export PKG_CONFIG_LIBDIR="${SYSROOT}/usr/lib/pkgconfig:${SYSROOT}/usr/lib/riscv64-linux-gnu/pkgconfig:${SYSROOT}/usr/share/pkgconfig"
export PKG_CONFIG_SYSROOT_DIR="${SYSROOT}"
export PKG_CONFIG_PATH="${PKG_CONFIG_LIBDIR}"

# Optional: expose sysroot var for helper scripts
export SPACEMIT_SYSROOT="${SYSROOT}"

# CPU affinity configuration
export NN_CPUS="0,1,2,3"    # Cores with 'ime' extension for NN
export IO_CPUS="4,5,6,7"    # Cores for video I/O

# Performance settings
export CONF_THRESH="0.25"   # YOLOv5 confidence threshold
export IOU_THRESH="0.45"    # NMS IOU threshold

# Build directories
export BUILD_DIR="build"
export LOG_DIR="/data/Work_Logs"

# Device settings
export DEVICE_IP="192.168.1.12"
export DEVICE_USER="svt"
export DEVICE_SSH_NAME="banana" # SSH config name for device in ~/.ssh/config
export DEVICE_WORK_DIR="/data"  # Working directory on SSD /dev/nvme0n1p1 on device 

# Function to check environment
check_env() {
    echo "=== Environment Check ==="
    echo "Toolchain: ${CROSS_PREFIX}-gcc"
    if [ -x "${CROSS_PREFIX}-gcc" ]; then
        echo "  ✓ GCC found: $(${CROSS_PREFIX}-gcc --version | head -1)"
    else
        echo "  ✗ GCC not found!"
    fi
    
    echo "CSI-NN2: ${INSTALL_NN2_PREFIX}/${CPU_MODEL}"
    if [ -f "${INSTALL_NN2_PREFIX}/${CPU_MODEL}/lib/libshl.a" ]; then
        echo "  ✓ libshl.a found"
    else
        echo "  ✗ libshl.a not found!"
    fi
    
    echo "Sysroot: ${SYSROOT}"
    if [ -d "${SYSROOT}/usr/lib/riscv64-linux-gnu" ]; then
        echo "  ✓ Libraries found"
    else
        echo "  ✗ Libraries not found!"
    fi
    
    echo "ISA: ${ISA_FLAGS}"
    echo "Model: ${MODEL_INPUT_SHAPE} (384x640)"
}

# Export functions
if [ -n "${BASH_VERSION:-}" ]; then
    export -f check_env
fi

echo "Environment configured for BPI-F3 YOLOv5n project"
echo "Run 'check_env' to verify configuration"
