# ======================================================================
# Makefile for BPI-F3 YOLOv5n Video Pipeline Project
# Cross-compilation for RISC-V with SpaceMIT toolchain
# ======================================================================

# Use bash for advanced shell features in recipes
SHELL := /bin/bash

# --- Project & device paths (auto-detected) ----------------------------------
# Absolute path to *this* Makefile (robust even if run with -C or from elsewhere)
THIS_MAKEFILE := $(abspath $(lastword $(MAKEFILE_LIST)))
# Directory that contains this Makefile (strip the trailing slash)
PROJECT_ROOT  := $(patsubst %/,%,$(dir $(THIS_MAKEFILE)))
# Basename of the project directory, e.g. "bpi-f3-yolov5n"
PROJECT_NAME  := $(notdir $(PROJECT_ROOT))

# Where to place/run the project on the device (can be overridden by environment)
DEVICE_WORK_DIR ?= /data
DEVICE_PROJECT_DIR := $(DEVICE_WORK_DIR)/$(PROJECT_NAME)

# SSH target (host or user@host); override if needed
SSH_TARGET ?= banana

# Handy helper for remote "cd into project dir"
REMOTE_CD := cd $(DEVICE_PROJECT_DIR)

# Toolchain configuration
COMPILER ?= gcc
CROSS_PREFIX := /opt/spacemit/bin/riscv64-unknown-linux-gnu
ifeq ($(COMPILER),clang)
  CC := clang
  CXX := clang++
else
  CC := $(CROSS_PREFIX)-gcc
  CXX := $(CROSS_PREFIX)-g++
endif
AR := $(CROSS_PREFIX)-ar
STRIP := $(CROSS_PREFIX)-strip

# Architecture flags
ISA_FLAGS := -march=rv64gcv_zvfh -mabi=lp64d

# Optimization flags
OPT_FLAGS := -O3 -ffast-math
DEBUG_FLAGS := -O0 -g3 -DDEBUG
SYSROOT := /opt/spacemit/sysroot
CFLAGS_BASE := $(ISA_FLAGS) --sysroot=$(SYSROOT)
CXXFLAGS_BASE := $(ISA_FLAGS) --sysroot=$(SYSROOT) -std=c++17
ifeq ($(COMPILER),clang)
  CFLAGS_BASE += --target=riscv64-unknown-linux-gnu --gcc-toolchain=/opt/spacemit
  CXXFLAGS_BASE += --target=riscv64-unknown-linux-gnu --gcc-toolchain=/opt/spacemit
endif

# Set build type (default: release)
BUILD_TYPE ?= release
ifeq ($(BUILD_TYPE),debug)
    CFLAGS := $(CFLAGS_BASE) $(DEBUG_FLAGS)
    CXXFLAGS := $(CXXFLAGS_BASE) $(DEBUG_FLAGS)
else
    CFLAGS := $(CFLAGS_BASE) $(OPT_FLAGS) -DNDEBUG
    CXXFLAGS := $(CXXFLAGS_BASE) $(OPT_FLAGS) -DNDEBUG
endif

# CSI-NN2 install root and CPU model (fallback defaults)
CPU_MODEL ?= c920v2
INSTALL_NN2_PREFIX ?= /opt/csi-nn2/install_nn2
CSI_INC_PREFIX := ${INSTALL_NN2_PREFIX}/${CPU_MODEL}
# Include directories
INC_CSI := -I${CSI_INC_PREFIX}/include \
           -I${CSI_INC_PREFIX}/include/csinn \
           -I${CSI_INC_PREFIX}/include/shl_public

INC_OCV := -I/opt/spacemit/sysroot/usr/include/opencv4
INC_FF := -I/opt/spacemit/sysroot/usr/include -I/opt/spacemit/sysroot/usr/include/riscv64-linux-gnu
INC_SDL := -I/opt/spacemit/sysroot/usr/include/SDL2
INC_LOCAL := -I./include

ENABLE_SDL ?= 0
PKG_CONFIG ?= pkg-config
PKG_CONFIG_PATH_SYSROOT := $(SYSROOT)/usr/lib/riscv64-linux-gnu/pkgconfig:$(SYSROOT)/usr/lib/pkgconfig:$(SYSROOT)/usr/share/pkgconfig
PKG_CONFIG_LIBDIR_SYSROOT := $(SYSROOT)/usr/lib/pkgconfig:$(SYSROOT)/usr/lib/riscv64-linux-gnu/pkgconfig
PKG_CONFIG_ENV := PKG_CONFIG_PATH=$(PKG_CONFIG_PATH_SYSROOT) PKG_CONFIG_LIBDIR=$(PKG_CONFIG_LIBDIR_SYSROOT) PKG_CONFIG_SYSROOT_DIR=$(SYSROOT) $(PKG_CONFIG)
PC_CORE_PKGS := libavformat libavcodec libavutil libswscale
PC_OK_CORE := $(shell $(PKG_CONFIG_ENV) --exists $(PC_CORE_PKGS) opencv4 && echo yes || echo no)

ifeq ($(PC_OK_CORE),yes)
    OPENCV_CFLAGS := $(shell $(PKG_CONFIG_ENV) --cflags opencv4 2>/dev/null || echo "")
    FFMPEG_CFLAGS := $(shell $(PKG_CONFIG_ENV) --cflags $(PC_CORE_PKGS) 2>/dev/null || echo "")
    PC_CFLAGS := $(OPENCV_CFLAGS) $(FFMPEG_CFLAGS)
    INCLUDES := $(INC_CSI) $(INC_LOCAL) $(PC_CFLAGS)
else
    INCLUDES := $(INC_CSI) $(INC_OCV) $(INC_FF) $(INC_SDL) $(INC_LOCAL)
endif

SDL_CFLAGS := $(strip $(shell $(PKG_CONFIG_ENV) --cflags sdl2 2>/dev/null || true))
SDL2_LIBS := $(strip $(shell $(PKG_CONFIG_ENV) --libs sdl2 2>/dev/null || true))

SDL_ENABLED :=
ifeq ($(ENABLE_SDL),1)
  SDL_ENABLED := 1
endif
ifneq ($(SDL_CFLAGS),)
  SDL_ENABLED := 1
endif

# Library directories and libraries
LIB_DIR := -L/opt/spacemit/sysroot/usr/lib/riscv64-linux-gnu \
           -L/opt/spacemit/sysroot/lib/riscv64-linux-gnu \
           -L${CSI_INC_PREFIX}/lib

LIBS_CSI := -lshl
OPENCV_LIBS := -lopencv_core -lopencv_imgproc
FFMPEG_LIBS := -lavformat -lavcodec -lavutil -lswscale -lpthread -ldl -lm -lz -latomic
LIBS_SYS := -lpthread -ldl -lm -latomic
LIBS_STATIC := -static-libgcc -static-libstdc++

LDFLAGS := --sysroot=$(SYSROOT) \
           -Wl,-rpath-link=/opt/spacemit/sysroot/usr/lib/riscv64-linux-gnu \
           -Wl,-rpath-link=/opt/spacemit/sysroot/lib/riscv64-linux-gnu \
           $(LIB_DIR)
ifeq ($(COMPILER),clang)
  LDFLAGS += --target=riscv64-unknown-linux-gnu --gcc-toolchain=/opt/spacemit
endif

# Prefer pkg-config link flags when available
LIBS := $(LIBS_CSI) $(OPENCV_LIBS) $(FFMPEG_LIBS) $(LIBS_SYS) $(LIBS_STATIC)

# Minimal libs for simple tests (avoid heavy deps from imgcodecs/videoio)
LIBS_MIN := $(LIBS_CSI) -lopencv_core -lopencv_imgproc $(LIBS_SYS) $(LIBS_STATIC)

# Directories
BUILD_DIR := build
SRC_DIR := src
INC_DIR := include
MODEL_DIR := cpu_model
DEVICE_WORK_DIR := /data

# Create build directory
$(shell mkdir -p $(BUILD_DIR))

# Source files for different targets
# Simple test sources
SIMPLE_TEST_SRCS := yolov5n_simple_test.cpp

# CSI test sources
CSI_TEST_SRCS := yolov5n_csi.cpp

# Pipeline sources (exclude standalone test mains)
PIPELINE_SRCS := $(filter-out $(SRC_DIR)/test_simple_pipeline.cpp, $(wildcard $(SRC_DIR)/*.cpp))
ifneq ($(SDL_ENABLED),)
  CXXFLAGS += $(SDL_CFLAGS) -DHAVE_SDL2
  LIBS += $(SDL2_LIBS)
else
  PIPELINE_SRCS := $(filter-out $(SRC_DIR)/display_sdl.cpp, $(PIPELINE_SRCS))
endif

# Model object (always needed)
MODEL_OBJ := $(BUILD_DIR)/model.o

# Object files
SIMPLE_TEST_OBJS := $(SIMPLE_TEST_SRCS:%.cpp=$(BUILD_DIR)/%.o)
CSI_TEST_OBJS := $(CSI_TEST_SRCS:%.cpp=$(BUILD_DIR)/%.o)
PIPELINE_OBJS := $(PIPELINE_SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Targets (build artifacts)
TARGETS := $(BUILD_DIR)/yolov5n_simple_test $(BUILD_DIR)/yolov5n_csi
TEST_BIN := $(BUILD_DIR)/test_reorderer
PIPELINE_TARGET := yolov5n_pipeline
TEST_PIPELINE_TARGET := test_simple_pipeline

# Default target
all: $(TARGETS)

# Pipeline target (only if sources exist)
ifneq ($(PIPELINE_SRCS),)
pipeline: $(BUILD_DIR)/$(PIPELINE_TARGET)
else
pipeline:
	@echo "Pipeline sources not found in $(SRC_DIR)/"
	@echo "Create the following files first:"
	@echo "  - src/main.cpp"
	@echo "  - src/capture_file.cpp"
	@echo "  - src/preprocess.cpp"
	@echo "  - src/engine_csi.cpp"
	@echo "  - src/pipeline.cpp"
	@echo "  - src/utils.cpp"
endif

# Rules for building executables
$(BUILD_DIR)/yolov5n_simple_test: $(SIMPLE_TEST_OBJS) $(MODEL_OBJ)
	@echo "[LINK] $@"
	@$(CXX) $(LDFLAGS) $^ $(LIBS_MIN) -o $@
	@echo "[SUCCESS] Built $@"
$(BUILD_DIR)/yolov5n_csi: $(CSI_TEST_OBJS) $(MODEL_OBJ)
	@echo "[LINK] $@"
	@$(CXX) $(LDFLAGS) $^ $(LIBS_MIN) -o $@
	@echo "[SUCCESS] Built $@"

$(BUILD_DIR)/$(PIPELINE_TARGET): $(PIPELINE_OBJS) $(MODEL_OBJ)
	@echo "[LINK] $@"
	@$(CXX) $(LDFLAGS) $^ $(LIBS) -o $@
	@echo "[SUCCESS] Built $@"

# Simple test target (uses only core modules)
test-simple-pipeline: $(BUILD_DIR)/$(TEST_PIPELINE_TARGET)
$(BUILD_DIR)/$(TEST_PIPELINE_TARGET): $(BUILD_DIR)/test_simple_pipeline.o $(BUILD_DIR)/preprocess.o $(BUILD_DIR)/engine_csi.o $(MODEL_OBJ)
	@echo "[LINK] $@"
	@$(CXX) $(LDFLAGS) $^ $(LIBS) -o $@
	@echo "[SUCCESS] Built $@"

$(BUILD_DIR)/test_simple_pipeline.o: $(SRC_DIR)/test_simple_pipeline.cpp
	@echo "[CXX] $<"
	@$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Rule for compiling model.c (special case - C not C++)
$(MODEL_OBJ): $(MODEL_DIR)/model.c
	@echo "[CC] $<"
	@$(CC) $(CFLAGS) $(INC_CSI) -c $< -o $@

# Rules for compiling C++ sources
$(BUILD_DIR)/%.o: %.cpp
	@echo "[CXX] $<"
	@$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "[CXX] $<"
	@$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean build artifacts
clean:
	@echo "[CLEAN] Removing build directory"
	@rm -rf $(BUILD_DIR)
	@mkdir -p $(BUILD_DIR)

# Deploy to device
deploy:
	@echo "[DEPLOY] Copying to Banana Pi BPI-F3"
	@./22-send_to_banana.sh

# Test on device (via SSH)
test-simple:
	@echo "[TEST] Running simple test on device"
	@ssh $(SSH_TARGET) "$(REMOTE_CD) && ./build/yolov5n_simple_test"

test-csi:
	@echo "[TEST] Running CSI test on device"
	@ssh $(SSH_TARGET) "$(REMOTE_CD) && ./build/yolov5n_csi photo.ppm output.ppm"

test-pipeline:
	@echo "[TEST] Running pipeline on device"
	@ssh $(SSH_TARGET) "$(REMOTE_CD) && ./build/yolov5n_pipeline --src input_video.mp4 --out result.mp4"

# Run helpers on device
.PHONY: run-file run-v4l2-yuyv run-v4l2-mjpeg
run-file: $(BUILD_DIR)/$(PIPELINE_TARGET)
	@echo "[RUN] file input on device"
	@ssh $(SSH_TARGET) "$(REMOTE_CD) && \
  ./build/yolov5n_pipeline \
    --src file:$(DEVICE_PROJECT_DIR)/input_video.mp4 \
    --out $(DEVICE_PROJECT_DIR)/out.mp4 \
    --display off \
    --weights cpu_model/hhb.bm \
    --nn-cpus auto --io-cpus auto \
    --perf-json $(DEVICE_PROJECT_DIR)/metrics.jsonl"
	@$(MAKE) --no-print-directory cleanup-all || true

.PHONY: run-file-check run-file-fast
run-file-check: $(BUILD_DIR)/$(PIPELINE_TARGET)
	@echo "[RUN] file input with checks on device"
	@ssh $(SSH_TARGET) "set -euo pipefail; $(REMOTE_CD) && \
  timeout 120s ./build/yolov5n_pipeline \
    --src file:$(DEVICE_PROJECT_DIR)/input_video.mp4 \
    --out $(DEVICE_PROJECT_DIR)/out.mp4 \
    --display off \
    --weights cpu_model/hhb.bm \
    --nn-cpus auto --io-cpus auto \
    --perf-json $(DEVICE_PROJECT_DIR)/metrics.jsonl; \
  if ! ffprobe -v error -show_entries format=format_name -of default=nk=1:nw=1 $(DEVICE_PROJECT_DIR)/out.mp4 | grep -q 'mov,mp4'; then echo '[FAIL] ffprobe format'; false; fi; \
  if ! ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=nk=1:nw=1 $(DEVICE_PROJECT_DIR)/out.mp4 | grep -q 'h264'; then echo '[FAIL] ffprobe codec'; false; fi"
	@mkdir -p artifacts
	@scp $(SSH_TARGET):$(DEVICE_PROJECT_DIR)/out.mp4 artifacts/out.mp4
	@scp $(SSH_TARGET):$(DEVICE_PROJECT_DIR)/metrics.jsonl artifacts/metrics.jsonl
	@$(MAKE) --no-print-directory cleanup-all || true

run-file-fast: $(BUILD_DIR)/$(PIPELINE_TARGET)
	@echo "[RUN] file input fast (120 frames) on device"
	@ssh $(SSH_TARGET) "$(REMOTE_CD) && \
  timeout 120s ./build/yolov5n_pipeline \
    --src file:$(DEVICE_PROJECT_DIR)/input_video.mp4 \
    --out $(DEVICE_PROJECT_DIR)/out.mp4 \
    --display off \
    --weights cpu_model/hhb.bm \
    --nn-cpus auto --io-cpus auto \
    --max-frames 120 \
    --perf-json $(DEVICE_PROJECT_DIR)/metrics.jsonl"

.PHONY: run-file-sdl-sw run-cam-yuyv-sdl-sw run-cam-yuyv-sdl-rvv \
        run-file-sdl-sw-local run-file-sdl-rvv-local \
        run-cam-yuyv-sdl-sw-local run-cam-yuyv-sdl-rvv-local

run-file-sdl-sw-local: $(BUILD_DIR)/$(PIPELINE_TARGET)
	@echo "[RUN-LOCAL] file input with SDL (sw)"
	@mkdir -p artifacts
	@PROBE=$(DEVICE_PROJECT_DIR)/artifacts/display_probe_last_file_sdl_sw.ppm; \
	OUT=$(DEVICE_PROJECT_DIR)/artifacts/run-file-sdl-sw-local.avi; \
	MET=$(DEVICE_PROJECT_DIR)/artifacts/run-file-sdl-sw-local.jsonl; \
	rm -f "$$PROBE"; \
	SDL_VIDEODRIVER=wayland ./build/yolov5n_pipeline \
	  --src file:$(DEVICE_PROJECT_DIR)/input_video.mp4 \
	  --out "$$OUT" \
	  --enc raw \
	  --display sdl \
	  --sdl-driver wayland \
	  --display-probe "$$PROBE" \
	  --weights cpu_model/hhb.bm \
	  --nn-cpus auto --io-cpus auto \
	  --pp sw \
	  --perf-json "$$MET"

run-file-sdl-rvv-local: $(BUILD_DIR)/$(PIPELINE_TARGET)
	@echo "[RUN-LOCAL] file input with SDL (rvv)"
	@mkdir -p artifacts
	@PROBE=$(DEVICE_PROJECT_DIR)/artifacts/display_probe_last_file_sdl_rvv.ppm; \
	OUT=$(DEVICE_PROJECT_DIR)/artifacts/run-file-sdl-rvv-local.avi; \
	MET=$(DEVICE_PROJECT_DIR)/artifacts/run-file-sdl-rvv-local.jsonl; \
	rm -f "$$PROBE"; \
	SDL_VIDEODRIVER=wayland ./build/yolov5n_pipeline \
	  --src file:$(DEVICE_PROJECT_DIR)/input_video.mp4 \
	  --out "$$OUT" \
	  --enc raw \
	  --display sdl \
	  --sdl-driver wayland \
	  --display-probe "$$PROBE" \
	  --weights cpu_model/hhb.bm \
	  --nn-cpus auto --io-cpus auto \
	  --pp rvv \
	  --perf-json "$$MET"

run-cam-yuyv-sdl-sw-local: $(BUILD_DIR)/$(PIPELINE_TARGET)
	@echo "[RUN-LOCAL] camera YUYV with SDL (sw)"
	@mkdir -p artifacts
	@PROBE=$(DEVICE_PROJECT_DIR)/artifacts/display_probe_last_cam_yuyv_sdl_sw.ppm; \
	OUT=$(DEVICE_PROJECT_DIR)/artifacts/run-cam-yuyv-sdl-sw-local.avi; \
	MET=$(DEVICE_PROJECT_DIR)/artifacts/run-cam-yuyv-sdl-sw-local.jsonl; \
	rm -f "$$PROBE"; \
	SDL_VIDEODRIVER=wayland ./build/yolov5n_pipeline \
	  --src "v4l2:auto?fmt=yuyv" \
	  --out "$$OUT" \
	  --enc raw \
	  --display sdl \
	  --sdl-driver wayland \
	  --display-probe "$$PROBE" \
	  --weights cpu_model/hhb.bm \
	  --nn-cpus auto --io-cpus auto \
	  --pp sw \
	  --perf-json "$$MET"

run-cam-yuyv-sdl-rvv-local: $(BUILD_DIR)/$(PIPELINE_TARGET)
	@echo "[RUN-LOCAL] camera YUYV with SDL (rvv)"
	@mkdir -p artifacts
	@PROBE=$(DEVICE_PROJECT_DIR)/artifacts/display_probe_last_cam_yuyv_sdl_rvv.ppm; \
	OUT=$(DEVICE_PROJECT_DIR)/artifacts/run-cam-yuyv-sdl-rvv-local.avi; \
	MET=$(DEVICE_PROJECT_DIR)/artifacts/run-cam-yuyv-sdl-rvv-local.jsonl; \
	rm -f "$$PROBE"; \
	SDL_VIDEODRIVER=wayland ./build/yolov5n_pipeline \
	  --src "v4l2:auto?fmt=yuyv" \
	  --out "$$OUT" \
	  --enc raw \
	  --display sdl \
	  --sdl-driver wayland \
	  --display-probe "$$PROBE" \
	  --weights cpu_model/hhb.bm \
	  --nn-cpus auto --io-cpus auto \
	  --pp rvv \
	  --perf-json "$$MET"
run-file-sdl-sw: $(BUILD_DIR)/$(PIPELINE_TARGET)
	@echo "[RUN] file input with SDL (sw) on device"
	@$(MAKE) --no-print-directory cleanup-all || true
	@echo "mode|driver|in_fps|out_fps|disp_ms|file_size"
	@ssh $(SSH_TARGET) bash -lc 'set -Eeuo pipefail; cd $(DEVICE_PROJECT_DIR); \
	  mkdir -p artifacts; \
	  OUT=artifacts/run-file-sdl-sw.avi; \
	  MET=artifacts/run-file-sdl-sw.jsonl; \
	  LOG=artifacts/run-file-sdl-sw.log; \
	  PROBE=artifacts/display_probe_last_file_sdl_sw.ppm; \
	  rm -f "$$OUT" "$$MET" "$$LOG" "$$PROBE"; \
	  timeout 180s ./build/yolov5n_pipeline \
	    --src file:$(DEVICE_PROJECT_DIR)/input_video.mp4 \
	    --out "$$OUT" \
	    --enc raw \
	    --display sdl \
	    --sdl-driver auto \
	    --display-probe "$$PROBE" \
	    --weights cpu_model/hhb.bm \
	    --nn-cpus auto --io-cpus auto \
	    --pp sw \
	    --max-frames 120 \
	    --perf-json "$$MET" > "$$LOG" 2>&1; \
	  test -s "$$PROBE"; \
	  DRIVER=$$(grep -m1 "\\[display\\] SDL driver=" "$$LOG" | sed -E "s/.*SDL driver=([^ ]+) .*/\\1/" || echo null); \
	  METRIC=$$(grep "\\[metrics\\]" "$$LOG" | tail -n1 || echo ""); \
	  IN_FPS=$$(echo "$$METRIC" | sed -n "s/.*in_fps=\([0-9.]*\).*/\\1/p"); \
	  OUT_FPS=$$(echo "$$METRIC" | sed -n "s/.*out_fps=\([0-9.]*\).*/\\1/p"); \
	  DISP=$$(echo "$$METRIC" | sed -n "s/.*disp_ms=\([0-9.]*\).*/\\1/p"); \
	  SIZE=$$(stat -c%s "$$OUT" 2>/dev/null || echo 0); \
	  echo "file-sdl-sw|$$DRIVER|$$IN_FPS|$$OUT_FPS|$$DISP|$$SIZE"'
	@mkdir -p artifacts
	@scp $(SSH_TARGET):$(DEVICE_PROJECT_DIR)/artifacts/display_probe_last_file_sdl_sw.ppm artifacts/display_probe_last_file_sdl_sw.ppm >/dev/null || echo "[WARN] missing display probe"
	@[ -s artifacts/display_probe_last_file_sdl_sw.ppm ] || echo "[WARN] display probe artifact empty"
	@scp $(SSH_TARGET):$(DEVICE_PROJECT_DIR)/artifacts/run-file-sdl-sw.log artifacts/run-file-sdl-sw.log >/dev/null || true
	@scp $(SSH_TARGET):$(DEVICE_PROJECT_DIR)/artifacts/run-file-sdl-sw.jsonl artifacts/run-file-sdl-sw.jsonl >/dev/null || true
	@scp $(SSH_TARGET):$(DEVICE_PROJECT_DIR)/artifacts/run-file-sdl-sw.avi artifacts/run-file-sdl-sw.avi >/dev/null || true
	@$(MAKE) --no-print-directory cleanup-all || true

run-cam-yuyv-sdl-sw: $(BUILD_DIR)/$(PIPELINE_TARGET)
	@echo "[RUN] camera YUYV SDL (sw) on device"
	@$(MAKE) --no-print-directory cleanup-all || true
	@echo "mode|driver|in_fps|out_fps|disp_ms|file_size"
	@ssh $(SSH_TARGET) bash -lc 'set -Eeuo pipefail; cd $(DEVICE_PROJECT_DIR); \
	  if ! command -v v4l2-ctl >/dev/null 2>&1; then echo NO_CAMERA; exit 0; fi; \
	  mkdir -p artifacts; \
	  CAM=""; \
	  for link in /dev/v4l/by-id/*-video-index0; do \
	    [ -e "$$link" ] || continue; \
	    real_path=$$(readlink -f "$$link" 2>/dev/null || true); \
	    [ -n "$$real_path" ] || continue; \
	    CAM="$$real_path"; \
	    break; \
	  done; \
	  if [ -z "$$CAM" ]; then \
	    for dev in /dev/video{0..63}; do \
	      [ -e "$$dev" ] || continue; \
	      CAM="$$dev"; \
	      break; \
	    done; \
	  fi; \
	  if [ -z "$$CAM" ]; then echo NO_CAMERA; exit 0; fi; \
	  OUT=artifacts/run-cam-yuyv-sdl-sw.avi; \
	  MET=artifacts/run-cam-yuyv-sdl-sw.jsonl; \
	  LOG=artifacts/run-cam-yuyv-sdl-sw.log; \
	  PROBE=artifacts/display_probe_last_cam_yuyv_sdl_sw.ppm; \
	  rm -f "$$OUT" "$$MET" "$$LOG" "$$PROBE"; \
	  timeout 180s ./build/yolov5n_pipeline \
	    --src "v4l2:$$CAM?fmt=yuyv" \
	    --out "$$OUT" \
	    --enc raw \
	    --display sdl \
	    --sdl-driver auto \
	    --display-probe "$$PROBE" \
	    --weights cpu_model/hhb.bm \
	    --nn-cpus auto --io-cpus auto \
	    --pp sw \
	    --max-frames 120 \
	    --perf-json "$$MET" > "$$LOG" 2>&1; \
	  test -s "$$PROBE"; \
	  DRIVER=$$(grep -m1 "\\[display\\] SDL driver=" "$$LOG" | sed -E "s/.*SDL driver=([^ ]+) .*/\\1/" || echo null); \
	  METRIC=$$(grep "\\[metrics\\]" "$$LOG" | tail -n1 || echo ""); \
	  IN_FPS=$$(echo "$$METRIC" | sed -n "s/.*in_fps=\([0-9.]*\).*/\\1/p"); \
	  OUT_FPS=$$(echo "$$METRIC" | sed -n "s/.*out_fps=\([0-9.]*\).*/\\1/p"); \
	  DISP=$$(echo "$$METRIC" | sed -n "s/.*disp_ms=\([0-9.]*\).*/\\1/p"); \
	  SIZE=$$(stat -c%s "$$OUT" 2>/dev/null || echo 0); \
	  echo "cam-yuyv-sdl-sw|$$DRIVER|$$IN_FPS|$$OUT_FPS|$$DISP|$$SIZE"'
	@mkdir -p artifacts
	@scp $(SSH_TARGET):$(DEVICE_PROJECT_DIR)/artifacts/display_probe_last_cam_yuyv_sdl_sw.ppm artifacts/display_probe_last_cam_yuyv_sdl_sw.ppm >/dev/null || echo "[WARN] missing display probe"
	@[ -s artifacts/display_probe_last_cam_yuyv_sdl_sw.ppm ] || echo "[WARN] display probe artifact empty"
	@scp $(SSH_TARGET):$(DEVICE_PROJECT_DIR)/artifacts/run-cam-yuyv-sdl-sw.log artifacts/run-cam-yuyv-sdl-sw.log >/dev/null || true
	@scp $(SSH_TARGET):$(DEVICE_PROJECT_DIR)/artifacts/run-cam-yuyv-sdl-sw.jsonl artifacts/run-cam-yuyv-sdl-sw.jsonl >/dev/null || true
	@scp $(SSH_TARGET):$(DEVICE_PROJECT_DIR)/artifacts/run-cam-yuyv-sdl-sw.avi artifacts/run-cam-yuyv-sdl-sw.avi >/dev/null || true
	@$(MAKE) --no-print-directory cleanup-all || true

run-cam-yuyv-sdl-rvv: $(BUILD_DIR)/$(PIPELINE_TARGET)
	@echo "[RUN] camera YUYV SDL (rvv) on device"
	@$(MAKE) --no-print-directory cleanup-all || true
	@echo "mode|driver|in_fps|out_fps|disp_ms|file_size"
	@ssh $(SSH_TARGET) bash -lc 'set -Eeuo pipefail; cd $(DEVICE_PROJECT_DIR); \
	  if ! command -v v4l2-ctl >/dev/null 2>&1; then echo NO_CAMERA; exit 0; fi; \
	  mkdir -p artifacts; \
	  CAM=""; \
	  for link in /dev/v4l/by-id/*-video-index0; do \
	    [ -e "$$link" ] || continue; \
	    real_path=$$(readlink -f "$$link" 2>/dev/null || true); \
	    [ -n "$$real_path" ] || continue; \
	    CAM="$$real_path"; \
	    break; \
	  done; \
	  if [ -z "$$CAM" ]; then \
	    for dev in /dev/video{0..63}; do \
	      [ -e "$$dev" ] || continue; \
	      CAM="$$dev"; \
	      break; \
	    done; \
	  fi; \
	  if [ -z "$$CAM" ]; then echo NO_CAMERA; exit 0; fi; \
	  OUT=artifacts/run-cam-yuyv-sdl-rvv.avi; \
	  MET=artifacts/run-cam-yuyv-sdl-rvv.jsonl; \
	  LOG=artifacts/run-cam-yuyv-sdl-rvv.log; \
	  PROBE=artifacts/display_probe_last_cam_yuyv_sdl_rvv.ppm; \
	  rm -f "$$OUT" "$$MET" "$$LOG" "$$PROBE"; \
	  timeout 180s ./build/yolov5n_pipeline \
	    --src "v4l2:$$CAM?fmt=yuyv" \
	    --out "$$OUT" \
	    --enc raw \
	    --display sdl \
	    --sdl-driver auto \
	    --display-probe "$$PROBE" \
	    --weights cpu_model/hhb.bm \
	    --nn-cpus auto --io-cpus auto \
	    --pp rvv \
	    --max-frames 120 \
	    --perf-json "$$MET" > "$$LOG" 2>&1; \
	  test -s "$$PROBE"; \
	  DRIVER=$$(grep -m1 "\\[display\\] SDL driver=" "$$LOG" | sed -E "s/.*SDL driver=([^ ]+) .*/\\1/" || echo null); \
	  METRIC=$$(grep "\\[metrics\\]" "$$LOG" | tail -n1 || echo ""); \
	  IN_FPS=$$(echo "$$METRIC" | sed -n "s/.*in_fps=\([0-9.]*\).*/\\1/p"); \
	  OUT_FPS=$$(echo "$$METRIC" | sed -n "s/.*out_fps=\([0-9.]*\).*/\\1/p"); \
	  DISP=$$(echo "$$METRIC" | sed -n "s/.*disp_ms=\([0-9.]*\).*/\\1/p"); \
	  SIZE=$$(stat -c%s "$$OUT" 2>/dev/null || echo 0); \
	  echo "cam-yuyv-sdl-rvv|$$DRIVER|$$IN_FPS|$$OUT_FPS|$$DISP|$$SIZE"'
	@mkdir -p artifacts
	@scp $(SSH_TARGET):$(DEVICE_PROJECT_DIR)/artifacts/display_probe_last_cam_yuyv_sdl_rvv.ppm artifacts/display_probe_last_cam_yuyv_sdl_rvv.ppm >/dev/null || echo "[WARN] missing display probe"
	@[ -s artifacts/display_probe_last_cam_yuyv_sdl_rvv.ppm ] || echo "[WARN] display probe artifact empty"
	@scp $(SSH_TARGET):$(DEVICE_PROJECT_DIR)/artifacts/run-cam-yuyv-sdl-rvv.log artifacts/run-cam-yuyv-sdl-rvv.log >/dev/null || true
	@scp $(SSH_TARGET):$(DEVICE_PROJECT_DIR)/artifacts/run-cam-yuyv-sdl-rvv.jsonl artifacts/run-cam-yuyv-sdl-rvv.jsonl >/dev/null || true
	@scp $(SSH_TARGET):$(DEVICE_PROJECT_DIR)/artifacts/run-cam-yuyv-sdl-rvv.avi artifacts/run-cam-yuyv-sdl-rvv.avi >/dev/null || true
	@$(MAKE) --no-print-directory cleanup-all || true

run-v4l2-yuyv: $(BUILD_DIR)/$(PIPELINE_TARGET)
	@echo "[RUN] v4l2:yuyv on device"
	@ssh $(SSH_TARGET) "$(REMOTE_CD) && \
  ./build/yolov5n_pipeline \
    --src v4l2:/dev/video0 \
    --out $(DEVICE_PROJECT_DIR)/out.mkv \
    --display off \
    --weights cpu_model/hhb.bm \
    --nn-cpus auto --io-cpus auto \
    --perf-json $(DEVICE_PROJECT_DIR)/metrics_v4l2_yuyv.jsonl"

run-v4l2-mjpeg: $(BUILD_DIR)/$(PIPELINE_TARGET)
	@echo "[RUN] v4l2:mjpeg on device"
	@ssh $(SSH_TARGET) "$(REMOTE_CD) && \
  ./build/yolov5n_pipeline \
    --src v4l2:/dev/video0 \
    --out $(DEVICE_PROJECT_DIR)/out_mjpeg.mkv \
    --display off \
    --weights cpu_model/hhb.bm \
    --nn-cpus auto --io-cpus auto \
    --perf-json $(DEVICE_PROJECT_DIR)/metrics_v4l2_mjpeg.jsonl"

.PHONY: run-cam-probe run-cam-yuyv-sw run-cam-yuyv-rvv
run-cam-probe: $(BUILD_DIR)/$(PIPELINE_TARGET)
	@echo "[RUN] camera probe on device"
	@ssh $(SSH_TARGET) bash -lc 'set -Eeuo pipefail; cd $(DEVICE_PROJECT_DIR); \
	  ls -l /dev/v4l/by-id 2>/dev/null || echo "no /dev/v4l/by-id directory"; \
	  if command -v v4l2-ctl >/dev/null 2>&1; then \
	    v4l2-ctl --list-devices 2>/dev/null || true; \
	    found=0; \
	    for dev in /dev/video{0..63}; do \
	      [ -e "$$dev" ] || continue; \
	      found=1; \
	      echo "== $$dev =="; \
	      v4l2-ctl -D -d "$$dev" 2>/dev/null || echo "[WARN] v4l2-ctl -D failed for $$dev"; \
	      v4l2-ctl --list-formats-ext -d "$$dev" 2>/dev/null | sed -n "1,60p" || true; \
	    done; \
	    if [ "$$found" -eq 0 ]; then echo NO_CAMERA; fi; \
	  else \
	    echo "[WARN] v4l2-ctl not available"; \
	    echo NO_CAMERA; \
	  fi'

run-cam-yuyv-sw: $(BUILD_DIR)/$(PIPELINE_TARGET)
	@echo "[RUN] camera (YUYV sw) on device"
	@$(MAKE) --no-print-directory cleanup-all || true
	@ssh $(SSH_TARGET) bash -lc 'set -Eeuo pipefail >/dev/null; cd $(DEVICE_PROJECT_DIR); \
	  if ! command -v v4l2-ctl >/dev/null 2>&1; then echo NO_CAMERA; exit 0; fi; \
	  cam=""; \
	  for link in /dev/v4l/by-id/*-video-index0; do \
	    [ -e "$$link" ] || continue; \
	    real_path=$$(readlink -f "$$link" 2>/dev/null || true); \
	    [ -n "$$real_path" ] || continue; \
	    cam="$$real_path"; \
	    break; \
	  done; \
	  if [ -z "$$cam" ]; then \
	    for dev in /dev/video{0..63}; do \
	      [ -e "$$dev" ] || continue; \
	      cam="$$dev"; \
	      break; \
	    done; \
	  fi; \
	  if [ -z "$$cam" ]; then echo NO_CAMERA; exit 0; fi; \
	  echo "[INFO] camera candidate: $$cam"; \
	  OUT_FILE=out_cam_yuyv_sw.avi; \
	  METRICS_FILE=metrics_cam_yuyv_sw.jsonl; \
	  rm -f "$$OUT_FILE" "$$METRICS_FILE"; \
	  timeout 240s ./build/yolov5n_pipeline \
	    --src v4l2:auto \
	    --cam-fmt yuyv \
	    --pp sw \
	    --enc raw \
	    --display off \
	    --weights cpu_model/hhb.bm \
	    --nn-cpus auto --io-cpus auto \
	    --max-frames 120 \
	    --out "$$OUT_FILE" \
	    --perf-json "$$METRICS_FILE"; \
	  if [ ! -s "$$METRICS_FILE" ]; then echo "[ERROR] metrics file empty" >&2; exit 1; fi; \
	  if command -v ffprobe >/dev/null 2>&1; then \
	    ffprobe -v error -show_entries format=format_name -of default=nk=1:nw=1 "$$OUT_FILE" | grep -q "avi" || { echo "[ERROR] ffprobe format check failed" >&2; exit 1; }; \
	    ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=nk=1:nw=1 "$$OUT_FILE" | grep -q "rawvideo" || { echo "[ERROR] ffprobe codec check failed" >&2; exit 1; }; \
	  else \
	    echo "[WARN] ffprobe not available, skipping validation"; \
	  fi'
	@mkdir -p artifacts
	@scp $(SSH_TARGET):$(DEVICE_PROJECT_DIR)/out_cam_yuyv_sw.avi artifacts/out_cam_yuyv_sw.avi >/dev/null || echo "[WARN] missing out_cam_yuyv_sw.avi"
	@scp $(SSH_TARGET):$(DEVICE_PROJECT_DIR)/metrics_cam_yuyv_sw.jsonl artifacts/metrics_cam_yuyv_sw.jsonl >/dev/null || echo "[WARN] missing metrics_cam_yuyv_sw.jsonl"
	@$(MAKE) --no-print-directory cleanup-all || true

run-cam-yuyv-rvv: $(BUILD_DIR)/$(PIPELINE_TARGET)
	@echo "[RUN] camera (YUYV rvv) on device"
	@$(MAKE) --no-print-directory cleanup-all || true
	@ssh $(SSH_TARGET) bash -lc 'set -Eeuo pipefail; cd $(DEVICE_PROJECT_DIR); \
	  if ! command -v v4l2-ctl >/dev/null 2>&1; then echo NO_CAMERA; exit 0; fi; \
	  cam=""; \
	  for link in /dev/v4l/by-id/*-video-index0; do \
	    [ -e "$$link" ] || continue; \
	    real_path=$$(readlink -f "$$link" 2>/dev/null || true); \
	    [ -n "$$real_path" ] || continue; \
	    cam="$$real_path"; \
	    break; \
	  done; \
	  if [ -z "$$cam" ]; then \
	    for dev in /dev/video{0..63}; do \
	      [ -e "$$dev" ] || continue; \
	      cam="$$dev"; \
	      break; \
	    done; \
	  fi; \
	  if [ -z "$$cam" ]; then echo NO_CAMERA; exit 0; fi; \
	  echo "[INFO] camera candidate: $$cam"; \
	  OUT_FILE=out_cam_yuyv_rvv.avi; \
	  METRICS_FILE=metrics_cam_yuyv_rvv.jsonl; \
	  rm -f "$$OUT_FILE" "$$METRICS_FILE"; \
	  timeout 240s ./build/yolov5n_pipeline \
	    --src v4l2:auto \
	    --cam-fmt yuyv \
	    --pp rvv \
	    --enc raw \
	    --display off \
	    --weights cpu_model/hhb.bm \
	    --nn-cpus auto --io-cpus auto \
	    --max-frames 120 \
	    --out "$$OUT_FILE" \
	    --perf-json "$$METRICS_FILE"; \
	  if [ ! -s "$$METRICS_FILE" ]; then echo "[ERROR] metrics file empty" >&2; exit 1; fi; \
	  if command -v ffprobe >/dev/null 2>&1; then \
	    ffprobe -v error -show_entries format=format_name -of default=nk=1:nw=1 "$$OUT_FILE" | grep -q "avi" || { echo "[ERROR] ffprobe format check failed" >&2; exit 1; }; \
	    ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=nk=1:nw=1 "$$OUT_FILE" | grep -q "rawvideo" || { echo "[ERROR] ffprobe codec check failed" >&2; exit 1; }; \
	  else \
	    echo "[WARN] ffprobe not available, skipping validation"; \
	  fi'
	@mkdir -p artifacts
	@scp $(SSH_TARGET):$(DEVICE_PROJECT_DIR)/out_cam_yuyv_rvv.avi artifacts/out_cam_yuyv_rvv.avi >/dev/null || echo "[WARN] missing out_cam_yuyv_rvv.avi"
	@scp $(SSH_TARGET):$(DEVICE_PROJECT_DIR)/metrics_cam_yuyv_rvv.jsonl artifacts/metrics_cam_yuyv_rvv.jsonl >/dev/null || echo "[WARN] missing metrics_cam_yuyv_rvv.jsonl"
	@$(MAKE) --no-print-directory cleanup-all || true

# Benchmarks on device (file source, max 120 frames)
.PHONY: run-bench-raw run-bench-mjpeg run-bench-h264 run-bench clean-remote-procs clean-local-procs cleanup-all

clean-remote-procs:
	@echo "[CLEAN] remote processes"
	@ssh $(SSH_TARGET) '\
	  for p in yolov5n_pipeline gdbserver; do \
	    pkill -TERM -f "$$p" >/dev/null 2>&1 || true; \
	  done; sleep 1; \
	  for p in yolov5n_pipeline gdbserver; do \
	    pkill -KILL -f "$$p" >/dev/null 2>&1 || true; \
	  done; \
	  exit 0' || true

.PHONY: clean-local-procs
clean-local-procs:
	@echo "[CLEAN] local ssh clients for yolov5n_pipeline"
	@pids="$$(pgrep -af 'ssh .*yolov5n_pipeline' | grep -v -E 'make|run-bench' | awk '{print $$1}')" ; \
	[ -z "$$pids" ] || { echo "  - killing: $$pids"; kill -TERM $$pids || true; sleep 1; kill -KILL $$pids || true; } ; true

cleanup-all: clean-remote-procs clean-local-procs
run-bench-raw: $(BUILD_DIR)/$(PIPELINE_TARGET)
	@echo "[BENCH] raw encoder (120 frames)"
	@$(MAKE) --no-print-directory cleanup-all || true
	@ssh $(SSH_TARGET) 'set +e; set -u; $(REMOTE_CD) && \
  ./build/yolov5n_pipeline --src file:$(DEVICE_PROJECT_DIR)/input_video.mp4 \
    --out $(DEVICE_PROJECT_DIR)/out_raw.avi --enc raw --display off \
    --weights cpu_model/hhb.bm --nn-cpus auto --io-cpus auto --max-frames 120 \
    $(if $(pp),--pp $(pp),) \
    --perf-json $(DEVICE_PROJECT_DIR)/metrics_raw.jsonl; \
  if command -v ffprobe >/dev/null 2>&1; then \
    ffprobe -v error -show_entries format=format_name -of default=nk=1:nw=1 out_raw.avi 2>/dev/null || echo "[WARN] ffprobe format check failed"; \
    ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=nk=1:nw=1 out_raw.avi 2>/dev/null || echo "[WARN] ffprobe codec check failed"; \
  else \
    (sudo -n apt-get update -y && sudo -n apt-get install -y ffmpeg) || true; \
    if command -v ffprobe >/dev/null 2>&1; then \
      ffprobe -v error -show_entries format=format_name -of default=nk=1:nw=1 out_raw.avi 2>/dev/null || echo "[WARN] ffprobe format check failed"; \
      ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=nk=1:nw=1 out_raw.avi 2>/dev/null || echo "[WARN] ffprobe codec check failed"; \
    else \
      echo "[WARN] ffprobe missing on device, skipping validation"; \
    fi; \
  fi; \
  exit 0'
	@mkdir -p artifacts
	@scp $(SSH_TARGET):$(DEVICE_PROJECT_DIR)/out_raw.avi artifacts/out_raw.avi || true
	@scp $(SSH_TARGET):$(DEVICE_PROJECT_DIR)/metrics_raw.jsonl artifacts/metrics_raw.jsonl || true
	@$(MAKE) --no-print-directory cleanup-all || true

run-bench-mjpeg: $(BUILD_DIR)/$(PIPELINE_TARGET)
	@echo "[BENCH] mjpeg encoder (120 frames)"
	@$(MAKE) --no-print-directory cleanup-all || true
	@ssh $(SSH_TARGET) 'set +e; set -u; $(REMOTE_CD) && \
  ./build/yolov5n_pipeline --src file:$(DEVICE_PROJECT_DIR)/input_video.mp4 \
    --out $(DEVICE_PROJECT_DIR)/out_mjpeg.avi --enc mjpeg --display off \
    --weights cpu_model/hhb.bm --nn-cpus auto --io-cpus auto --max-frames 120 \
    $(if $(pp),--pp $(pp),) \
    --perf-json $(DEVICE_PROJECT_DIR)/metrics_mjpeg.jsonl; \
  if command -v ffprobe >/dev/null 2>&1; then \
    ffprobe -v error -show_entries format=format_name -of default=nk=1:nw=1 out_mjpeg.avi 2>/dev/null || echo "[WARN] ffprobe format check failed"; \
    ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=nk=1:nw=1 out_mjpeg.avi 2>/dev/null || echo "[WARN] ffprobe codec check failed"; \
  else \
    (sudo -n apt-get update -y && sudo -n apt-get install -y ffmpeg) || true; \
    if command -v ffprobe >/dev/null 2>&1; then \
      ffprobe -v error -show_entries format=format_name -of default=nk=1:nw=1 out_mjpeg.avi 2>/dev/null || echo "[WARN] ffprobe format check failed"; \
      ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=nk=1:nw=1 out_mjpeg.avi 2>/dev/null || echo "[WARN] ffprobe codec check failed"; \
    else \
      echo "[WARN] ffprobe missing on device, skipping validation"; \
    fi; \
  fi; \
  exit 0'
	@mkdir -p artifacts
	@scp $(SSH_TARGET):$(DEVICE_PROJECT_DIR)/out_mjpeg.avi artifacts/out_mjpeg.avi || true
	@scp $(SSH_TARGET):$(DEVICE_PROJECT_DIR)/metrics_mjpeg.jsonl artifacts/metrics_mjpeg.jsonl || true
	@$(MAKE) --no-print-directory cleanup-all || true

run-bench-h264: $(BUILD_DIR)/$(PIPELINE_TARGET)
	@echo "[BENCH] h264 encoder (120 frames)"
	@$(MAKE) --no-print-directory cleanup-all || true
	@ssh $(SSH_TARGET) 'set +e; set -u; $(REMOTE_CD) && \
  ./build/yolov5n_pipeline --src file:$(DEVICE_PROJECT_DIR)/input_video.mp4 \
    --out $(DEVICE_PROJECT_DIR)/out_h264.mp4 --enc h264 --display off \
    --weights cpu_model/hhb.bm --nn-cpus auto --io-cpus auto --max-frames 120 \
    $(if $(pp),--pp $(pp),) \
    --perf-json $(DEVICE_PROJECT_DIR)/metrics_h264.jsonl; \
  if command -v ffprobe >/dev/null 2>&1; then \
    ffprobe -v error -show_entries format=format_name -of default=nk=1:nw=1 out_h264.mp4 2>/dev/null || echo "[WARN] ffprobe format check failed"; \
    ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=nk=1:nw=1 out_h264.mp4 2>/dev/null || echo "[WARN] ffprobe codec check failed"; \
  else \
    (sudo -n apt-get update -y && sudo -n apt-get install -y ffmpeg) || true; \
    if command -v ffprobe >/dev/null 2>&1; then \
      ffprobe -v error -show_entries format=format_name -of default=nk=1:nw=1 out_h264.mp4 2>/dev/null || echo "[WARN] ffprobe format check failed"; \
      ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=nk=1:nw=1 out_h264.mp4 2>/dev/null || echo "[WARN] ffprobe codec check failed"; \
    else \
      echo "[WARN] ffprobe missing on device, skipping validation"; \
    fi; \
  fi; \
  exit 0'
	@mkdir -p artifacts
	@scp $(SSH_TARGET):$(DEVICE_PROJECT_DIR)/out_h264.mp4 artifacts/out_h264.mp4 || true
	@scp $(SSH_TARGET):$(DEVICE_PROJECT_DIR)/metrics_h264.jsonl artifacts/metrics_h264.jsonl || true
	@$(MAKE) --no-print-directory cleanup-all || true

run-bench:
	@echo "[BENCH] running all encoders (raw,mjpeg,h264)"
	@$(MAKE) -j1 --no-print-directory run-bench-raw || true
	@$(MAKE) -j1 --no-print-directory run-bench-mjpeg || true
	@$(MAKE) -j1 --no-print-directory run-bench-h264 || true
	@echo "[BENCH] done"

 

# Dependency checks via pkg-config inside sysroot
.PHONY: deps-check deps-install
deps-check:
	@bash -lc '. ./env.sh; \
	PKG_PATH="$(PKG_CONFIG_PATH_SYSROOT)"; \
	PKG_LIBDIR="$(PKG_CONFIG_LIBDIR_SYSROOT)"; \
	STATUS=0; \
	for p in sdl2 libavformat libavcodec libavutil libswscale opencv4; do \
	  if PKG_CONFIG_PATH="$$PKG_PATH" PKG_CONFIG_LIBDIR="$$PKG_LIBDIR" PKG_CONFIG_SYSROOT_DIR="$(SYSROOT)" pkg-config --exists $$p; then \
	    echo "[deps] $$p: PASS"; \
	  else \
	    echo "[deps] $$p: FAIL"; \
	    STATUS=1; \
	  fi; \
	done; \
	exit $$STATUS' 

deps-install:
	@echo "[DEPS] Installing/updating sysroot via install_all_libs_to_spacemit.sh"
	@bash ./install_all_libs_to_spacemit.sh

# Check environment
check:
	@echo "=== Build Environment Check ==="
	@echo "Compiler: $(CC)"
	@$(CC) --version | head -1
	@echo "C++ Compiler: $(CXX)"
	@$(CXX) --version | head -1
	@echo "ISA Flags: $(ISA_FLAGS)"
	@echo "Build Type: $(BUILD_TYPE)"
	@echo ""
	@echo "=== Library Check ==="
	@if [ -f ${CSI_INC_PREFIX}/lib/libshl.a ]; then \
		echo "✓ CSI-NN2 library found"; \
	else \
		echo "✗ CSI-NN2 library NOT found!"; \
	fi
	@if [ -d /opt/spacemit/sysroot/usr/lib/riscv64-linux-gnu ]; then \
		echo "✓ SpaceMIT sysroot found"; \
	else \
		echo "✗ SpaceMIT sysroot NOT found!"; \
	fi

# Help
help:
	@echo "YOLOv5n BPI-F3 Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all              - Build all test programs"
	@echo "  pipeline         - Build video pipeline (if sources exist)"
	@echo "  fix-sysroot-sdl  - Ensure libSDL2.so symlink exists in sysroot"
	@echo "  clean            - Remove build artifacts"
	@echo "  deploy           - Copy to device"
	@echo "  test-simple      - Run simple test on device"
	@echo "  test-csi         - Run CSI test on device"
	@echo "  test-pipeline    - Run pipeline on device"
	@echo "  check            - Check build environment"
	@echo ""
	@echo "Build types:"
	@echo "  make BUILD_TYPE=release  (default)"
	@echo "  make BUILD_TYPE=debug"
	@echo ""
	@echo "Examples:"
	@echo "  make                     # Build all tests"
	@echo "  make pipeline            # Build pipeline"
	@echo "  make clean all           # Clean and rebuild"
	@echo "  make deploy test-csi     # Deploy and test"

.PHONY: all pipeline clean deploy test-simple test-csi test-pipeline check help

# Verbose output (set V=1 for verbose)
ifndef V
.SILENT:
endif

# Utilities
fix-sysroot-sdl:
	@echo "[FIX] Creating libSDL2.so symlink in sysroot if missing"
	@[ -e /opt/spacemit/sysroot/usr/lib/riscv64-linux-gnu/libSDL2.so ] || \
	 (cd /opt/spacemit/sysroot/usr/lib/riscv64-linux-gnu && (ln -sf libSDL2-2.0.so.0 libSDL2.so || sudo ln -sf libSDL2-2.0.so.0 libSDL2.so) && echo "  ✓ libSDL2.so -> libSDL2-2.0.so.0" || echo "  ✗ Permission denied: run installer or use sudo")
tests: $(TEST_BIN)

$(TEST_BIN): tests/test_reorderer.cpp $(BUILD_DIR)/pipeline.o
	@echo "[CXX] $<"
	@$(CXX) $(CXXFLAGS) $(INCLUDES) -c tests/test_reorderer.cpp -o $(BUILD_DIR)/test_reorderer.o
	@echo "[LINK] $@"
	@$(CXX) $(LDFLAGS) $(BUILD_DIR)/test_reorderer.o $(BUILD_DIR)/pipeline.o $(LIBS_MIN) -o $@
	@echo "[SUCCESS] Built $@"

# Override run-bench-summary with a thin wrapper script to avoid complex quoting
.PHONY: run-bench-summary
run-bench-summary: pipeline deploy
	@PP=$(pp) bash tools/run_bench_summary.sh
