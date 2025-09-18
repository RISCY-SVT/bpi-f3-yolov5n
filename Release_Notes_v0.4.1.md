## Post-release update (2025-09-17)

### Highlights
- Eliminated RSS growth by caching CSI-NN2 output tensors and reusing per-worker buffers.
- Added `--mem-json <path>` for per-second RSS/VM JSONL logging during endurance runs.
- Hardened FFmpeg encoder finalization (drain → trailer → `avio_close`) even on error paths.
- Camera YUYV + SDL live display now auto-probes `v4l2:auto?fmt=yuyv` and works on Wayland (`SDL_VIDEODRIVER=wayland`).

### How to reproduce memory logs
- File source (software preprocess, 5–10 min):
  ```bash
  ./yolov5n_pipeline \
    --src file:/data/bpi-f3-yolov5n/input_video.mp4 \
    --out /data/bpi-f3-yolov5n/out_leakcheck_file.avi --enc raw \
    --display off --pp sw \
    --perf-json /data/bpi-f3-yolov5n/metrics_leakcheck_file.jsonl \
    --mem-json  /data/bpi-f3-yolov5n/mem_leakcheck_file.jsonl
  ```
- Camera YUYV (RVV preprocess, 5–10 min):
  ```bash
  ./yolov5n_pipeline \
    --src 'v4l2:auto?fmt=yuyv' \
    --display off --pp rvv \
    --weights /data/bpi-f3-yolov5n/cpu_model/hhb.bm \
    --perf-json /data/bpi-f3-yolov5n/metrics_leakcheck_cam.jsonl \
    --mem-json  /data/bpi-f3-yolov5n/mem_leakcheck_cam.jsonl
  ```

### Observed results snapshot
- File run (10 min): RSS Δ around −20 MB…+20 MB (steady-state plateau after warm-up).
- Camera run (5 min): RSS Δ within roughly ±50 MB (V4L2 buffers influence peak though trend is flat).

### Artifacts & references
- `artifacts/mem_leakcheck_file.jsonl`, `artifacts/mem_leakcheck_cam.jsonl`
- `artifacts/metrics_leakcheck_file.jsonl`, `artifacts/metrics_leakcheck_cam.jsonl`
- Output videos: `out_leakcheck_file.avi`, `out_leakcheck_cam.avi` on the device.

# bpi-f3-yolov5n v0.4.0 (Beta)

**Highlights**
- V4L2 YUYV camera capture with auto-probe (`v4l2:auto?fmt=yuyv`)
- Live display via SDL2 (Wayland/EGL on device), with watchdog & display-probe
- Clean encoder drain/trailer/close; robust metrics JSONL writer
- Bench targets for raw/mjpeg/h264; consolidated `run-bench-summary`
- Comprehensive README and inline code documentation

**Artifacts**
- `bpi-f3-yolov5n-v0.4.0-linux-riscv64.tar.gz` (binary + runtime notes + ldd deps + README)
- `.sha256` checksum

**Requirements (device)**
- Banana Pi BPI‑F3 (K1X), distro based on Spacemit stack
- FFmpeg, SDL2 and OpenCV (core/imgproc) packages installed on device
- The binary is dynamically linked; see `ldd_yolov5n_pipeline.txt` for exact deps

**Quick Start (file source)**
```bash
./yolov5n_pipeline --src file:/data/bpi-f3-yolov5n/input_video.mp4 \
  --out out.mp4 --display off --weights cpu_model/hhb.bm \
  --nn-cpus auto --io-cpus auto --perf-json metrics.jsonl
````

**Live Camera + Display (YUYV, SW preproc)**

```bash
./yolov5n_pipeline --src 'v4l2:auto?fmt=yuyv' \
  --display sdl --sdl-driver wayland \
  --weights cpu_model/hhb.bm --nn-cpus auto --io-cpus auto --perf-json metrics_cam.jsonl
```

**Known limitations**

* Performance still limited by CPU inference (no hardware NPU offload in this build).
* SDL runs in Wayland on device; X11 may fall back to software.

**Changelog**

* Add V4L2 capture (YUYV), auto-probe & robust stream loop
* Add SDL2 display path with watchdog and frame probe
* Finalize FFmpeg encoder (safe drain & trailer)
* Metrics JSONL and console metrics stabilized
* Add `run-bench-summary` and docs coverage report
