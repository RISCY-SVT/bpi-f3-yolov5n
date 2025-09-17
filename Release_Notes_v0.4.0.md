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
