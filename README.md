Build: cross-compile (Spacemit), Input: FFmpeg/V4L2, Display: SDL2 (Wayland/headless)

# 1. Project Overview
The YOLOv5n video pipeline ingests media from files or USB cameras, preprocesses frames to the model’s fixed 1×3×384×640 FP16 NCHW layout, executes CSI-NN2 inference on the HHB-compiled network, and overlays detections before streaming results to disk and/or a display. The data path is:

```
source capture → preprocess → scheduler → N× inference workers → postprocess/NMS → reorderer → overlay → output (encoder + display)
```

Key capabilities:
- FFmpeg-based file ingestion and encoding (H.264, MJPEG, raw BGR).
- V4L2 capture supporting YUYV422 and MJPEG on Banana Pi BPI-F3.
- SDL2 display with watchdog, auto driver selection, and optional probe snapshot.
- JSONL performance metrics compatible with `--perf-json`/`--perf-interval`.
- Thread-aware CPU affinity and queue management with drop policies for overload.

# 2. Hardware & OS
- Target: Banana Pi BPI-F3 (SpacemiT K1X) running the vendor Debian-based image.
- Display: Wayland/KMS SDL2 window when available; gracefully falls back to headless.
- USB camera: auto-probes `/dev/video*` (prefers `/dev/v4l/by-id` groups) and can be forced via `--src v4l2:/dev/videoX` or `--cam-fmt fmt=yuyv|mjpeg`.

# 3. Toolchains & Dependencies
- Cross toolchain lives under `/opt/spacemit`; sysroot staged at `/opt/spacemit/sysroot`.
- Bootstrap all third-party libraries via the idempotent script:
  ```bash
  SSH_TARGET=banana bash ./install_all_libs_to_spacemit.sh
  ```
  It copies FFmpeg, SDL2, OpenCV, V4L2, codecs, Wayland/X11 stacks, etc. from the device.
- `env.sh` exports `PKG_CONFIG_LIBDIR` and `PKG_CONFIG_SYSROOT_DIR` so `pkg-config` resolves against the sysroot. Always rely on the Makefile; no ad-hoc build scripts.
- SDL2 support is optional; enable via `ENABLE_SDL=1` when `pkg-config sdl2` works.

# 4. Build Instructions
1. Configure the environment:
   ```bash
   cd /data/projects/bpi-f3-yolov5n
   . ./env.sh
   ```
2. Build the pipeline (defaults to GCC; set `COMPILER=clang` if the toolchain provides it):
   ```bash
   make -j"$(nproc)" pipeline ENABLE_SDL=1
   ```
3. Typical verification steps:
   ```bash
   pkg-config --cflags --libs sdl2 libavformat libavcodec libavutil libswscale opencv4
   ```
   Resolve missing packages by extending `install_all_libs_to_spacemit.sh` and rerunning it (do not manually copy libraries).

# 5. Deploy & Run (Device)
1. Deploy binaries and supporting assets to the Banana Pi:
   ```bash
   make deploy  # uses SSH target from env.sh / ~/.ssh/config
   ```
2. File playback with recording and metrics (runs on device):
   ```bash
   ./yolov5n_pipeline \
     --src file:/data/bpi-f3-yolov5n/input.mp4 \
     --weights cpu_model/hhb.bm \
     --enc h264 --out /data/bpi-f3-yolov5n/out.mp4 \
     --display off \
     --perf-json /data/bpi-f3-yolov5n/metrics_file.jsonl \
     --perf-interval 1000 --max-frames 600
   ```
3. USB camera (YUYV) with SDL display and software preprocess:
   ```bash
   ./yolov5n_pipeline \
     --src v4l2:auto?fmt=yuyv \
     --weights cpu_model/hhb.bm \
     --display sdl --sdl-driver wayland \
     --pp sw --out /data/bpi-f3-yolov5n/cam_sw.avi \
     --perf-json /data/bpi-f3-yolov5n/metrics_cam_sw.jsonl
   ```
4. Toggle RVV kernels for preprocessing and YUV conversion by adding `--pp rvv` or `--rvv on`.
5. Affinity examples:
   ```bash
   # Pin inference to big cores 4-7 and preprocessing to little cores 0-3
   ./yolov5n_pipeline --nn-cpus 4,5,6,7 --io-cpus 0,1,2,3 ...

   # Let auto-detect micro-benchmark choose the faster cluster
   ./yolov5n_pipeline --nn-cpus auto --nn-workers auto ...
   ```
6. Display utilities:
   - Force driver: `--sdl-driver kmsdrm` (supports Wayland/KMS/X11/dummy).
   - Save the first presented frame: `--display-probe /data/bpi-f3-yolov5n/probes/first.ppm`.
   - Watchdog: `--watchdog-sec 10` aborts output thread if presents stall.
7. Bench helpers (run on device):
   ```bash
   make run-bench-summary -- pp=sw
   make run-bench-summary -- pp=rvv
   ```
   These execute the standard file + camera scenarios and collect metrics/artifacts under `artifacts/`.

# 6. CLI Reference
| Flag | Default | Description |
| --- | --- | --- |
| `--src` | _required_ | Input source. `file:/path`, bare path (treated as file), or `v4l2:/dev/videoX`, `v4l2:auto`. |
| `--out` | _empty_ | Output container written via FFmpeg (`.mp4` for H.264, `.avi` for MJPEG/raw). |
| `--enc` | `h264` | Encoder: `h264`, `mjpeg`, or `raw` (BGR24 AVI). |
| `--display` | `off` | `off` or `sdl`. `auto` maps to SDL when available. |
| `--sdl-driver` | `auto` | SDL driver hint (`wayland`, `kmsdrm`, `x11`, `dummy`). |
| `--cam-fmt` | `auto` | Preferred V4L2 format when probing cameras (`yuyv`, `mjpeg`). |
| `--weights` | `cpu_model/hhb.bm` | Path to HHB binary; compiled into project tree. |
| `--imgsz` | `640x384` | Must remain fixed for current model. Validation enforced. |
| `--pp` | `sw` | Preprocessing backend: `sw` or `rvv`. |
| `--rvv` | `off` | Compatibility toggle mapping to `--pp`. |
| `--conf` | `0.25` | Confidence threshold for detections. |
| `--nms` | `0.45` | IOU threshold for NMS. |
| `--nn-workers` | `4` | Number of inference worker threads (`auto` picks logical cores). |
| `--nn-cpus` | `auto` | CPU list (comma separated) for inference threads; `auto` runs micro-bench to select cluster. |
| `--io-cpus` | `auto` | CPU list for capture/preprocess threads. |
| `--queue-cap` | `8` | Capacity for each stage queue. |
| `--drop` | `front:wm=3` | Queue drop policy when watermark exceeded (`front` / `new`). |
| `--perf-interval` | `1000` | Metrics reporting interval in milliseconds. |
| `--perf-json` | _empty_ | JSONL output path for metrics snapshots. |
| `--rt` | `off` | Enable SCHED_FIFO when permitted; falls back with warning. |
| `--display-probe` | _empty_ | Path to write first displayed frame (PPM). |
| `--watchdog-sec` | `0` | Abort display thread if no presents within N seconds (0 disables). |
| `--max-frames` | `-1` | Stop after N frames (`<=0`: unlimited). |
| `--log-level` | `info` | Logging verbosity (`debug`, `info`, `warn`, `error`). |
| `--test` | _disabled_ | Run the single-threaded functional smoke test instead of full pipeline. |

# 7. Metrics & Profiling
- Metrics stream is JSON Lines with fixed keys: `ts_ms`, `in_fps`, `out_fps`, `drop_pct`, `latency_ms{cap,pp,inf_p50,inf_p95,post,overlay,enc,display}`, `qsize{cap_pp,pp_sched,sched_inf,inf_post,post_reord}`, and `workers_busy_pct[]`.
- Generate metrics by supplying `--perf-json /path/run.jsonl` and set `--perf-interval` as needed.
- Quick inspection:
  ```bash
  tail -n 5 /data/bpi-f3-yolov5n/metrics_file.jsonl
  jq -r '.out_fps' /data/bpi-f3-yolov5n/metrics_cam_sw.jsonl | stats.py
  ```
- Collect artifacts back to host via `scp banana:/data/bpi-f3-yolov5n/*.jsonl ./artifacts/`.

# 8. Known Performance Characteristics
Recent bench runs (Banana Pi BPI-F3, 2024-09 builds, 4 inference workers):

| Scenario | Preprocess | Avg out FPS | `inf_p50` (ms) | Notes |
| --- | --- | --- | --- | --- |
| File playback → raw AVI (`artifacts/run-file-sdl-sw.log`) | SW | ~5.3 | ~720 | Display disabled (null), heavy encoder → queue drop ~30%. |
| V4L2 YUYV → SDL (`artifacts/run-cam-yuyv-sdl-sw.log`) | SW | ~5.4 | ~695 | Capture dominates latency (~80–160 ms); display null when SDL missing. |
| V4L2 YUYV → SDL (`artifacts/run-cam-yuyv-sdl-rvv.log`) | RVV | ~5.4 | ~700 | RVV path reduces preprocessing CPU but inference dominates overall. |

> FFmpeg prints `using cpu capabilities: none!` when libx264 is built without NEON optimizations on this toolchain. This is expected and does not affect functional output.

# 9. Troubleshooting
- **V4L2 camera**
  - Enumerate devices: `v4l2-ctl --list-devices`, `v4l2-ctl -d /dev/video0 --list-formats-ext`.
  - If the probe selects the wrong node, pass `--src v4l2:/dev/video2` and/or `--cam-fmt fmt=mjpeg`.
  - Permission issues: ensure the deploying user is in the `video` group or run via `sudo` on the device.
- **SDL/Wayland**
  - If SDL falls back to `dummy`, confirm `WAYLAND_DISPLAY` or `XDG_RUNTIME_DIR` is present, and run `ENABLE_SDL=1 make pipeline` to ensure the library was linked.
  - Use `--display-probe` to confirm rendered frames even when the window is hidden.
  - Watchdog expirations (`--watchdog-sec`) usually indicate the display thread stalled; inspect GPU/Wayland logs.
- **Sysroot / linking**
  - Re-run `install_all_libs_to_spacemit.sh` whenever headers or `.pc` files are missing.
  - Verify environment with `pkg-config --modversion sdl2` using the exported `PKG_CONFIG_LIBDIR`/`PKG_CONFIG_SYSROOT_DIR`.
- **Performance**
  - High drop percentage: raise `--queue-cap`, reduce `--nn-workers`, or adjust `--drop front:wm=N` to shed earlier.
  - To compare RVV vs SW, re-run `make run-bench-summary -- pp={sw,rvv}` and inspect the generated JSONL under `artifacts/`.

# 10. Repository Layout
- `include/` – Public headers for capture, display, engine, pipeline, metrics, preprocessing, and shared types.
- `src/` – Stage implementations, CLI, SDL renderer, FFmpeg encoder, and helper tools.
- `cpu_model/` – HHB-generated `model.c` and `hhb.bm` artifacts (read-only except integration notes).
- `tests/` – Unit tests for reorderer and NMS (extend as functionality grows).
- `artifacts/` – Sample runs, metrics, and display probes produced by bench targets.
- `tools/` – Utility scripts (e.g., `run_bench_summary.sh`).
- `Work_Logs/` – Session reports produced after each development task.
- `Makefile` – Sole build and deployment entry point.

# 11. License & Acknowledgments
Project-specific licensing is TBD. HHB/CSI-NN2 binaries and third-party libraries remain under their respective licenses; consult vendor documentation when redistributing.
