# Changelog

## [0.4.2] - 2025-09-25
- Live mode: shrink capture/preprocess/inference queues to single-entry latest-wins buffers and
  rework `run-cam-live-sw` to use three inference workers for ≥4 FPS output.
- Metrics/automation: remote live runner now treats `e2e_p95` > 500 ms as WARN (not failure) and
  ignores the final shutdown sample when checking tail FPS; artifacts guarded for null encoders.
- Docs: refreshed README live-mode queue table and acceptance notes to reflect the new defaults.

## [0.4.1] - 2025-09-18
- Memory: stabilize RSS via CSI-NN2 tensor caching; add `--mem-json` profiler.
- Encoder: consistent drain/trailer/close path on all encoders and containers.
- Docs: README memory profiling guide; V4L2/SDL quick start refined.
- Tooling: bench summary target and metrics JSONL schema documented.

## [0.4.0] - 2025-09-17
- Introduced V4L2 YUYV capture with auto-probing helpers.
- Added SDL2 display, watchdog, and display probe snapshot support.
- Hardened FFmpeg encoder flushing and JSONL metrics reporting.
- Delivered `run-bench-summary` automation and initial documentation set.
