# BPI‑F3 YOLOv5n — Release v0.4.2 (Low‑Latency Live Mode)

**Date:** 2025‑09‑25  
**Target:** Banana Pi BPI‑F3 (Spacemit K1X, RVV‑1.0)  
**Binary:** `build/yolov5n_pipeline`

## Highlights
- **Live mode / Low‑latency path**: latest‑wins очереди, TTL/gap‑skip в реордерере, отключение VSYNC в SDL; «живое» окно на Wayland.
- **Ring logger** (stage‑level трассировка): JSONL‑дампы ключевых событий пайплайна (capture/pp/infer/post/overlay/encode/display).
- **zerolatency encoder**: H.264 «ultrafast + zerolatency» для on‑screen сценариев.
- **Готовые цели запуска**: `make run-cam-live-sw`, `make run-cam-live-rvv` (и debug‑вариант) выводят видео в окно.  
⚠️ **Live‑режим не пишет исходящее видео в файл по умолчанию** — чтобы избежать «раздувания» диска и лишней задержки.

## What’s new
- Очереди в live‑режиме сведены к **latest‑wins** (длина 1) от capture до postprocess; реордерер удерживает монотонность `frame_id` и TTL‑сброс «залежавшихся» кадров.
- **SDL (Wayland)**: принудительно выключен VSYNC; предусмотрен **probe‑fallback** при недоступном рендерере.
- **Ring logger**: по сигналу watchdog/stop делает дамп последовательности стадий в `artifacts/ring_cam_live_*.jsonl`.
- Метрики (консоль и JSONL): добавлены `e2e_ms`, алиасы очередей (`q_cap/q_pp/q_inf/q_post/q_ord`), счётчики дропов и gap‑skip.

## Quick start (device)
```bash
# 1) Deploy (из хоста)
. ./env.sh && make -s deploy

# 2) Live SW preprocess (быстрый путь для проверки окна)
ssh banana "cd /data/bpi-f3-yolov5n && ./run_pipe.sh run-cam-live-sw"

# 3) Live RVV preprocess (обычный путь)
ssh banana "cd /data/bpi-f3-yolov5n && ./run_pipe.sh run-cam-live-rvv"

## Typical results (BPI‑F3, USB cam 1280×720 YUYV)

* **Output FPS (median)**: 4–6 FPS (SW/RVV).
* **End‑to‑end latency (median)**: ~1.5–2.0 s on‑screen (зависит от камеры/USB, фоновой нагрузки и масштабирования).
* **Drop policy**: допускается пропуск кадров (latest‑wins), порядок на выходе монотонный по `frame_id`.

## Behavior changes

* Live‑цели **не создают видеофайлы** (только окно + метрики + probe PPM).
  Для записи используйте off‑screen/benchmark‑цели.
* Скрипты развёртывания и live‑запуска обновлены; см. `tools/run_live_remote*.sh`, `run_pipe.sh`.

## Known issues

* При отсутствии Wayland/рендерера SDL может деградировать в «null» — окно не появится, но пайплайн и метрики будут валидны.
* USB‑камеры могут иметь буферизацию/агрессивный авто‑экспо‑профиль ⇒ дополнительные сотни миллисекунд задержки.
  Рекомендуем уменьшать `VIDIOC_REQBUFS` (live‑путь делает это автоматически) и избегать дополнительных фильтров.
