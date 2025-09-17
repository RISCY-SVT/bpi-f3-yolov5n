#include "types.hpp"
#include "capture.hpp"
#include "preprocess.hpp"
#include "engine.hpp"
#include "pipeline.hpp"
#include "metrics.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <csignal>
#include <cstdlib>
#include <limits>

/**
 * @file main.cpp
 * @brief CLI entry point configuring and launching the YOLOv5n pipeline.
 */

using namespace yolov5;

// Forward declaration for overlay helper
namespace yolov5 { void draw_detections(cv::Mat& frame_bgr, const std::vector<Detection>& dets); }

static volatile bool g_stop = false;

/** @brief Catch termination signals and request pipeline shutdown. */
void signal_handler(int sig) {
    g_stop = true;
    std::cout << "\nReceived signal " << sig << ", stopping..." << std::endl;
}

/** @brief Print CLI usage with supported flags and defaults. */
void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [options]\n"
              << "\nInput/Output:\n"
              << "  --src file:path|v4l2:/dev/videoX|v4l2:auto  Input source (required)\n"
              << "  --out path                        Output video file (optional)\n"
              << "  --enc h264|mjpeg|raw              Encoder (default: h264)\n"
              << "  --display off|sdl                 Display mode (default: off)\n"
              << "  --sdl-driver auto|wayland|kmsdrm|x11|dummy\n"
              << "  --cam-fmt auto|yuyv|mjpeg         Preferred V4L2 pixel format (default: auto)\n"
              << "\nModel:\n"
              << "  --weights path                    Path to hhb.bm (default: cpu_model/hhb.bm)\n"
              << "  --imgsz 640x384                  Model image size (fixed)\n"
              << "  --pp sw|rvv                      Preprocess backend (default: sw)\n"
              << "  --rvv on|off                     [Compat] Map to --pp (on=>rvv)\n"
              << "  --conf threshold                  Confidence threshold (default: 0.25)\n"
              << "  --nms threshold                   NMS IOU threshold (default: 0.45)\n"
              << "\nThreading:\n"
              << "  --nn-workers N                    Number of inference workers (default: 4)\n"
              << "  --nn-cpus auto|0,1,2,3            CPU cores for NN (default: auto)\n"
              << "  --io-cpus auto|4,5,6,7            CPU cores for I/O (default: auto)\n"
              << "\nPerformance:\n"
              << "  --queue-cap N                     Queue capacity (default: 8)\n"
              << "  --drop front:wm=N|new:wm=N        Drop policy (default: front:wm=3)\n"
              << "  --perf-interval ms                Performance interval (default: 1000)\n"
              << "  --perf-json path                  JSONL metrics output\n"
              << "  --rt on|off                       Real-time priority (default: off)\n"
              << "  --display-probe path.ppm          Save first displayed frame to file\n"
              << "  --watchdog-sec SEC                Abort if no progress for SEC seconds (0=off)\n"
              << "  --test                           Run simple single-threaded test\n"
              << "\nOther:\n"
              << "  --log-level info|debug|warn|error Log level (default: info)\n"
              << "  --help                            Show this help\n";
}

extern "C" {
#include <libavutil/log.h>
}

namespace {

constexpr int kCliUserError = 2;

/** @brief Print error, usage, and exit with CLI failure code. */
[[noreturn]] void cli_error(const char* program, const std::string& opt, const std::string& message) {
    if (!message.empty()) {
        std::cerr << message << std::endl;
    }
    print_usage(program);
    std::exit(kCliUserError);
}

/** @brief Parse integer argument with bounds checking. */
int parse_int_option(const char* program, const std::string& opt, const std::string& value,
                     int min_value, int max_value) {
    if (value.empty()) {
        cli_error(program, opt, "Missing value for " + opt);
    }
    try {
        size_t idx = 0;
        long long parsed = std::stoll(value, &idx, 10);
        if (idx != value.size()) {
            cli_error(program, opt, "Invalid integer for " + opt + ": " + value);
        }
        if (parsed < static_cast<long long>(min_value) || parsed > static_cast<long long>(max_value)) {
            std::ostringstream oss;
            oss << "Value for " << opt << " must be between " << min_value << " and " << max_value;
            cli_error(program, opt, oss.str());
        }
        return static_cast<int>(parsed);
    } catch (const std::exception&) {
        cli_error(program, opt, "Invalid integer for " + opt + ": " + value);
    }
    return min_value; // Unreachable, keeps compiler happy
}

/** @brief Parse float argument with bounds checking. */
float parse_float_option(const char* program, const std::string& opt, const std::string& value,
                         float min_value, float max_value) {
    if (value.empty()) {
        cli_error(program, opt, "Missing value for " + opt);
    }
    try {
        size_t idx = 0;
        float parsed = std::stof(value, &idx);
        if (idx != value.size()) {
            cli_error(program, opt, "Invalid float for " + opt + ": " + value);
        }
        if (parsed < min_value || parsed > max_value) {
            std::ostringstream oss;
            oss << "Value for " << opt << " must be between " << min_value << " and " << max_value;
            cli_error(program, opt, oss.str());
        }
        return parsed;
    } catch (const std::exception&) {
        cli_error(program, opt, "Invalid float for " + opt + ": " + value);
    }
    return min_value; // Unreachable, keeps compiler happy
}

/** @brief Fetch next CLI token, erroring if absent. */
std::string require_value(int& index, int argc, char* argv[], const std::string& opt, const char* program) {
    if (index + 1 >= argc) {
        cli_error(program, opt, "Missing value for " + opt);
    }
    return argv[++index];
}

} // namespace

/**
 * @brief Parse CPU affinity list (comma-separated) or "auto" sentinel.
 */
std::vector<int> parse_cpu_list(const std::string& str, const char* opt, const char* program) {
    std::vector<int> cpus;
    if (str == "auto") {
        return cpus;  // Empty means auto-detect
    }
    
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        cpus.push_back(parse_int_option(program, opt, token, 0, 1023));
    }
    return cpus;
}

/** @brief Lowercase helper preserving original string immutability. */
static std::string to_lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return s;
}

/**
 * @brief Parse CLI arguments and populate PipelineConfig struct.
 * @throws Exits process via cli_error on invalid input.
 */
PipelineConfig parse_args(int argc, char* argv[]) {
    PipelineConfig config;
    bool test_mode = false;
    const char* program = argv[0];
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            exit(0);
        }

        if (arg == "--src") {
            config.source = require_value(i, argc, argv, arg, program);
            // Treat bare path as file:/path to match CLI contract
            if (config.source.find(":") == std::string::npos && config.source.rfind("/dev/video", 0) != 0) {
                config.source = std::string("file:") + config.source;
            }
        } else if (arg == "--out") {
            config.output_path = require_value(i, argc, argv, arg, program);
        } else if (arg == "--enc") {
            config.encoder = require_value(i, argc, argv, arg, program);
        } else if (arg == "--display") {
            std::string v = to_lower_copy(require_value(i, argc, argv, arg, program));
            if (v != "off" && v != "sdl") {
                cli_error(program, arg, "Invalid --display value: " + v + " (use off|sdl)");
            }
            config.display_mode = v;
        } else if (arg == "--sdl-driver") {
            std::string v = to_lower_copy(require_value(i, argc, argv, arg, program));
            if (v != "auto" && v != "wayland" && v != "kmsdrm" && v != "x11" && v != "dummy") {
                cli_error(program, arg, "Invalid --sdl-driver value: " + v);
            }
            config.sdl_driver = v;
        } else if (arg == "--cam-fmt") {
            std::string v = to_lower_copy(require_value(i, argc, argv, arg, program));
            if (v != "auto" && v != "yuyv" && v != "mjpeg") {
                cli_error(program, arg, "Invalid --cam-fmt value: " + v + " (use auto|yuyv|mjpeg)");
            }
            config.cam_format = v;
        } else if (arg == "--weights") {
            config.weights_path = require_value(i, argc, argv, arg, program);
        } else if (arg == "--imgsz") {
            std::string val = require_value(i, argc, argv, arg, program);
            auto x = val.find('x');
            if (x == std::string::npos) {
                cli_error(program, arg, "Invalid --imgsz format, expected WxH (e.g., 640x384)");
            }
            int w = parse_int_option(program, arg, val.substr(0, x), 1, std::numeric_limits<int>::max());
            int h = parse_int_option(program, arg, val.substr(x + 1), 1, std::numeric_limits<int>::max());
            if (w != MODEL_WIDTH || h != MODEL_HEIGHT) {
                cli_error(program, arg, "--imgsz must be exactly 640x384 for this model");
            }
            config.img_width = w;
            config.img_height = h;
        } else if (arg == "--pp") {
            std::string v = require_value(i, argc, argv, arg, program);
            if (v == "sw") config.pp_mode = PreprocMode::SW;
            else if (v == "rvv") config.pp_mode = PreprocMode::RVV;
            else { cli_error(program, arg, "Invalid --pp value: " + v + " (use sw|rvv)"); }
        } else if (arg == "--rvv") {
            std::string v = require_value(i, argc, argv, arg, program);
            if (v != "on" && v != "off") {
                cli_error(program, arg, "--rvv must be 'on' or 'off'");
            }
            config.rvv = (v == "on");
            // Map compatibility flag to pp_mode
            if (config.rvv) config.pp_mode = PreprocMode::RVV; else config.pp_mode = PreprocMode::SW;
        } else if (arg == "--conf") {
            config.conf_threshold = parse_float_option(program, arg, require_value(i, argc, argv, arg, program), 0.0f, 1.0f);
        } else if (arg == "--nms") {
            config.nms_threshold = parse_float_option(program, arg, require_value(i, argc, argv, arg, program), 0.0f, 1.0f);
        } else if (arg == "--nn-workers") {
            config.nn_workers = parse_int_option(program, arg, require_value(i, argc, argv, arg, program), 1, 32);
        } else if (arg == "--nn-cpus") {
            std::string val = require_value(i, argc, argv, arg, program);
            if (val == "auto") {
                config.auto_cpu_detect = true;
                config.nn_cpus.clear();
            } else {
                config.auto_cpu_detect = false;
                config.nn_cpus = parse_cpu_list(val, "--nn-cpus", program);
            }
        } else if (arg == "--io-cpus") {
            std::string val = require_value(i, argc, argv, arg, program);
            if (val != "auto") {
                config.io_cpus = parse_cpu_list(val, "--io-cpus", program);
            }
        } else if (arg == "--queue-cap") {
            config.queue_capacity = parse_int_option(program, arg, require_value(i, argc, argv, arg, program), 1, 1024);
        } else if (arg == "--drop") {
            config.drop_policy = require_value(i, argc, argv, arg, program);
            // Parse watermark
            size_t pos = config.drop_policy.find("wm=");
            if (pos != std::string::npos) {
                config.drop_watermark = parse_int_option(program, arg, config.drop_policy.substr(pos + 3), 0, 1024);
            }
        } else if (arg == "--perf-interval") {
            config.perf_interval_ms = parse_int_option(program, arg, require_value(i, argc, argv, arg, program), 10, 600000);
        } else if (arg == "--perf-json") {
            config.perf_json_path = require_value(i, argc, argv, arg, program);
        } else if (arg == "--rt") {
            std::string val = to_lower_copy(require_value(i, argc, argv, arg, program));
            if (val != "on" && val != "off") {
                cli_error(program, arg, "--rt must be on or off");
            }
            config.use_rt_priority = (val == "on");
        } else if (arg == "--display-probe") {
            config.display_probe_path = require_value(i, argc, argv, arg, program);
            if (config.display_probe_path.empty()) {
                cli_error(program, arg, "--display-probe requires a non-empty path");
            }
        } else if (arg == "--watchdog-sec") {
            config.watchdog_sec = parse_int_option(program, arg, require_value(i, argc, argv, arg, program), 0, 3600);
        } else if (arg == "--test") {
            test_mode = true;
        } else if (arg == "--max-frames") {
            config.max_frames = parse_int_option(program, arg, require_value(i, argc, argv, arg, program), 0, std::numeric_limits<int>::max());
        } else if (arg == "--log-level") {
            config.log_level = require_value(i, argc, argv, arg, program);
        } else {
            cli_error(program, arg, "Unknown argument: " + arg);
        }
    }
    
    // Validate required arguments
    if (config.source.empty()) {
        cli_error(program, "--src", "Error: --src is required");
    }

    if (config.source.rfind("v4l2:", 0) == 0 && !config.cam_format.empty() && config.cam_format != "auto") {
        if (config.source.find('?') == std::string::npos) {
            config.source += "?fmt=" + config.cam_format;
        } else {
            config.source += "&fmt=" + config.cam_format;
        }
    }

#ifndef HAVE_SDL2
    if (config.display_mode == "sdl") {
        std::cerr << "[display] SDL support not built into this binary; forcing --display=off" << std::endl;
        config.display_mode = "off";
    }
#endif

    if (test_mode) {
        // Encode test mode selection via log_level flag to avoid changing struct
        if (config.log_level.empty()) config.log_level = "info";
        // Use a side-channel env for main(); kept simple
        setenv("Y5_TEST_MODE", "1", 1);
    }
    return config;
}

/**
 * @brief Lightweight single-threaded test harness for sanity checks.
 *
 * Exercises capture → preprocess → engine without full pipeline fan-out.
 */
void run_simple_test(const PipelineConfig& config) {
    std::cout << "Running simple test mode (single-threaded)..." << std::endl;
    
    // Create capture
    auto capture = createCapture(config.source);
    if (!capture || !capture->init(config.source)) {
        std::cerr << "Failed to initialize capture from: " << config.source << std::endl;
        return;
    }
    
    std::cout << "Video: " << capture->getWidth() << "x" << capture->getHeight() 
              << " @ " << capture->getFPS() << " FPS" << std::endl;
    
    // Create engine
    auto engine = createEngine("csi");
    if (!engine || !engine->init(config.weights_path)) {
        std::cerr << "Failed to initialize engine with weights: " << config.weights_path << std::endl;
        return;
    }
    
    // Create preprocessor
    Preprocessor preprocessor;
    
    // Allocate model input buffer (FP32 for now, will be converted to FP16 inside engine)
    const size_t input_size = MODEL_CHANNELS * MODEL_HEIGHT * MODEL_WIDTH * sizeof(float);
    void* model_input = utils::alignedAlloc(input_size, 64);
    
    // Metrics writer (optional)
    std::unique_ptr<JSONLMetricsWriter> metrics_writer;
    if (!config.perf_json_path.empty()) {
        metrics_writer = std::make_unique<JSONLMetricsWriter>(config.perf_json_path);
    }

    // Process frames
    int frame_count = 0;
    auto start_time = std::chrono::steady_clock::now();
    auto interval_start = start_time;

    // Per-interval accumulators
    std::vector<double> cap_ms, pp_ms, inf_ms;
    
    while (!g_stop) {
        Frame frame;
        auto cap_t0 = std::chrono::steady_clock::now();
        if (!capture->getFrame(frame)) {
            std::cout << "No more frames" << std::endl;
            break;
        }
        auto cap_t1 = std::chrono::steady_clock::now();
        
        // Preprocess
        float scale;
        int dx, dy;
        auto pp_t0 = std::chrono::steady_clock::now();
        preprocessor.preprocess(frame, model_input, scale, dx, dy);
        auto pp_t1 = std::chrono::steady_clock::now();
        
        // Run inference
        auto inf_start = std::chrono::steady_clock::now();
        std::vector<Detection> detections = engine->infer(model_input);
        auto inf_end = std::chrono::steady_clock::now();
        
        // Rescale detections to original coordinates
        for (auto& det : detections) {
            det.x1 = (det.x1 - dx) / scale;
            det.y1 = (det.y1 - dy) / scale;
            det.x2 = (det.x2 - dx) / scale;
            det.y2 = (det.y2 - dy) / scale;
        }

        // Overlay detections onto frame image
        yolov5::draw_detections(frame.image, detections);
        
        // Print results
        auto cap_dur = std::chrono::duration_cast<std::chrono::microseconds>(cap_t1 - cap_t0).count() / 1000.0;
        auto pp_dur  = std::chrono::duration_cast<std::chrono::microseconds>(pp_t1 - pp_t0).count() / 1000.0;
        auto inf_dur = std::chrono::duration_cast<std::chrono::microseconds>(inf_end - inf_start).count() / 1000.0;
        cap_ms.push_back(cap_dur);
        pp_ms.push_back(pp_dur);
        inf_ms.push_back(inf_dur);
        auto inf_ms_disp = static_cast<long long>(inf_dur);
        std::cout << "Frame " << frame.frame_id << ": " 
                  << detections.size() << " detections, "
                  << "inference: " << inf_ms_disp << "ms" << std::endl;
        
        for (const auto& det : detections) {
            std::cout << "  [" << (int)det.x1 << "," << (int)det.y1 << ","
                      << (int)det.x2 << "," << (int)det.y2 << "] "
                      << det.label << ": " << std::fixed << std::setprecision(2) 
                      << det.confidence << std::endl;
        }
        
        frame_count++;
        
        // Periodic metrics
        auto now = std::chrono::steady_clock::now();
        auto interval_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - interval_start).count();
        if (interval_ms >= config.perf_interval_ms) {
            // Helper to compute percentiles
            auto percentile = [](std::vector<double>& v, double p) -> double {
                if (v.empty()) return 0.0;
                std::sort(v.begin(), v.end());
                size_t idx = static_cast<size_t>(std::clamp(p, 0.0, 1.0) * (v.size() - 1));
                return v[idx];
            };

            PerfMetrics m{};
            m.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::system_clock::now().time_since_epoch()).count();
            m.input_fps = (frame_count * 1000.0) / std::max<int64_t>(interval_ms, 1);
            m.output_fps = m.input_fps;
            m.drop_percentage = 0.0f;
            m.latency_ms.capture = static_cast<float>(cap_ms.empty() ? 0.0 : percentile(cap_ms, 0.5));
            m.latency_ms.preprocess = static_cast<float>(pp_ms.empty() ? 0.0 : percentile(pp_ms, 0.5));
            m.latency_ms.inference_p50 = static_cast<float>(percentile(inf_ms, 0.5));
            m.latency_ms.inference_p95 = static_cast<float>(percentile(inf_ms, 0.95));
            m.latency_ms.postprocess = 0.0f;
            m.latency_ms.overlay = 0.0f;
            m.latency_ms.encode = 0.0f;
            m.queue_sizes = {
                {"cap_pp", 0}, {"pp_sched", 0}, {"sched_inf", 0}, {"inf_post", 0}, {"post_reord", 0}
            };
            m.worker_busy_pct.assign(std::max(1, config.nn_workers), 0.0f);

            if (metrics_writer) metrics_writer->write(m);

            // Log human-readable summary too
            std::cout << std::fixed << std::setprecision(2)
                      << "[metrics] in_fps=" << m.input_fps
                      << " out_fps=" << m.output_fps
                      << " inf_p50=" << m.latency_ms.inference_p50 << "ms"
                      << " inf_p95=" << m.latency_ms.inference_p95 << "ms" << std::endl;

            // Reset interval accumulators
            frame_count = 0;
            cap_ms.clear(); pp_ms.clear(); inf_ms.clear();
            interval_start = now;
        }
        
        // Limit to 100 frames for testing
        if (frame_count >= 100) {
            std::cout << "Processed 100 frames, stopping test" << std::endl;
            break;
        }
    }
    
    // Cleanup
    utils::alignedFree(model_input);
    
    // Final statistics
    auto end_time = std::chrono::steady_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    float avg_fps = frame_count * 1000.0f / total_ms;
    
    std::cout << "\nTest complete:\n"
              << "  Frames processed: " << frame_count << "\n"
              << "  Total time: " << total_ms << "ms\n"
              << "  Average FPS: " << std::fixed << std::setprecision(2) << avg_fps << std::endl;
}

/**
 * @brief Program entry: parse CLI, optionally run test, launch full pipeline.
 */
int main(int argc, char* argv[]) {
    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGHUP, signal_handler); // handle SSH session termination gracefully
    // Reduce FFmpeg log verbosity by default
    av_log_set_level(AV_LOG_ERROR);
    
    // Parse command line arguments
    PipelineConfig config = parse_args(argc, argv);
    
    std::cout << "YOLOv5n Video Pipeline for Banana Pi BPI-F3\n"
              << "============================================\n"
              << "Source: " << config.source << "\n"
              << "Weights: " << config.weights_path << "\n"
              << "Confidence: " << config.conf_threshold << "\n"
              << "NMS: " << config.nms_threshold << "\n";
    
    if (config.auto_cpu_detect) {
        std::cout << "CPU affinity: auto-detect\n";
    } else {
        std::cout << "NN CPUs: ";
        for (int cpu : config.nn_cpus) std::cout << cpu << " ";
        std::cout << "\nIO CPUs: ";
        for (int cpu : config.io_cpus) std::cout << cpu << " ";
        std::cout << "\n";
    }
    
    std::cout << "============================================\n" << std::endl;
    
    const char* t = getenv("Y5_TEST_MODE");
    if (t && std::string(t) == "1") {
        run_simple_test(config);
        return 0;
    }

    // Full threaded pipeline
    Pipeline pipeline(config);
    if (!pipeline.start()) {
        std::cerr << "[ERROR] Failed to start pipeline" << std::endl;
        return 1;
    }
    auto wall_t0 = std::chrono::steady_clock::now();
    while (pipeline.isRunning() && !g_stop) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    pipeline.stop();
    pipeline.join();
    auto wall_t1 = std::chrono::steady_clock::now();
    auto wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(wall_t1 - wall_t0).count();
    auto m = pipeline.getMetrics();
    const uint64_t frames_in = pipeline.inCount();
    const uint64_t frames_out = pipeline.processedCount();
    const uint64_t frames_drop = (frames_in >= frames_out) ? (frames_in - frames_out) : 0;
    double avg_fps = (wall_ms > 0) ? (frames_out * 1000.0 / wall_ms) : 0.0;
    std::cout << "\nDone." << std::endl;
    std::cout << "Frames in: " << frames_in << ", processed: " << frames_out << ", dropped: " << frames_drop << std::endl;
    std::cout << "Average FPS: " << std::fixed << std::setprecision(2) << avg_fps << std::endl;
    
    return 0;
}
