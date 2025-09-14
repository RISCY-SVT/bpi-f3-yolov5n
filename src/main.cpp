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
#include <cstring>
#include <csignal>

using namespace yolov5;

// Forward declaration for overlay helper
namespace yolov5 { void draw_detections(cv::Mat& frame_bgr, const std::vector<Detection>& dets); }

static volatile bool g_stop = false;

void signal_handler(int sig) {
    g_stop = true;
    std::cout << "\nReceived signal " << sig << ", stopping..." << std::endl;
}

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [options]\n"
              << "\nInput/Output:\n"
              << "  --src file:path|v4l2:/dev/videoX  Input source (required)\n"
              << "  --out path                        Output video file (optional)\n"
              << "  --enc h264|mjpeg|raw              Encoder (default: h264)\n"
              << "  --display auto|sdl|off            Display mode (default: auto)\n"
              << "\nModel:\n"
              << "  --weights path                    Path to hhb.bm (default: cpu_model/hhb.bm)\n"
              << "  --imgsz 640x384                  Model image size (fixed)\n"
              << "  --rvv on|off                     Enable RVV preprocessing path (default: off)\n"
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
              << "  --test                           Run simple single-threaded test\n"
              << "\nOther:\n"
              << "  --log-level info|debug|warn|error Log level (default: info)\n"
              << "  --help                            Show this help\n";
}

std::vector<int> parse_cpu_list(const std::string& str) {
    std::vector<int> cpus;
    if (str == "auto") {
        return cpus;  // Empty means auto-detect
    }
    
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try {
            cpus.push_back(std::stoi(token));
        } catch (...) {
            std::cerr << "Invalid CPU number: " << token << std::endl;
        }
    }
    return cpus;
}

PipelineConfig parse_args(int argc, char* argv[]) {
    PipelineConfig config;
    bool test_mode = false;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            exit(0);
        }
        
        // Get next argument value
        auto get_value = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << arg << std::endl;
                exit(1);
            }
            return argv[++i];
        };
        
        if (arg == "--src") {
            config.source = get_value();
            // Treat bare path as file:/path to match CLI contract
            if (config.source.find(":") == std::string::npos && config.source.rfind("/dev/video", 0) != 0) {
                config.source = std::string("file:") + config.source;
            }
        } else if (arg == "--out") {
            config.output_path = get_value();
        } else if (arg == "--enc") {
            config.encoder = get_value();
        } else if (arg == "--display") {
            config.display_mode = get_value();
        } else if (arg == "--weights") {
            config.weights_path = get_value();
        } else if (arg == "--imgsz") {
            std::string val = get_value();
            auto x = val.find('x');
            if (x == std::string::npos) {
                std::cerr << "Invalid --imgsz format, expected WxH (e.g., 640x384)" << std::endl;
                exit(1);
            }
            int w = std::stoi(val.substr(0, x));
            int h = std::stoi(val.substr(x + 1));
            if (w != MODEL_WIDTH || h != MODEL_HEIGHT) {
                std::cerr << "--imgsz must be exactly 640x384 for this model" << std::endl;
                exit(1);
            }
            config.img_width = w;
            config.img_height = h;
        } else if (arg == "--rvv") {
            std::string v = get_value();
            if (v != "on" && v != "off") {
                std::cerr << "--rvv must be 'on' or 'off'" << std::endl;
                exit(1);
            }
            config.rvv = (v == "on");
        } else if (arg == "--conf") {
            config.conf_threshold = std::stof(get_value());
        } else if (arg == "--nms") {
            config.nms_threshold = std::stof(get_value());
        } else if (arg == "--nn-workers") {
            config.nn_workers = std::stoi(get_value());
        } else if (arg == "--nn-cpus") {
            std::string val = get_value();
            if (val == "auto") {
                config.auto_cpu_detect = true;
                config.nn_cpus.clear();
            } else {
                config.auto_cpu_detect = false;
                config.nn_cpus = parse_cpu_list(val);
            }
        } else if (arg == "--io-cpus") {
            std::string val = get_value();
            if (val != "auto") {
                config.io_cpus = parse_cpu_list(val);
            }
        } else if (arg == "--queue-cap") {
            config.queue_capacity = std::stoi(get_value());
        } else if (arg == "--drop") {
            config.drop_policy = get_value();
            // Parse watermark
            size_t pos = config.drop_policy.find("wm=");
            if (pos != std::string::npos) {
                config.drop_watermark = std::stoi(config.drop_policy.substr(pos + 3));
            }
        } else if (arg == "--perf-interval") {
            config.perf_interval_ms = std::stoi(get_value());
        } else if (arg == "--perf-json") {
            config.perf_json_path = get_value();
        } else if (arg == "--rt") {
            config.use_rt_priority = (get_value() == "on");
        } else if (arg == "--test") {
            test_mode = true;
        } else if (arg == "--max-frames") {
            config.max_frames = std::stoi(get_value());
        } else if (arg == "--log-level") {
            config.log_level = get_value();
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            exit(1);
        }
    }
    
    // Validate required arguments
    if (config.source.empty()) {
        std::cerr << "Error: --src is required\n";
        print_usage(argv[0]);
        exit(1);
    }
    
    if (test_mode) {
        // Encode test mode selection via log_level flag to avoid changing struct
        if (config.log_level.empty()) config.log_level = "info";
        // Use a side-channel env for main(); kept simple
        setenv("Y5_TEST_MODE", "1", 1);
    }
    return config;
}

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

int main(int argc, char* argv[]) {
    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGHUP, signal_handler); // handle SSH session termination gracefully
    
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
