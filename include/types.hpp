#ifndef TYPES_HPP
#define TYPES_HPP

#include <opencv2/core.hpp>
#include <chrono>
#include <vector>
#include <string>
#include <map>

namespace yolov5 {

/**
 * @file types.hpp
 * @brief Common enums, structs, and constants shared across modules.
 */

/**
 * @brief Preprocessing backend selection.
 */
enum class PreprocMode {
    SW,   //!< FFmpeg/swscale + scalar chain.
    RVV   //!< RVV kernels chain.
};

/**
 * @brief Pixel formats transported by Frame.
 */
enum class PixelFormat {
    BGR,
    YUV420P
};

/**
 * @brief Frame data structure flowing through pipeline stages.
 */
struct Frame {
    uint64_t frame_id;                               //!< Monotonic frame identifier.
    cv::Mat image;                                   //!< BGR payload when available.
    std::chrono::steady_clock::time_point timestamp; //!< Capture timestamp.
    void* model_input = nullptr;                     //!< Preprocessed tensor buffer.
    bool eos = false;                                //!< End-of-stream sentinel marker.

    // Preprocessing metadata
    float scale;    //!< Scale factor applied during letterbox.
    int dx, dy;     //!< Padding offsets from letterbox operation.

    // Optional metadata
    int source_width;
    int source_height;

    // Source pixel format and planes (when not BGR)
    PixelFormat format = PixelFormat::BGR;
    // YUV420P planes (valid when format == YUV420P)
    std::vector<uint8_t> y_plane;
    std::vector<uint8_t> u_plane;
    std::vector<uint8_t> v_plane;
    int y_stride = 0;
    int uv_stride = 0;
};

/**
 * @brief Detection result from YOLOv5n.
 */
struct Detection {
    float x1, y1, x2, y2;  //!< Bounding box coordinates in letterboxed space.
    float confidence;       //!< Detection confidence score.
    int class_id;          //!< COCO class identifier.
    std::string label;     //!< Resolved class label.
};

/**
 * @brief Frame with detections and inference timing.
 */
struct ProcessedFrame {
    Frame frame;
    std::vector<Detection> detections;
    std::chrono::steady_clock::time_point inference_start;
    std::chrono::steady_clock::time_point inference_end;
};

/**
 * @brief Pipeline configuration aggregated from CLI options.
 */
struct PipelineConfig {
    // Input/Output
    std::string source;           // file:path or v4l2:/dev/videoX
    std::string output_path;      // Output video path (optional)
    std::string encoder;          // h264, mjpeg, raw
    std::string display_mode;     // off, sdl
    std::string sdl_driver;       // auto|wayland|kmsdrm|x11|dummy
    std::string cam_format;       // auto, yuyv, mjpeg
    
    // Model parameters
    std::string weights_path;     // Path to hhb.bm
    // Fixed to current model; only 640x384 accepted
    int img_width;                // Width (must be 640)
    int img_height;               // Height (must be 384)
    bool rvv;                     // Enable RVV path (on/off)
    PreprocMode pp_mode;          // Preprocessing backend (sw|rvv)
    float conf_threshold;         // Confidence threshold
    float nms_threshold;          // NMS IOU threshold
    
    // Threading
    int nn_workers;               // Number of inference workers
    std::vector<int> nn_cpus;     // CPU cores for NN inference
    std::vector<int> io_cpus;     // CPU cores for I/O operations
    bool auto_cpu_detect;         // Auto-detect best CPU cluster
    
    // Queue management
    int queue_capacity;           // Max queue size
    std::string drop_policy;      // front:wm=N or new:wm=N
    int drop_watermark;           // Watermark for dropping
    
    // Performance
    int perf_interval_ms;         // Performance reporting interval
    std::string perf_json_path;   // JSONL metrics output path
    
    // Misc
    std::string log_level;        // info, debug, warn, error
    bool use_rt_priority;         // Use real-time scheduling
    int max_frames;               // Max frames to process (<=0 means unlimited)
    std::string display_probe_path; // Save display probe to path once after first present
    int watchdog_sec;             // Watchdog timeout (0=off)

    // Default constructor with reasonable defaults
    PipelineConfig() : 
        source(""),
        output_path(""),
        encoder("h264"),
        display_mode("off"),
        sdl_driver("auto"),
        cam_format("auto"),
        weights_path("cpu_model/hhb.bm"),
        img_width(640),
        img_height(384),
        rvv(false),
        pp_mode(PreprocMode::SW),
        conf_threshold(0.25f),
        nms_threshold(0.45f),
        nn_workers(4),
        auto_cpu_detect(true),
        queue_capacity(8),
        drop_policy("front:wm=3"),
        drop_watermark(3),
        perf_interval_ms(1000),
        perf_json_path(""),
        log_level("info"),
        use_rt_priority(false),
        max_frames(-1),
        display_probe_path(""),
        watchdog_sec(0) {}
};

/**
 * @brief Aggregated performance metrics emitted as JSONL.
 */
struct PerfMetrics {
    int64_t timestamp_ms;
    float input_fps;
    float output_fps;
    float drop_percentage;
    
    // Latencies in milliseconds
    struct {
        float capture;
        float preprocess;
        float inference_p50;
        float inference_p95;
        float postprocess;
        float overlay;
        float encode;
        float display;
    } latency_ms;
    
    // Queue sizes
    std::map<std::string, int> queue_sizes;
    
    // Worker utilization percentages
    std::vector<float> worker_busy_pct;
};

// Model dimensions - CRITICAL: 384x640, not 640x640!
constexpr int MODEL_WIDTH = 640;
constexpr int MODEL_HEIGHT = 384;
constexpr int MODEL_CHANNELS = 3;

/** @brief COCO class names aligned with YOLOv5n outputs (80 entries). */
extern const std::vector<std::string> COCO_CLASSES;

} // namespace yolov5

#endif // TYPES_HPP
