#ifndef TYPES_HPP
#define TYPES_HPP

#include <opencv2/core.hpp>
#include <chrono>
#include <vector>
#include <string>
#include <map>

namespace yolov5 {

// Frame data structure for pipeline
struct Frame {
    uint64_t frame_id;
    cv::Mat image;  // BGR format
    std::chrono::steady_clock::time_point timestamp;
    void* model_input = nullptr; // Preprocessed NCHW FP32 buffer
    bool eos = false; // end-of-stream sentinel
    
    // Preprocessing metadata
    float scale;    // Scale factor from letterboxing
    int dx, dy;     // Offset from letterboxing
    
    // Optional metadata
    int source_width;
    int source_height;
};

// Detection result
struct Detection {
    float x1, y1, x2, y2;  // Bounding box coordinates
    float confidence;       // Detection confidence
    int class_id;          // COCO class ID
    std::string label;     // Class label
};

// Frame with detections
struct ProcessedFrame {
    Frame frame;
    std::vector<Detection> detections;
    std::chrono::steady_clock::time_point inference_start;
    std::chrono::steady_clock::time_point inference_end;
};

// Pipeline configuration
struct PipelineConfig {
    // Input/Output
    std::string source;           // file:path or v4l2:/dev/videoX
    std::string output_path;      // Output video path (optional)
    std::string encoder;          // h264, mjpeg, raw
    std::string display_mode;     // auto, sdl, off
    
    // Model parameters
    std::string weights_path;     // Path to hhb.bm
    // Fixed to current model; only 640x384 accepted
    int img_width;                // Width (must be 640)
    int img_height;               // Height (must be 384)
    bool rvv;                     // Enable RVV path (on/off)
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
    
    // Default constructor with reasonable defaults
    PipelineConfig() : 
        source(""),
        output_path(""),
        encoder("h264"),
        display_mode("auto"),
        weights_path("cpu_model/hhb.bm"),
        img_width(640),
        img_height(384),
        rvv(false),
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
        max_frames(-1) {}
};

// Performance metrics
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

// COCO classes (80 classes)
extern const std::vector<std::string> COCO_CLASSES;

} // namespace yolov5

#endif // TYPES_HPP
