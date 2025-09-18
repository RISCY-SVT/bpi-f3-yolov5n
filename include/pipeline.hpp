#ifndef PIPELINE_HPP
#define PIPELINE_HPP

#include "types.hpp"
#include "metrics.hpp"
#include "capture.hpp"
#include "engine.hpp"
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <map>
#include <set>

namespace yolov5 {

/**
 * @file pipeline.hpp
 * @brief Declares pipeline queues, frame reorderer, and executor orchestrating threads.
 *
 * Each stage in the pipeline trades ownership of frames through bounded queues
 * while keeping frame order and metrics reporting compliant with TechSpec.
 */

/**
 * @brief Thread-safe bounded queue used between pipeline stages.
 * @threading Multiple producers and consumers push/pop concurrently.
 * @ownership Queue owns its elements until popped.
 */
template<typename T>
class ThreadSafeQueue {
public:
    /**
     * @brief Construct queue with bounded capacity.
     * @param capacity Maximum number of elements before producers block.
     */
    ThreadSafeQueue(size_t capacity = 8);

    /**
     * @brief Push an item into the queue.
     * @param item Element moved into storage.
     * @param wait Block until capacity frees when true; otherwise fail fast.
     */
    bool push(T item, bool wait = true);

    /**
     * @brief Pop next item from queue.
     * @param item Destination reference populated when available.
     * @param wait Block for data when true; otherwise fail fast on empty queue.
     */
    bool pop(T& item, bool wait = true);

    /**
     * @brief Attempt non-blocking push.
     */
    bool tryPush(const T& item);

    /**
     * @brief Attempt non-blocking pop.
     */
    bool tryPop(T& item);

    /**
     * @brief Current queue size.
     */
    size_t size() const;

    /**
     * @brief Whether the queue has no elements.
     */
    bool empty() const;
    /**
     * @brief Whether the queue reached capacity.
     */
    bool full() const;

    /**
     * @brief Remove all elements without waking producers/consumers.
     */
    void clear();

    /**
     * @brief Stop queue and wake all waiters so they can exit gracefully.
     */
    void stop();
    
private:
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::queue<T> queue_;
    size_t capacity_;
    std::atomic<bool> stopped_;
};

// Inline implementation for template methods
template<typename T>
ThreadSafeQueue<T>::ThreadSafeQueue(size_t capacity)
    : capacity_(capacity), stopped_(false) {}

template<typename T>
bool ThreadSafeQueue<T>::push(T item, bool wait) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (wait) {
        not_full_.wait(lock, [&]{ return stopped_.load() || queue_.size() < capacity_; });
        if (stopped_) return false;
    } else if (queue_.size() >= capacity_) {
        return false;
    }
    queue_.push(std::move(item));
    not_empty_.notify_one();
    return true;
}

template<typename T>
bool ThreadSafeQueue<T>::pop(T& item, bool wait) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (wait) {
        not_empty_.wait(lock, [&]{ return stopped_.load() || !queue_.empty(); });
        if (stopped_ && queue_.empty()) return false;
    } else if (queue_.empty()) {
        return false;
    }
    item = std::move(queue_.front());
    queue_.pop();
    not_full_.notify_one();
    return true;
}

template<typename T>
bool ThreadSafeQueue<T>::tryPush(const T& item) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (queue_.size() >= capacity_) return false;
    queue_.push(item);
    not_empty_.notify_one();
    return true;
}

template<typename T>
bool ThreadSafeQueue<T>::tryPop(T& item) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (queue_.empty()) return false;
    item = std::move(queue_.front());
    queue_.pop();
    not_full_.notify_one();
    return true;
}

template<typename T>
size_t ThreadSafeQueue<T>::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

template<typename T>
bool ThreadSafeQueue<T>::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
}

template<typename T>
bool ThreadSafeQueue<T>::full() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size() >= capacity_;
}

template<typename T>
void ThreadSafeQueue<T>::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::queue<T> empty;
    std::swap(queue_, empty);
}

template<typename T>
void ThreadSafeQueue<T>::stop() {
    stopped_.store(true);
    not_empty_.notify_all();
    not_full_.notify_all();
}

/**
 * @brief Ensures processed frames exit in monotonically increasing order.
 * @threading Owned by overlay/output thread; fed by postprocess thread.
 * @ownership Holds `ProcessedFrame` instances until their frame_id is expected.
 */
class FrameReorderer {
public:
    FrameReorderer();

    /**
     * @brief Add processed frame awaiting ordered emission.
     * @param frame Frame tagged with source frame_id.
     */
    void addFrame(const ProcessedFrame& frame);

    /**
     * @brief Mark frame id as intentionally dropped by upstream logic.
     * @param frame_id Identifier to skip during ordering.
     */
    void markDropped(uint64_t frame_id);

    /**
     * @brief Get next frame in monotonically increasing order.
     * @param frame Output container receiving the frame.
     * @return True when a frame is returned, false after stop().
     */
    bool getNextFrame(ProcessedFrame& frame);

    /**
     * @brief Reset internal buffers and expected id to zero.
     */
    void reset();

    /**
     * @brief Wake waiters and prevent further blocking operations.
     */
    void stop();

    /**
     * @brief Number of frames currently buffered awaiting reorder.
     */
    size_t pendingCount() const;
    
private:
    std::map<uint64_t, ProcessedFrame> buffer_;
    std::set<uint64_t> dropped_ids_;
    uint64_t expected_id_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stopped_;
};

/**
 * @brief Orchestrates capture → preprocess → inference → post → overlay → output.
 * @threading Owns multiple stage threads plus N inference workers.
 * @ownership Owns capture engines, queues, metrics writer, and display.
 */
class Pipeline {
public:
    Pipeline(const PipelineConfig& config);
    ~Pipeline();

    /**
     * @brief Start all pipeline threads and enqueue first actions.
     * @return True when all stages initialize successfully.
     */
    bool start();

    /**
     * @brief Signal all stages to exit and wake blocked threads.
     */
    void stop();

    /**
     * @brief Join stage threads; safe to call multiple times.
     */
    void join();

    /**
     * @brief Fetch latest metrics snapshot (protected by mutex).
     */
    PerfMetrics getMetrics() const;
    uint64_t processedCount() const { return out_count_.load(); }
    uint64_t inCount() const { return in_count_.load(); }
    uint64_t droppedCount() const { return dropped_frames_.load(); }
    bool isRunning() const { return running_.load(); }
    
private:
    // Pipeline stages (run in separate threads)
    /** @brief Capture thread pulling frames from FFmpeg/V4L2 sources. */
    void captureThread();
    /** @brief Preprocess thread executing resize/letterbox and RVV conversions. */
    void preprocessThread();
    /** @brief Scheduler assigns preprocessed frames to inference workers. */
    void schedulerThread();
    /** @brief Inference worker executing CSI-NN2 session. */
    void inferenceWorker(int worker_id);
    /** @brief Postprocess thread aggregates detections and prepares overlays. */
    void postprocessThread();
    /** @brief Overlay thread draws bounding boxes and composes display frames. */
    void overlayThread();
    /** @brief Output thread handles encoding, file sink, and display submission. */
    void outputThread();
    /** @brief Metrics thread samples queues and writes JSONL metrics. */
    void metricsThread();
    
    // CPU affinity helpers
    /** @brief Apply pthread affinity mask for current stage. */
    void setCPUAffinity(const std::vector<int>& cpus);
    /** @brief Run micro-benchmark to select faster CPU cluster when auto. */
    void runCPUBenchmark();

    // Frame dropping logic
    /** @brief Evaluate drop policy against current queue pressure. */
    bool shouldDropFrame(uint64_t frame_id);
    
    // Configuration
    PipelineConfig config_;
    
    // Components
    std::unique_ptr<ICapture> capture_;
    std::vector<std::unique_ptr<IEngine>> engines_;
    std::unique_ptr<FrameReorderer> reorderer_;
    
    // Queues between stages
    ThreadSafeQueue<Frame> capture_queue_;
    ThreadSafeQueue<Frame> preprocess_queue_;
    ThreadSafeQueue<Frame> inference_queue_;
    ThreadSafeQueue<ProcessedFrame> postprocess_queue_;
    ThreadSafeQueue<ProcessedFrame> overlay_queue_;
    ThreadSafeQueue<ProcessedFrame> output_queue_;
    
    // Threads
    std::thread capture_thread_;
    std::thread preprocess_thread_;
    std::thread scheduler_thread_;
    std::vector<std::thread> inference_workers_;
    std::thread postprocess_thread_;
    std::thread overlay_thread_;
    std::thread output_thread_;
    std::thread metrics_thread_;
    std::thread mem_logger_thread_;
    
    // Control
    std::atomic<bool> running_;
    std::atomic<bool> capture_eof_{false};
    std::atomic<uint64_t> inflight_{0};
    std::atomic<uint64_t> frame_counter_;
    std::atomic<uint64_t> dropped_frames_;
    std::atomic<bool> mem_logger_stop_{false};
    
    // Metrics
    mutable std::mutex metrics_mutex_;
    PerfMetrics current_metrics_;
    // Metrics accumulators
    std::vector<double> cap_lat_;
    std::vector<double> pp_lat_;
    std::vector<double> inf_lat_;
    std::vector<std::vector<double>> inf_hist_; // per-worker sliding window
    std::vector<uint64_t> worker_busy_ns_;      // per-worker busy ns since last metrics
    std::vector<double> post_lat_;
    std::vector<double> overlay_lat_;
    std::vector<double> enc_lat_;
    std::atomic<uint64_t> in_count_{0};
    std::atomic<uint64_t> out_count_{0};
    std::unique_ptr<JSONLMetricsWriter> metrics_writer_;
    std::condition_variable metrics_cv_;
    std::mutex metrics_cv_mu_;
};

} // namespace yolov5

#endif // PIPELINE_HPP
