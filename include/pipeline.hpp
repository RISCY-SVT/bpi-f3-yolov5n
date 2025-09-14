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

// Thread-safe queue for pipeline stages
template<typename T>
class ThreadSafeQueue {
public:
    ThreadSafeQueue(size_t capacity = 8);
    
    // Push item to queue (may block if full)
    bool push(T item, bool wait = true);
    
    // Pop item from queue (may block if empty)
    bool pop(T& item, bool wait = true);
    
    // Try to push without blocking
    bool tryPush(const T& item);
    
    // Try to pop without blocking
    bool tryPop(T& item);
    
    // Get current size
    size_t size() const;
    
    // Check if empty/full
    bool empty() const;
    bool full() const;
    
    // Clear queue
    void clear();
    
    // Stop the queue (unblock all waiting threads)
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

// Frame reorderer to ensure output order
class FrameReorderer {
public:
    FrameReorderer();
    
    // Add processed frame
    void addFrame(const ProcessedFrame& frame);
    
    // Mark frame as dropped
    void markDropped(uint64_t frame_id);
    
    // Get next frame in order (blocks if not ready)
    bool getNextFrame(ProcessedFrame& frame);
    
    // Reset reorderer
    void reset();
    
    // Stop reorderer
    void stop();
    
private:
    std::map<uint64_t, ProcessedFrame> buffer_;
    std::set<uint64_t> dropped_ids_;
    uint64_t expected_id_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stopped_;
};

// Video processing pipeline
class Pipeline {
public:
    Pipeline(const PipelineConfig& config);
    ~Pipeline();
    
    // Start pipeline processing
    bool start();
    
    // Stop pipeline
    void stop();
    
    // Wait for pipeline to complete
    void join();
    
    // Get performance metrics
    PerfMetrics getMetrics() const;
    uint64_t processedCount() const { return out_count_.load(); }
    uint64_t inCount() const { return in_count_.load(); }
    uint64_t droppedCount() const { return dropped_frames_.load(); }
    bool isRunning() const { return running_.load(); }
    
private:
    // Pipeline stages (run in separate threads)
    void captureThread();
    void preprocessThread();
    void schedulerThread();
    void inferenceWorker(int worker_id);
    void postprocessThread();
    void overlayThread();
    void outputThread();
    void metricsThread();
    
    // CPU affinity helpers
    void setCPUAffinity(const std::vector<int>& cpus);
    void runCPUBenchmark();
    
    // Frame dropping logic
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
    
    // Control
    std::atomic<bool> running_;
    std::atomic<bool> capture_eof_{false};
    std::atomic<uint64_t> inflight_{0};
    std::atomic<uint64_t> frame_counter_;
    std::atomic<uint64_t> dropped_frames_;
    
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
