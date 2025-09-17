#include "pipeline.hpp"
#include "capture.hpp"
#include "preprocess.hpp"
#include "engine.hpp"
#include "display.hpp"
#include <pthread.h>
#include <sched.h>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>
#include <chrono>
#include <atomic>
#include <condition_variable>
#include <iomanip>
#include <sstream>
#include <string>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <cstdio>
#include <cmath>

#include <opencv2/imgproc.hpp>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/opt.h>
#include <libavutil/error.h>
#include <libavutil/pixfmt.h>
#include <libavutil/pixdesc.h>
}

/**
 * @file pipeline.cpp
 * @brief Implements multi-threaded video pipeline linking capture to sinks.
 */

namespace yolov5 {

namespace {

/**
 * @brief Tracks SDL display status, watchdog timers, and metrics overlay cache.
 */
struct DisplayState {
    std::shared_ptr<IDisplay> display;
    std::atomic<int64_t> last_present_ns{0};
    std::atomic<int64_t> last_metrics_ns{0};
    std::atomic<bool> stop_requested{false};
    std::atomic<bool> watchdog_triggered{false};
    std::atomic<bool> probe_saved{false};
    int watchdog_sec{0};
    std::string driver_name{"null"};
    std::string probe_path;
    std::mutex display_mutex;
    std::mutex lat_mutex;
    std::vector<double> display_lat;
    std::thread watchdog_thread;
};

static int64_t to_ns(const std::chrono::steady_clock::time_point& tp) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count();
}

static int64_t now_ns() {
    return to_ns(std::chrono::steady_clock::now());
}

static std::mutex g_display_state_mu;
static std::map<const Pipeline*, std::shared_ptr<DisplayState>> g_display_state;

static std::shared_ptr<DisplayState> get_state(const Pipeline* p) {
    std::lock_guard<std::mutex> lk(g_display_state_mu);
    auto it = g_display_state.find(p);
    if (it != g_display_state.end()) return it->second;
    return nullptr;
}

static void set_state(const Pipeline* p, const std::shared_ptr<DisplayState>& state) {
    std::lock_guard<std::mutex> lk(g_display_state_mu);
    g_display_state[p] = state;
}

static void erase_state(const Pipeline* p) {
    std::lock_guard<std::mutex> lk(g_display_state_mu);
    g_display_state.erase(p);
}

static bool ensure_probe_parent_dir(const std::string& path) {
    try {
        std::filesystem::path p(path);
        auto parent = p.parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }
        return true;
    } catch (const std::exception& ex) {
        std::cerr << "[display] failed to prepare directory for probe '" << path
                  << "': " << ex.what() << std::endl;
        return false;
    }
}

static bool write_probe_ppm(const cv::Mat& frame, const std::string& path) {
    if (frame.empty()) return false;
    cv::Mat resized;
    const int target_w = 320;
    if (frame.cols > target_w) {
        const double scale = static_cast<double>(target_w) / static_cast<double>(frame.cols);
        const int target_h = std::max(1, static_cast<int>(std::round(frame.rows * scale)));
        cv::resize(frame, resized, cv::Size(target_w, target_h), 0.0, 0.0, cv::INTER_NEAREST);
    } else {
        resized = frame;
    }
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;
    ofs << "P6\n" << rgb.cols << ' ' << rgb.rows << "\n255\n";
    ofs.write(reinterpret_cast<const char*>(rgb.data), static_cast<std::streamsize>(rgb.total() * rgb.elemSize()));
    return ofs.good();
}

// Save first displayed frame to CLI-specified path (used for diagnostics).
static void save_display_probe(const std::shared_ptr<DisplayState>& state, const cv::Mat& frame) {
    if (!state) return;
    if (state->probe_saved.load()) return;
    if (state->probe_path.empty()) return;
    if (!ensure_probe_parent_dir(state->probe_path)) {
        return;
    }
    if (!write_probe_ppm(frame, state->probe_path)) {
        std::cerr << "[display] probe snapshot failed for path " << state->probe_path << std::endl;
        return;
    }
    state->probe_saved.store(true);
    std::cout << "[display] probe saved to " << state->probe_path << std::endl;
}

} // namespace

FrameReorderer::FrameReorderer() : expected_id_(0), stopped_(false) {}

void FrameReorderer::addFrame(const ProcessedFrame& frame) {
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_.emplace(frame.frame.frame_id, frame);
    cv_.notify_all();
}

void FrameReorderer::markDropped(uint64_t frame_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    dropped_ids_.insert(frame_id);
    // If we just marked the expected frame as dropped, advance expected_id_
    while (dropped_ids_.count(expected_id_) > 0) {
        dropped_ids_.erase(expected_id_);
        expected_id_++;
    }
    cv_.notify_all();
}

bool FrameReorderer::getNextFrame(ProcessedFrame& out) {
    std::unique_lock<std::mutex> lock(mutex_);
    // Wait until the next expected frame arrives or is marked dropped to maintain strict ordering.
    for (;;) {
        if (stopped_) return false;
        // Release dropped frames at head
        while (dropped_ids_.count(expected_id_) > 0) {
            dropped_ids_.erase(expected_id_);
            expected_id_++;
        }
        // If the next expected frame is available, return it
        auto it = buffer_.find(expected_id_);
        if (it != buffer_.end()) {
            out = std::move(it->second);
            buffer_.erase(it);
            expected_id_++;
            return true;
        }
        // Otherwise wait for new frames or stop
        cv_.wait(lock);
    }
}

void FrameReorderer::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_.clear();
    dropped_ids_.clear();
    expected_id_ = 0;
}

void FrameReorderer::stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    stopped_ = true;
    cv_.notify_all();
}

// Helper: bind current thread to specific CPU list
static void bind_to_cpus(const std::vector<int>& cpus) {
    if (cpus.empty()) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : cpus) if (c >= 0 && c < CPU_SETSIZE) CPU_SET(c, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

// Simple micro workload for benchmarking (simulate compute). Measures cluster performance.
static double micro_work_ms(int iters = 500000) {
    volatile float x = 1.0f, y = 2.0f;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) { x = x * 1.000001f + y * 0.999999f; y = y * 0.999997f + x * 1.000003f; }
    auto t1 = std::chrono::steady_clock::now();
    (void)x; (void)y;
    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
}

void Pipeline::setCPUAffinity(const std::vector<int>& cpus) {
    bind_to_cpus(cpus);
}

void Pipeline::runCPUBenchmark() {
    if (!config_.auto_cpu_detect) return;
    // Bench on clusters 0-3 and 4-7 (if present)
    std::vector<int> c0 = {0,1,2,3};
    std::vector<int> c1 = {4,5,6,7};
    auto bench = [&](const std::vector<int>& cpus) {
        if (cpus.empty()) return 1e9;
        double best = 1e9;
        std::thread t([&]{
            bind_to_cpus(cpus);
            // Run multiple trials
            double acc = 0.0; int n = 8;
            for (int i = 0; i < n; ++i) acc += micro_work_ms();
            best = acc / n;
        });
        t.join();
        return best;
    };
    double t0 = bench(c0);
    double t1 = bench(c1);
    if (t1 < t0) config_.nn_cpus = c1; else config_.nn_cpus = c0;
    config_.auto_cpu_detect = false; // decision made
    std::cout << "[INFO] Auto nn-cpus selected: ";
    for (auto c : config_.nn_cpus) std::cout << c << ' ';
    std::cout << std::endl;
}

// ---------------------- FFmpeg encoder (minimal) ----------------------
/**
 * @brief Minimal FFmpeg encoder supporting H264, MJPEG, and raw BGR outputs.
 *
 * Owned by output thread. Handles container auto-selection, color conversion,
 * and draining during shutdown.
 */
class FFmpegEncoder {
public:
    FFmpegEncoder() : fmt_(nullptr), oc_(nullptr), st_(nullptr), enc_(nullptr), sws_(nullptr), frame_(nullptr), pkt_(nullptr), opened_(false), drained_pkts_(0) {}
    ~FFmpegEncoder() { close(); }

    /**
     * @brief Open encoder with requested codec and container derived from CLI.
     */
    bool open(const std::string& path, const std::string& enc_name, int w, int h, int fps) {
        output_path_.clear();
        width_ = w; height_ = h; fps_ = fps > 0 ? fps : 30;
        // Select container by encoder and auto-rename output extension if needed
        std::string out_path = path;
        std::string selected_mux = (enc_name == "h264") ? "mp4" : "avi";
        auto has_ext = [&](const std::string& p, const char* ext){ return p.size() >= strlen(ext) && p.rfind(ext) == p.size() - strlen(ext); };
        if (selected_mux == "mp4" && !has_ext(out_path, ".mp4")) {
            size_t dot = out_path.find_last_of('.'); std::string stem = dot==std::string::npos? out_path : out_path.substr(0,dot);
            out_path = stem + ".mp4";
            std::cout << "[INFO] encoder: chosen container=mp4; renamed output to " << out_path << std::endl;
        } else if (selected_mux == "avi" && !has_ext(out_path, ".avi")) {
            size_t dot = out_path.find_last_of('.'); std::string stem = dot==std::string::npos? out_path : out_path.substr(0,dot);
            std::string suf = (enc_name == "raw") ? "_raw" : (enc_name == "mjpeg" ? "_mjpeg" : "");
            out_path = stem + suf + ".avi";
            std::cout << "[INFO] encoder: chosen container=avi; renamed output to " << out_path << std::endl;
        }
        const char* filename = out_path.c_str();
        if (avformat_alloc_output_context2(&oc_, nullptr, selected_mux.c_str(), filename) < 0 || !oc_) {
            std::cerr << "[WARN] encoder: failed to alloc output context for '" << path << "'" << std::endl;
            close();
            return false;
        }
        fmt_ = oc_->oformat;
        // Select codec
        const AVCodec* codec = nullptr;
        if (enc_name == "h264") {
            codec = avcodec_find_encoder_by_name("libx264");
            if (!codec) codec = avcodec_find_encoder(AV_CODEC_ID_H264);
        } else if (enc_name == "mjpeg") {
            codec = avcodec_find_encoder(AV_CODEC_ID_MJPEG);
        } else {
            codec = avcodec_find_encoder(AV_CODEC_ID_RAWVIDEO);
        }
        if (!codec) { std::cerr << "[WARN] encoder: codec not found for " << enc_name << std::endl; close(); return false; }
        st_ = avformat_new_stream(oc_, codec);
        if (!st_) { std::cerr << "[WARN] encoder: avformat_new_stream failed" << std::endl; close(); return false; }
        st_->id = oc_->nb_streams - 1;
        enc_ = avcodec_alloc_context3(codec);
        if (!enc_) { std::cerr << "[WARN] encoder: avcodec_alloc_context3 failed" << std::endl; close(); return false; }
        enc_->codec_id = codec->id;
        enc_->width = width_;
        enc_->height = height_;
        enc_->time_base = AVRational{1, fps_};
        st_->time_base = enc_->time_base;
        if (enc_->codec_id == AV_CODEC_ID_H264) enc_->pix_fmt = AV_PIX_FMT_YUV420P;
        else if (enc_->codec_id == AV_CODEC_ID_MJPEG) enc_->pix_fmt = AV_PIX_FMT_YUVJ420P;
        else if (enc_->codec_id == AV_CODEC_ID_RAWVIDEO) enc_->pix_fmt = AV_PIX_FMT_BGR24;
        if (enc_->codec_id == AV_CODEC_ID_H264) {
            av_opt_set(enc_->priv_data, "preset", "ultrafast", 0);
            av_opt_set(enc_->priv_data, "tune", "zerolatency", 0);
        }
        if (oc_->oformat->flags & AVFMT_GLOBALHEADER) enc_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        int ret = avcodec_open2(enc_, codec, nullptr);
        if (ret < 0) { std::cerr << "[WARN] encoder: avcodec_open2 failed: " << err2str(ret) << std::endl; close(); return false; }
        ret = avcodec_parameters_from_context(st_->codecpar, enc_);
        if (ret < 0) { std::cerr << "[WARN] encoder: parameters_from_context failed: " << err2str(ret) << std::endl; close(); return false; }
        if (!(fmt_->flags & AVFMT_NOFILE)) {
            ret = avio_open(&oc_->pb, filename, AVIO_FLAG_WRITE);
            if (ret < 0) { std::cerr << "[WARN] encoder: avio_open failed: " << err2str(ret) << std::endl; close(); return false; }
        }
        // Optional faststart for mp4/mov
        AVDictionary* mux_opts = nullptr;
        if (selected_mux == "mp4") {
            av_dict_set(&mux_opts, "movflags", "+faststart", 0);
        }
        ret = avformat_write_header(oc_, &mux_opts);
        if (ret < 0) { std::cerr << "[WARN] encoder: write_header failed: " << err2str(ret) << std::endl; if (mux_opts) av_dict_free(&mux_opts); close(); return false; }
        if (mux_opts) av_dict_free(&mux_opts);
        std::cout << "[INFO] encoder: wrote header" << std::endl;
        // Allocate frame and packet
        frame_ = av_frame_alloc();
        frame_->format = enc_->pix_fmt; frame_->width = width_; frame_->height = height_;
        ret = av_frame_get_buffer(frame_, 32);
        if (ret < 0) { std::cerr << "[WARN] encoder: frame_get_buffer failed: " << err2str(ret) << std::endl; close(); return false; }
        pkt_ = av_packet_alloc();
        if (enc_->codec_id == AV_CODEC_ID_RAWVIDEO) {
            sws_ = nullptr;
        } else {
            sws_ = sws_getContext(width_, height_, AV_PIX_FMT_BGR24, width_, height_, enc_->pix_fmt, SWS_BILINEAR, nullptr, nullptr, nullptr);
            if (!sws_) { std::cerr << "[WARN] sws_getContext failed" << std::endl; close(); return false; }
        }
        opened_ = true; pts_ = 0;
        output_path_ = out_path;
        std::cout << "[INFO] encoder: container=" << selected_mux
                  << " codec=" << avcodec_get_name(enc_->codec_id)
                  << " pix_fmt=" << av_get_pix_fmt_name(enc_->pix_fmt) << std::endl;
        return true;
    }

    /**
     * @brief Encode one BGR frame into the configured container.
     */
    bool write(const cv::Mat& bgr) {
        if (!opened_) return false;
        int ret = av_frame_make_writable(frame_);
        if (ret < 0) {
            std::cerr << "[WARN] encoder: av_frame_make_writable failed: " << err2str(ret) << std::endl;
            return false;
        }
        if (enc_->codec_id == AV_CODEC_ID_RAWVIDEO) {
            // Copy BGR24 directly into frame buffer
            for (int y = 0; y < height_; ++y) {
                memcpy(frame_->data[0] + y * frame_->linesize[0], bgr.data + y * bgr.step, width_ * 3);
            }
        } else {
            const uint8_t* src_slices[1] = { bgr.data };
            int src_stride[1] = { (int)bgr.step };
            sws_scale(sws_, src_slices, src_stride, 0, height_, frame_->data, frame_->linesize);
        }
        frame_->pts = pts_++;
        ret = avcodec_send_frame(enc_, frame_);
        if (ret < 0) {
            std::cerr << "[WARN] encoder: avcodec_send_frame failed: " << err2str(ret) << std::endl;
            return false;
        }
        for (;;) {
            ret = avcodec_receive_packet(enc_, pkt_);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
            if (ret < 0) {
                std::cerr << "[WARN] encoder: avcodec_receive_packet failed: " << err2str(ret) << std::endl;
                return false;
            }
            av_packet_rescale_ts(pkt_, enc_->time_base, st_->time_base);
            pkt_->stream_index = st_->index;
            ret = av_interleaved_write_frame(oc_, pkt_);
            if (ret < 0) {
                std::cerr << "[WARN] encoder: interleaved_write_frame failed: " << err2str(ret) << std::endl;
                av_packet_unref(pkt_);
                return false;
            }
            av_packet_unref(pkt_);
        }
        return true;
    }

    /**
     * @brief Drain encoder and finalize file.
     */
    void close() {
        const bool was_open = opened_;
        drained_pkts_ = 0;
        if (was_open && enc_ && pkt_ && st_) {
            int ret = avcodec_send_frame(enc_, nullptr);
            if (ret < 0 && ret != AVERROR_EOF) {
                std::cerr << "[WARN] encoder: avcodec_send_frame(NULL) failed: " << err2str(ret) << std::endl;
            }
            for (;;) {
                ret = avcodec_receive_packet(enc_, pkt_);
                if (ret == AVERROR_EOF || ret == AVERROR(EAGAIN)) break;
                if (ret < 0) {
                    std::cerr << "[WARN] encoder: drain receive failed: " << err2str(ret) << std::endl;
                    break;
                }
                av_packet_rescale_ts(pkt_, enc_->time_base, st_->time_base);
                pkt_->stream_index = st_->index;
                int wret = av_interleaved_write_frame(oc_, pkt_);
                if (wret < 0) {
                    std::cerr << "[WARN] encoder: interleaved_write_frame (drain) failed: " << err2str(wret) << std::endl;
                    av_packet_unref(pkt_);
                    break;
                }
                av_packet_unref(pkt_);
                ++drained_pkts_;
            }
            std::cout << "[INFO] encoder: drained " << drained_pkts_ << " packets" << std::endl;
        }
        if (was_open && oc_) {
            int ret = av_write_trailer(oc_);
            if (ret < 0) {
                std::cerr << "[WARN] encoder: av_write_trailer failed: " << err2str(ret) << std::endl;
            } else {
                std::cout << "[INFO] encoder: wrote trailer" << std::endl;
            }
        }
        if (frame_) { av_frame_free(&frame_); frame_ = nullptr; }
        if (pkt_) { av_packet_free(&pkt_); pkt_ = nullptr; }
        if (enc_) { avcodec_free_context(&enc_); enc_ = nullptr; }
        if (sws_) { sws_freeContext(sws_); sws_ = nullptr; }
        if (oc_) {
            if (fmt_ && !(fmt_->flags & AVFMT_NOFILE) && oc_->pb) {
                avio_closep(&oc_->pb);
                std::cout << "[INFO] encoder: avio closed" << std::endl;
            }
            avformat_free_context(oc_);
            oc_ = nullptr;
        }
        st_ = nullptr;
        fmt_ = nullptr;
        opened_ = false;
        if (was_open) {
            if (!output_path_.empty()) {
                std::cout << "[INFO] encoder: closed output='" << output_path_ << "'" << std::endl;
            } else {
                std::cout << "[INFO] encoder: closed" << std::endl;
            }
        }
        output_path_.clear();
    }

private:
    static std::string err2str(int err) {
        char buf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(err, buf, sizeof(buf));
        return std::string(buf);
    }
    const AVOutputFormat* fmt_;
    AVFormatContext* oc_;
    AVStream* st_;
    AVCodecContext* enc_;
    SwsContext* sws_;
    AVFrame* frame_;
    AVPacket* pkt_;
    int width_, height_, fps_;
    int64_t pts_ = 0;
    bool opened_;
    int drained_pkts_;
    std::string output_path_;
};

// ---------------------- Pipeline implementation ----------------------
/**
 * @brief Construct pipeline, instantiate queues, capture, engines, and reorderer.
 */
Pipeline::Pipeline(const PipelineConfig& cfg)
    : config_(cfg),
      capture_queue_(cfg.queue_capacity),
      preprocess_queue_(cfg.queue_capacity),
      inference_queue_(cfg.queue_capacity),
      postprocess_queue_(cfg.queue_capacity),
      overlay_queue_(cfg.queue_capacity),
      output_queue_(cfg.queue_capacity),
      running_(false), frame_counter_(0), dropped_frames_(0) {
    reorderer_ = std::make_unique<FrameReorderer>();
}

/** @brief Ensure all threads stop before destruction. */
Pipeline::~Pipeline() { stop(); join(); }

/**
 * @brief Initialize components and spawn all stage threads.
 * @return False if capture or engine initialization fails.
 */
bool Pipeline::start() {
    if (running_) return true;
    // Auto affinity selection
    if (config_.auto_cpu_detect && config_.nn_cpus.empty()) runCPUBenchmark();
    // Create capture
    capture_ = createCapture(config_.source);
    if (!capture_ || !capture_->init(config_.source)) {
        std::cerr << "[ERROR] Capture init failed" << std::endl; return false;
    }
    // Create engines
    engines_.clear(); engines_.reserve(std::max(1, config_.nn_workers));
    for (int i = 0; i < std::max(1, config_.nn_workers); ++i) {
        auto e = createEngine("csi");
        if (!e || !e->init(config_.weights_path)) {
            std::cerr << "[ERROR] Engine init failed" << std::endl; return false;
        }
        engines_.push_back(std::move(e));
    }
    running_ = true;
    auto state = std::make_shared<DisplayState>();
    state->probe_path = config_.display_probe_path;
    state->watchdog_sec = config_.watchdog_sec;
    state->last_metrics_ns.store(now_ns());
    set_state(this, state);
    if (state->watchdog_sec > 0) {
        state->watchdog_thread = std::thread([this, state]() {
            const int64_t timeout_ns = static_cast<int64_t>(state->watchdog_sec) * 1000000000LL;
            while (!state->stop_requested.load()) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                if (state->stop_requested.load()) break;
                const int64_t now = now_ns();
                const int64_t lp = state->last_present_ns.load();
                const int64_t lm = state->last_metrics_ns.load();
                const bool present_stalled = (lp > 0) && (now - lp > timeout_ns);
                const bool metrics_stalled = (lm > 0) && (now - lm > timeout_ns);
                if (!present_stalled && !metrics_stalled) {
                    continue;
                }
                if (!state->watchdog_triggered.exchange(true)) {
                    std::cout << "[watchdog] no progress for " << state->watchdog_sec
                              << "s, shutting down..." << std::endl;
                }
                this->stop();
                state->stop_requested.store(true);
                break;
            }
        });
    }
    if (!config_.perf_json_path.empty()) {
        metrics_writer_ = std::make_unique<JSONLMetricsWriter>(config_.perf_json_path);
    }
    // Launch threads
    capture_thread_ = std::thread(&Pipeline::captureThread, this);
    preprocess_thread_ = std::thread(&Pipeline::preprocessThread, this);
    scheduler_thread_ = std::thread(&Pipeline::schedulerThread, this);
    inf_hist_.assign(std::max(1, config_.nn_workers), {});
    worker_busy_ns_.assign(std::max(1, config_.nn_workers), 0);
    for (int i = 0; i < std::max(1, config_.nn_workers); ++i) {
        inference_workers_.emplace_back(&Pipeline::inferenceWorker, this, i);
    }
    postprocess_thread_ = std::thread(&Pipeline::postprocessThread, this);
    overlay_thread_ = std::thread(&Pipeline::overlayThread, this);
    output_thread_ = std::thread(&Pipeline::outputThread, this);
    metrics_thread_ = std::thread(&Pipeline::metricsThread, this);
    return true;
}

/** @brief Signal all queues to stop and request thread termination. */
void Pipeline::stop() {
    if (!running_) return;
    running_ = false;
    capture_queue_.stop();
    preprocess_queue_.stop();
    inference_queue_.stop();
    postprocess_queue_.stop();
    overlay_queue_.stop();
    output_queue_.stop();
    if (reorderer_) reorderer_->stop();
    metrics_cv_.notify_all();
    if (auto state = get_state(this)) {
        state->stop_requested.store(true);
    }
}

/** @brief Join every thread spawned during start(). */
void Pipeline::join() {
    auto join_t = [](std::thread& t){ if (t.joinable()) t.join(); };
    // Join order: sink -> overlay -> post -> workers -> scheduler -> preprocess -> capture -> metrics
    join_t(output_thread_);
    join_t(overlay_thread_);
    join_t(postprocess_thread_);
    for (auto& t : inference_workers_) join_t(t);
    join_t(scheduler_thread_);
    join_t(preprocess_thread_);
    join_t(capture_thread_);
    join_t(metrics_thread_);
    if (auto state = get_state(this)) {
        if (state->watchdog_thread.joinable()) state->watchdog_thread.join();
    }
    erase_state(this);
}

/** @brief Return last published metrics snapshot (thread-safe). */
PerfMetrics Pipeline::getMetrics() const {
    std::lock_guard<std::mutex> lk(metrics_mutex_);
    return current_metrics_;
}

// ---------------------- Threads ----------------------
/**
 * @brief Capture thread pulling frames from source into capture_queue_.
 *
 * Applies drop policy checks before pushing to preprocess queue when pressure
 * arises. Maintains frame_id monotonic counter.
 */
void Pipeline::captureThread() {
    setCPUAffinity(config_.io_cpus);
    while (running_) {
        Frame f;
        auto t0 = std::chrono::steady_clock::now();
        bool got = false;
        try {
            got = capture_->getFrame(f);
        } catch (const std::exception& ex) {
            std::cerr << "[ERROR] capture thread exception: " << ex.what() << std::endl;
            running_ = false;
            capture_queue_.stop();
            break;
        }
        if (!got) {
            // For file: scheme, propagate EOS and exit; for v4l2 sleep and retry
            if (config_.source.rfind("file:", 0) == 0) {
                f.eos = true;
                f.timestamp = t0;
                capture_queue_.push(std::move(f));
                capture_queue_.stop();
                capture_eof_.store(true);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        auto t1 = std::chrono::steady_clock::now();
        if (f.timestamp.time_since_epoch().count() == 0) {
            f.timestamp = t0;
        }
        capture_queue_.push(std::move(f));
        {
            std::lock_guard<std::mutex> lk(metrics_mutex_);
            cap_lat_.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0);
            in_count_++;
            inflight_++;
        }
        // Respect max-frames early for file source by signaling EOS
        if (config_.max_frames > 0 && in_count_ >= (uint64_t)config_.max_frames && config_.source.rfind("file:", 0) == 0) {
            Frame eosF; eosF.eos = true; eosF.timestamp = std::chrono::steady_clock::now();
            capture_queue_.push(std::move(eosF));
            capture_queue_.stop();
            capture_eof_.store(true);
            break;
        }
    }
}

/**
 * @brief Preprocess thread handling letterbox, color conversion, and tensor prep.
 *
 * Uses configured preprocessor (SW/RVV) and writes results into inference queue.
 */
void Pipeline::preprocessThread() {
    setCPUAffinity(config_.io_cpus);
    Preprocessor pp(config_.pp_mode);
    // Model expects FP16 input; always allocate FP16 buffer
    const size_t in_size = MODEL_CHANNELS * MODEL_HEIGHT * MODEL_WIDTH * sizeof(uint16_t);
    while (running_) {
        Frame f;
        if (!capture_queue_.pop(f)) break;
        if (f.eos) {
            preprocess_queue_.push(std::move(f));
            preprocess_queue_.stop();
            break;
        }
        auto t0 = std::chrono::steady_clock::now();
        f.model_input = utils::alignedAlloc(in_size, 64);
        pp.preprocess(f, f.model_input, f.scale, f.dx, f.dy);
        auto t1 = std::chrono::steady_clock::now();
        preprocess_queue_.push(std::move(f));
        {
            std::lock_guard<std::mutex> lk(metrics_mutex_);
            pp_lat_.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0);
        }
    }
}

/**
 * @brief Scheduler dispatches preprocessed frames to available inference queues.
 *
 * Implements backpressure monitoring and drop policy based on queue depth.
 */
void Pipeline::schedulerThread() {
    // Simple pass-through scheduler for now
    while (running_) {
        Frame f; if (!preprocess_queue_.pop(f)) break;
        if (f.eos) {
            // Push EOS to all inference workers
            for (int i = 0; i < std::max(1, config_.nn_workers); ++i) {
                inference_queue_.push(f);
            }
            inference_queue_.stop();
            break;
        }
        inference_queue_.push(std::move(f));
    }
}

/**
 * @brief Inference worker runs CSI-NN2 on assigned frames and records latency.
 *
 * Each worker maintains its own engine instance and updates worker_busy_ns_.
 */
void Pipeline::inferenceWorker(int worker_id) {
    // Bind to nn_cpus if provided
    setCPUAffinity(config_.nn_cpus);
    IEngine* engine = engines_[worker_id].get();
    while (running_) {
        Frame f;
        if (!inference_queue_.pop(f)) break;
        if (f.eos) { ProcessedFrame pf; pf.frame = std::move(f); postprocess_queue_.push(std::move(pf)); postprocess_queue_.stop(); break; }
        auto t0 = std::chrono::steady_clock::now();
        std::vector<Detection> det = engine->infer(f.model_input);
        auto t1 = std::chrono::steady_clock::now();
        utils::alignedFree(f.model_input); f.model_input = nullptr;
        // Rescale detections
        for (auto& d : det) {
            d.x1 = (d.x1 - f.dx) / f.scale;
            d.y1 = (d.y1 - f.dy) / f.scale;
            d.x2 = (d.x2 - f.dx) / f.scale;
            d.y2 = (d.y2 - f.dy) / f.scale;
        }
        ProcessedFrame pf; pf.frame = std::move(f); pf.detections = std::move(det);
        pf.inference_start = t0; pf.inference_end = t1;
        postprocess_queue_.push(std::move(pf));
        {
            std::lock_guard<std::mutex> lk(metrics_mutex_);
            double d = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
            inf_lat_.push_back(d);
            // per-worker
            auto& hist = inf_hist_[worker_id];
            hist.push_back(d);
            if (hist.size() > 64) hist.erase(hist.begin());
            worker_busy_ns_[worker_id] += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        }
    }
}

/**
 * @brief Postprocess thread rescales detections and feeds reorderer/overlay queue.
 *
 * Handles frame drops, updates metrics accumulators, and preserves frame ids.
 */
void Pipeline::postprocessThread() {
    while (running_) {
        ProcessedFrame pf; if (!postprocess_queue_.pop(pf)) break;
        if (pf.frame.eos) {
            if (reorderer_) reorderer_->stop();
            overlay_queue_.stop();
            running_.store(false);
            metrics_cv_.notify_all();
            break;
        }
        auto t0 = std::chrono::steady_clock::now();
        // Postprocess is minimal here; measure routing overhead
        reorderer_->addFrame(pf);
        auto t1 = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lk(metrics_mutex_);
            post_lat_.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0);
        }
    }
}

/**
 * @brief Overlay thread draws detection boxes and prepares frames for display/output.
 *
 * Writes annotated frames into overlay_queue_ and triggers display probe snapshot.
 */
void Pipeline::overlayThread() {
    while (running_) {
        ProcessedFrame pf; if (!reorderer_->getNextFrame(pf)) break;
        // Draw boxes
        extern void draw_detections(cv::Mat& frame_bgr, const std::vector<Detection>& dets);
        auto t0 = std::chrono::steady_clock::now();
        draw_detections(pf.frame.image, pf.detections);
        auto t1 = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lk(metrics_mutex_);
            overlay_lat_.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0);
        }
        overlay_queue_.push(std::move(pf));
    }
}

/**
 * @brief Output thread handles encoder writes and display present operations.
 *
 * Maintains watchdog timestamps and flushes metrics for display latency.
 */
void Pipeline::outputThread() {
    auto state = get_state(this);
    const DisplayConfig disp_cfg{capture_->getWidth(), capture_->getHeight(), "YOLOv5n", config_.sdl_driver};
    auto make_display = [&](const std::string& mode) -> std::shared_ptr<IDisplay> {
        std::unique_ptr<IDisplay> raw;
        if (mode == "sdl") {
            raw = createDisplay("sdl", config_.sdl_driver);
        } else {
            raw = createNullDisplay();
        }
        if (!raw) return nullptr;
        if (!raw->init(disp_cfg)) return nullptr;
        return std::shared_ptr<IDisplay>(std::move(raw));
    };
    std::shared_ptr<IDisplay> display = make_display(config_.display_mode == "sdl" ? "sdl" : "off");
    if (!display && config_.display_mode == "sdl") {
        std::cerr << "[display] SDL unavailable, using null renderer" << std::endl;
        display = make_display("off");
    }
    if (state && display) {
        {
            std::lock_guard<std::mutex> lk(state->display_mutex);
            state->display = display;
            state->driver_name = display->driverName();
        }
        state->last_present_ns.store(now_ns());
    }
    FFmpegEncoder encoder;
    bool enc_ok = false;
    bool encoder_error_reported = false;
    if (!config_.output_path.empty()) {
        enc_ok = encoder.open(config_.output_path, config_.encoder, capture_->getWidth(), capture_->getHeight(), (int)capture_->getFPS());
        if (!enc_ok) std::cerr << "[WARN] Encoder open failed; proceeding without file output" << std::endl;
    }
    uint64_t processed = 0;
    while (running_) {
        ProcessedFrame pf; if (!overlay_queue_.pop(pf)) break;
        PerfMetrics metrics_snapshot{};
        bool metrics_valid = false;
        {
            std::lock_guard<std::mutex> lk(metrics_mutex_);
            metrics_snapshot = current_metrics_;
            metrics_valid = true;
        }
        DisplayFrameInfo frame_info{pf.frame.image, pf.frame.frame_id,
                                    metrics_valid ? &metrics_snapshot : nullptr,
                                    metrics_valid};
        auto disp_start = std::chrono::steady_clock::now();
        bool keep_running = true;
        if (display) {
            keep_running = display->present(frame_info);
        }
        auto disp_end = std::chrono::steady_clock::now();
        double disp_ms = std::chrono::duration_cast<std::chrono::microseconds>(disp_end - disp_start).count() / 1000.0;
        if (!keep_running) {
            if (state && display) {
                state->last_present_ns.store(to_ns(display->lastPresentMono()));
            }
            stop();
            break;
        }
        if (state) {
            if (display) {
                state->last_present_ns.store(to_ns(display->lastPresentMono()));
                if (state->driver_name != "null") {
                    save_display_probe(state, pf.frame.image);
                }
            }
            {
                std::lock_guard<std::mutex> lat_lk(state->lat_mutex);
                state->display_lat.push_back(disp_ms);
            }
        }
        if (enc_ok) {
            auto t0 = std::chrono::steady_clock::now();
            bool write_ok = encoder.write(pf.frame.image);
            auto t1 = std::chrono::steady_clock::now();
            if (!write_ok) {
                if (!encoder_error_reported) {
                    std::cerr << "[ERROR] encoder: write failed, disabling file output" << std::endl;
                    encoder_error_reported = true;
                }
                enc_ok = false;
            } else {
                std::lock_guard<std::mutex> lk(metrics_mutex_);
                enc_lat_.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0);
            }
        }
        ++processed;
        {
            std::lock_guard<std::mutex> lk(metrics_mutex_);
            out_count_++;
            if (inflight_ > 0) inflight_--;
        }
        // Finish when EOF reached and no inflight frames
        if (capture_eof_.load() && inflight_.load() == 0) {
            running_.store(false);
            if (reorderer_) reorderer_->stop();
            overlay_queue_.stop();
            capture_queue_.stop();
            preprocess_queue_.stop();
            inference_queue_.stop();
            postprocess_queue_.stop();
            metrics_cv_.notify_all();
            break;
        }
        if (config_.max_frames > 0 && out_count_ >= (uint64_t)config_.max_frames) {
            running_.store(false);
            if (reorderer_) reorderer_->stop();
            overlay_queue_.stop();
            postprocess_queue_.stop();
            inference_queue_.stop();
            preprocess_queue_.stop();
            capture_queue_.stop();
            metrics_cv_.notify_all();
            break;
        }
    }
    encoder.close();
    if (display) display->close();
}

static double percentile(std::vector<double>& v, double p) {
    if (v.empty()) return 0.0; std::sort(v.begin(), v.end()); size_t idx = (size_t)(std::clamp(p, 0.0, 1.0) * (v.size() - 1)); return v[idx];
}

/**
 * @brief Metrics thread samples latency counters and emits JSONL records.
 *
 * Runs at --perf-interval cadence and resets worker utilization windows.
 */
void Pipeline::metricsThread() {
    auto last = std::chrono::steady_clock::now();
    uint64_t last_in = 0, last_out = 0;
    const auto interval_ms = std::max(1, config_.perf_interval_ms);
    for (;;) {
        std::unique_lock<std::mutex> lk_wait(metrics_cv_mu_);
        bool should_exit = !running_.load();
        if (!should_exit) {
            metrics_cv_.wait_for(lk_wait, std::chrono::milliseconds(interval_ms), [&]{ return !running_.load(); });
            should_exit = !running_.load();
        }
        lk_wait.unlock();

        auto now = std::chrono::steady_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count();
        last = now;
        PerfMetrics m{};
        auto display_state = get_state(this);
        std::vector<double> display_lat_samples;
        if (display_state) {
            std::lock_guard<std::mutex> lat_lk(display_state->lat_mutex);
            display_lat_samples.swap(display_state->display_lat);
        }
        m.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        {
            std::lock_guard<std::mutex> lk(metrics_mutex_);
            double in_count = in_count_ - last_in; last_in = in_count_;
            double out_count = out_count_ - last_out; last_out = out_count_;
            m.input_fps = (float)(in_count * 1000.0 / std::max(1.0, ms));
            m.output_fps = (float)(out_count * 1000.0 / std::max(1.0, ms));
            const uint64_t in_total = in_count_;
            const uint64_t out_total = out_count_;
            float drop_pct = 0.0f;
            if (in_total > 0 && out_total <= in_total) {
                drop_pct = (float)((in_total - out_total) * 100.0 / (double)in_total);
            }
            m.drop_percentage = drop_pct;
            if (out_total <= in_total) {
                dropped_frames_.store(in_total - out_total);
            }
            m.latency_ms.capture = (float)percentile(cap_lat_, 0.5); cap_lat_.clear();
            m.latency_ms.preprocess = (float)percentile(pp_lat_, 0.5); pp_lat_.clear();
            std::vector<double> inf_all;
            for (auto& h : inf_hist_) inf_all.insert(inf_all.end(), h.begin(), h.end());
            m.latency_ms.inference_p50 = (float)percentile(inf_all, 0.5);
            m.latency_ms.inference_p95 = (float)percentile(inf_all, 0.95);
            m.latency_ms.postprocess = (float)percentile(post_lat_, 0.5); post_lat_.clear();
            m.latency_ms.overlay = (float)percentile(overlay_lat_, 0.5); overlay_lat_.clear();
            m.latency_ms.encode = (float)percentile(enc_lat_, 0.5); enc_lat_.clear();
            m.latency_ms.display = 0.0f;
            m.queue_sizes = {
                {"cap_pp", (int)capture_queue_.size()},
                {"pp_sched", (int)preprocess_queue_.size()},
                {"sched_inf", (int)inference_queue_.size()},
                {"inf_post", (int)postprocess_queue_.size()},
                {"post_reord", (int)overlay_queue_.size()}
            };
            m.worker_busy_pct.assign(std::max(1, config_.nn_workers), 0.0f);
            if (ms > 0.0) {
                for (size_t i = 0; i < m.worker_busy_pct.size(); ++i) {
                    double pct = (double)worker_busy_ns_[i] / (ms * 1e6) * 100.0;
                    m.worker_busy_pct[i] = (float)pct;
                    worker_busy_ns_[i] = 0;
                }
            }
        }
        if (!display_lat_samples.empty()) {
            m.latency_ms.display = (float)percentile(display_lat_samples, 0.5);
        } else {
            m.latency_ms.display = 0.0f;
        }
        {
            std::lock_guard<std::mutex> lk(metrics_mutex_);
            current_metrics_ = m;
        }
        if (metrics_writer_) metrics_writer_->write(current_metrics_);
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss << std::setprecision(2)
            << "[metrics] in_fps=" << current_metrics_.input_fps
            << " out_fps=" << current_metrics_.output_fps
            << " drop_pct=" << current_metrics_.drop_percentage
            << " cap_ms=" << current_metrics_.latency_ms.capture
            << " pp_ms=" << current_metrics_.latency_ms.preprocess
            << " inf_p50=" << current_metrics_.latency_ms.inference_p50
            << " inf_p95=" << current_metrics_.latency_ms.inference_p95
            << " post_ms=" << current_metrics_.latency_ms.postprocess
            << " ovl_ms=" << current_metrics_.latency_ms.overlay
            << " enc_ms=" << current_metrics_.latency_ms.encode
            << " disp_ms=" << current_metrics_.latency_ms.display;
        std::cout << oss.str() << std::endl;
        if (display_state) {
            display_state->last_metrics_ns.store(now_ns());
            std::shared_ptr<IDisplay> disp_copy;
            {
                std::lock_guard<std::mutex> lk(display_state->display_mutex);
                disp_copy = display_state->display;
            }
            if (disp_copy) {
                disp_copy->updateMetrics(current_metrics_);
            }
        }
        if (should_exit) break;
    }
}

} // namespace yolov5
