#include "pipeline.hpp"
#include "ringlog.hpp"
#include "capture.hpp"
#include "preprocess.hpp"
#include "engine.hpp"
#include "display.hpp"
#include <pthread.h>
#include <sched.h>
#include <iostream>
#include <malloc.h>
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
    std::atomic<int64_t> last_probe_save_ns{0};
    int watchdog_sec{0};
    bool probe_repeat{false};
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

static bool skip_infer() {
    static const bool skip = []() {
        const char* env = std::getenv("SKIP_INFER");
        return env && env[0] != '\0' && env[0] != '0';
    }();
    return skip;
}

static bool skip_postprocess_flag() {
    static const bool skip = []() {
        const char* env = std::getenv("SKIP_POSTPROCESS");
        return env && env[0] != '\0' && env[0] != '0';
    }();
    return skip;
}

static int worker_count(const PipelineConfig& cfg) {
    return std::max(1, cfg.nn_workers);
}

static size_t choose_capacity(const PipelineConfig& cfg, int override_value, size_t live_default) {
    if (override_value > 0) {
        return static_cast<size_t>(override_value);
    }
    if (cfg.latency_mode == LatencyMode::Live) {
        return live_default;
    }
    return static_cast<size_t>(std::max(1, cfg.queue_capacity));
}

static size_t live_capture_capacity(const PipelineConfig&) {
    return 1;
}

static size_t live_preprocess_capacity(const PipelineConfig&) {
    return 1;
}

static size_t live_infer_capacity(const PipelineConfig&) {
    return 1;
}

static size_t live_reorder_capacity(const PipelineConfig&) {
    return 1;
}

static size_t live_overlay_capacity(const PipelineConfig&) {
    return 1;
}

static size_t reorder_backlog_limit(const PipelineConfig& cfg) {
    if (cfg.q_cap_reorder > 0) {
        return static_cast<size_t>(cfg.q_cap_reorder);
    }
    return 1;
}

static void release_frame_resources(Frame& f) {
    if (f.model_input) {
        utils::alignedFree(f.model_input);
        f.model_input = nullptr;
    }
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
    if (state->probe_path.empty()) return;
    const int64_t now = now_ns();
    int64_t last = 0;
    if (state->probe_repeat) {
        last = state->last_probe_save_ns.load();
        if (last != 0 && (now - last) < 1000000000LL) {
            return;
        }
    } else if (state->probe_saved.load()) {
        return;
    }
    if (!ensure_probe_parent_dir(state->probe_path)) {
        return;
    }
    if (!write_probe_ppm(frame, state->probe_path)) {
        std::cerr << "[display] probe snapshot failed for path " << state->probe_path << std::endl;
        return;
    }
    const bool notify = !state->probe_repeat || last == 0;
    state->probe_saved.store(true);
    if (state->probe_repeat) {
        state->last_probe_save_ns.store(now);
    }
    if (notify) {
        std::cout << "[display] probe saved to " << state->probe_path << std::endl;
    }
}

static bool read_self_vm_stats(size_t& rss_kb, size_t& vm_kb) {
    std::ifstream status("/proc/self/status");
    if (!status) {
        return false;
    }
    std::string line;
    bool rss_found = false;
    bool vm_found = false;
    while (std::getline(status, line)) {
        if (!rss_found && line.rfind("VmRSS:", 0) == 0) {
            std::istringstream iss(line);
            std::string key;
            size_t value = 0;
            std::string unit;
            if (iss >> key >> value >> unit) {
                rss_kb = value;
                rss_found = true;
            }
        } else if (!vm_found && line.rfind("VmSize:", 0) == 0) {
            std::istringstream iss(line);
            std::string key;
            size_t value = 0;
            std::string unit;
            if (iss >> key >> value >> unit) {
                vm_kb = value;
                vm_found = true;
            }
        }
        if (rss_found && vm_found) {
            break;
        }
    }
    return rss_found && vm_found;
}

static void ensure_parent_dir(const std::string& path) {
    if (path.empty()) return;
    try {
        std::filesystem::path p(path);
        auto parent = p.parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }
    } catch (const std::exception& ex) {
        std::cerr << "[memlog] failed to create directories for '" << path << "': " << ex.what() << std::endl;
    }
}

} // namespace

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
    FFmpegEncoder() : fmt_(nullptr), oc_(nullptr), st_(nullptr), enc_(nullptr), sws_(nullptr), frame_(nullptr), pkt_(nullptr), opened_(false), drained_pkts_(0), low_latency_(false) {}
    ~FFmpegEncoder() { close(); }

    /**
     * @brief Open encoder with requested codec and container derived from CLI.
     */
    bool open(const std::string& path, const std::string& enc_name, int w, int h, int fps, bool low_latency) {
        output_path_.clear();
        width_ = w; height_ = h; fps_ = fps > 0 ? fps : 30;
        low_latency_ = low_latency;
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
            if (low_latency_) {
                av_opt_set(enc_->priv_data, "preset", "ultrafast", 0);
                av_opt_set(enc_->priv_data, "tune", "zerolatency", 0);
                enc_->max_b_frames = 0;
                enc_->gop_size = 30;
                enc_->refs = 1;
                enc_->has_b_frames = 0;
                enc_->thread_count = 1;
                av_opt_set(enc_->priv_data, "bf", "0", 0);
                av_opt_set(enc_->priv_data, "bframes", "0", 0);
                av_opt_set(enc_->priv_data, "scenecut", "0", 0);
                av_opt_set(enc_->priv_data, "rc-lookahead", "0", 0);
                enc_->flags |= AV_CODEC_FLAG_LOW_DELAY;
            } else {
                av_opt_set(enc_->priv_data, "preset", "veryfast", 0);
            }
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
        if (low_latency_ && enc_->codec_id == AV_CODEC_ID_H264) {
            std::cout << "[INFO] encoder: low-latency h264 preset applied (gop=30, max_b=0, refs=1)" << std::endl;
        }
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
        low_latency_ = false;
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
    bool low_latency_;
    std::string output_path_;
};

// ---------------------- Pipeline implementation ----------------------
/**
 * @brief Construct pipeline, instantiate queues, capture, engines, and reorderer.
 */
Pipeline::Pipeline(const PipelineConfig& cfg)
    : config_(cfg),
      capture_queue_(choose_capacity(cfg, cfg.q_cap_capture, live_capture_capacity(cfg))),
      preprocess_queue_(choose_capacity(cfg, cfg.q_cap_preprocess, live_preprocess_capacity(cfg))),
      inference_queue_(choose_capacity(cfg, cfg.q_cap_infer, live_infer_capacity(cfg))),
      postprocess_queue_(choose_capacity(cfg, cfg.q_cap_reorder, live_reorder_capacity(cfg))),
      overlay_queue_(choose_capacity(cfg, 0, live_overlay_capacity(cfg))),
      output_queue_(choose_capacity(cfg, 0, live_overlay_capacity(cfg))),
      running_(false), frame_counter_(0), dropped_frames_(0) {
    reorderer_ = std::make_unique<FrameReorderer>();
    reorder_capacity_hint_ = choose_capacity(cfg, cfg.q_cap_reorder, live_reorder_capacity(cfg));
    const size_t backlog_limit = reorder_backlog_limit(cfg);
    reorderer_->configureLive(cfg.latency_mode, cfg.live_ttl_ms, reorder_capacity_hint_, backlog_limit);
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
    if (!capture_) {
        std::cerr << "[ERROR] Capture init failed" << std::endl;
        return false;
    }
    if (auto v4l2 = dynamic_cast<CaptureV4L2*>(capture_.get())) {
        bool live_mode = (config_.latency_mode == LatencyMode::Live);
        int cam_bufs = live_mode ? config_.cam_buffers : 0;
        v4l2->setLowLatencyMode(live_mode, cam_bufs);
    }
    if (!capture_->init(config_.source)) {
        std::cerr << "[ERROR] Capture init failed" << std::endl;
        return false;
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
    try {
        std::string ring_label;
        if (!config_.perf_json_path.empty()) {
            ring_label = std::filesystem::path(config_.perf_json_path).stem().string();
        }
        if (ring_label.empty()) {
            ring_label = (config_.pp_mode == PreprocMode::RVV) ? "live_rvv" : "live_sw";
        }
        ring_set_run_label(ring_label);
    } catch (...) {
        ring_set_run_label("live");
    }
    std::cout << "[INFO] live_ttl_ms=" << config_.live_ttl_ms << std::endl;
    std::cout << "[INFO] nn_workers=" << config_.nn_workers << std::endl;
    drop_cap_.store(0);
    drop_pp_.store(0);
    drop_inf_.store(0);
    drop_post_.store(0);
    drop_display_.store(0);
    live_gate_block_.store(0);
    last_present_ok_.store(true);
    auto state = std::make_shared<DisplayState>();
    state->probe_path = config_.display_probe_path;
    state->watchdog_sec = config_.watchdog_sec;
    state->probe_repeat = (config_.latency_mode == LatencyMode::Live);
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
                    LOG_STAGE(RingStage::WDG_TRIGGER, 0, "watchdog");
                    ring_dump_artifacts("watchdog");
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
    if (!config_.mem_json_path.empty()) {
        ensure_parent_dir(config_.mem_json_path);
        mem_logger_stop_.store(false);
        mem_logger_thread_ = std::thread([this, path = config_.mem_json_path]() {
            std::ofstream ofs(path, std::ios::out | std::ios::trunc);
            if (!ofs) {
                std::cerr << "[memlog] failed to open '" << path << "' for writing" << std::endl;
                return;
            }
            while (!mem_logger_stop_.load()) {
                size_t rss_kb = 0;
                size_t vm_kb = 0;
                if (read_self_vm_stats(rss_kb, vm_kb)) {
                    auto now = std::chrono::system_clock::now();
                    int64_t ts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
                    ofs << "{\"ts_ms\":" << ts_ms
                        << ",\"rss_kb\":" << rss_kb
                        << ",\"vm_kb\":" << vm_kb
                        << ",\"note\":\"rss\"}"
                        << std::endl;
                }
                for (int i = 0; i < 10; ++i) {
                    if (mem_logger_stop_.load()) break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }
        });
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
    mem_logger_stop_.store(true);
    capture_queue_.stop();
    preprocess_queue_.stop();
    inference_queue_.stop();
    postprocess_queue_.stop();
    overlay_queue_.stop();
    output_queue_.stop();
    if (reorderer_) reorderer_->stop();
    metrics_cv_.notify_all();
    ring_dump_artifacts("stop");
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
    join_t(mem_logger_thread_);
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
        const uint64_t fid = f.frame_id;
        if (config_.latency_mode == LatencyMode::Live) {
            while (capture_queue_.full()) {
                Frame dropped;
                if (!capture_queue_.tryPop(dropped)) {
                    break;
                }
                if (reorderer_) {
                    reorderer_->markDropped(dropped.frame_id);
                }
                release_frame_resources(dropped);
                drop_cap_.fetch_add(1, std::memory_order_relaxed);
                LOG_STAGE_F(RingStage::CAP_DROP, dropped.frame_id, 0, "cap_drop");
                if (inflight_.load() > 0) {
                    inflight_--;
                }
            }
        }
        capture_queue_.push(std::move(f));
        LOG_STAGE_F(RingStage::CAP_ENQ, fid, 0, "cap_enq");
        if (config_.log_level == "debug") {
            if (fid % 50 == 0) {
                struct mallinfo2 mi = mallinfo2();
                std::cout << "[heap] stage=capture frame=" << fid
                          << " heap_mb=" << (mi.uordblks / (1024.0 * 1024.0)) << std::endl;
            }
        }
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
        const uint64_t fid = f.frame_id;
        if (config_.latency_mode == LatencyMode::Live) {
            while (preprocess_queue_.full()) {
                Frame dropped;
                if (!preprocess_queue_.tryPop(dropped)) {
                    break;
                }
                release_frame_resources(dropped);
                if (reorderer_) {
                    reorderer_->markDropped(dropped.frame_id);
                }
                drop_pp_.fetch_add(1, std::memory_order_relaxed);
                if (inflight_.load() > 0) {
                    inflight_--;
                }
            }
        }
        preprocess_queue_.push(std::move(f));
        LOG_STAGE_F(RingStage::PP_DONE, fid, 0, "pp_done");
        if (config_.log_level == "debug" && fid % 50 == 0) {
            struct mallinfo2 mi = mallinfo2();
            std::cout << "[heap] stage=preprocess frame=" << fid
                      << " heap_mb=" << (mi.uordblks / (1024.0 * 1024.0)) << std::endl;
        }
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
        if (config_.latency_mode == LatencyMode::Live) {
            while (inference_queue_.full()) {
                Frame dropped;
                if (!inference_queue_.tryPop(dropped)) {
                    break;
                }
                release_frame_resources(dropped);
                if (reorderer_) {
                    reorderer_->markDropped(dropped.frame_id);
                }
                drop_inf_.fetch_add(1, std::memory_order_relaxed);
                if (inflight_.load() > 0) {
                    inflight_--;
                }
            }
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
        LOG_STAGE_F(RingStage::INF_START, f.frame_id, 0, "inf_start");
        auto t0 = std::chrono::steady_clock::now();
        size_t infer_heap_before = 0;
        std::vector<Detection> det;
        if (!skip_infer()) {
            if (config_.log_level == "debug" && f.frame_id % 50 == 0) {
                infer_heap_before = mallinfo2().uordblks;
            }
            det = engine->infer(f.model_input);
            if (config_.log_level == "debug" && f.frame_id % 50 == 0) {
                struct mallinfo2 mi = mallinfo2();
                double delta_mb = (mi.uordblks - infer_heap_before) / (1024.0 * 1024.0);
                std::cout << "[heap] stage=inference_call frame=" << f.frame_id
                          << " delta_mb=" << delta_mb
                          << " heap_mb=" << (mi.uordblks / (1024.0 * 1024.0)) << std::endl;
            }
            if (config_.log_level == "debug" && (f.frame_id % 50 == 0)) {
                malloc_trim(0);
            }
        }
        if (skip_infer() || skip_postprocess_flag()) {
            det.clear();
        }
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
        const uint64_t fid = pf.frame.frame_id;
        LOG_STAGE_F(RingStage::INF_DONE, fid, 0, "inf_done");
        if (config_.latency_mode == LatencyMode::Live) {
            while (postprocess_queue_.full()) {
                ProcessedFrame dropped;
                if (!postprocess_queue_.tryPop(dropped)) {
                    break;
                }
                release_frame_resources(dropped.frame);
                if (reorderer_) {
                    reorderer_->markDropped(dropped.frame.frame_id);
                }
                drop_post_.fetch_add(1, std::memory_order_relaxed);
                if (inflight_.load() > 0) {
                    inflight_--;
                }
            }
        }
        postprocess_queue_.push(std::move(pf));
        if (config_.log_level == "debug" && fid % 50 == 0) {
            struct mallinfo2 mi = mallinfo2();
            std::cout << "[heap] stage=inference frame=" << fid
                      << " heap_mb=" << (mi.uordblks / (1024.0 * 1024.0)) << std::endl;
        }
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
        const uint64_t fid = pf.frame.frame_id;
        reorderer_->addFrame(pf);
        LOG_STAGE_F(RingStage::POST_DONE, fid, 0, "post_done");
        if (config_.log_level == "debug" && fid % 50 == 0) {
            struct mallinfo2 mi = mallinfo2();
            std::cout << "[heap] stage=postprocess frame=" << fid
                      << " heap_mb=" << (mi.uordblks / (1024.0 * 1024.0)) << std::endl;
        }
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
        ProcessedFrame pf;
        bool have_frame = false;
        if (config_.latency_mode == LatencyMode::Live) {
            have_frame = reorderer_->popLatestNonBlocking(pf);
            if (!have_frame) {
                if (!running_.load()) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
        } else {
            if (!reorderer_->getNextFrame(pf)) {
                break;
            }
            have_frame = true;
        }
        if (!have_frame) {
            continue;
        }
        if (config_.latency_mode == LatencyMode::Live) {
            pf.frame.timestamp = std::chrono::steady_clock::now();
        }
        // Draw boxes
        extern void draw_detections(cv::Mat& frame_bgr, const std::vector<Detection>& dets);
        auto t0 = std::chrono::steady_clock::now();
        draw_detections(pf.frame.image, pf.detections);
        auto t1 = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lk(metrics_mutex_);
            overlay_lat_.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0);
        }
        const uint64_t fid = pf.frame.frame_id;
        if (config_.latency_mode == LatencyMode::Live) {
            while (overlay_queue_.full()) {
                ProcessedFrame dropped;
                if (!overlay_queue_.tryPop(dropped)) {
                    break;
                }
                drop_display_.fetch_add(1, std::memory_order_relaxed);
            }
        }
        overlay_queue_.push(std::move(pf));
        if (config_.log_level == "debug" && fid % 50 == 0) {
            struct mallinfo2 mi = mallinfo2();
            std::cout << "[heap] stage=overlay frame=" << fid
                      << " heap_mb=" << (mi.uordblks / (1024.0 * 1024.0)) << std::endl;
        }
    }
}

/**
 * @brief Output thread handles encoder writes and display present operations.
 *
 * Maintains watchdog timestamps and flushes metrics for display latency.
 */
void Pipeline::outputThread() {
    auto state = get_state(this);
    const DisplayConfig disp_cfg{capture_->getWidth(), capture_->getHeight(), "YOLOv5n", config_.sdl_driver, config_.display_vsync};
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
    const bool use_file_output = !config_.output_path.empty() && config_.encoder != "null";
    if (use_file_output) {
        enc_ok = encoder.open(config_.output_path, config_.encoder, capture_->getWidth(), capture_->getHeight(), (int)capture_->getFPS(), config_.latency_mode == LatencyMode::Live);
        if (enc_ok) {
            LOG_STAGE(RingStage::ENC_OPEN, 0, config_.encoder.c_str());
        } else {
            LOG_STAGE(RingStage::ENC_OPEN, -1, "enc_open_fail");
            std::cerr << "[WARN] Encoder open failed; proceeding without file output" << std::endl;
        }
    } else {
        LOG_STAGE(RingStage::ENC_OPEN, 1, "enc_null");
    }
    uint64_t processed = 0;
    while (running_) {
        ProcessedFrame pf;
        bool have_frame = false;
        if (config_.latency_mode == LatencyMode::Live) {
            have_frame = overlay_queue_.tryPop(pf);
            if (!have_frame) {
                if (!running_.load()) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
        } else {
            if (!overlay_queue_.pop(pf)) {
                break;
            }
            have_frame = true;
        }
        if (!have_frame) {
            continue;
        }
        PerfMetrics metrics_snapshot{};
        bool metrics_valid = false;
        {
            std::lock_guard<std::mutex> lk(metrics_mutex_);
            metrics_snapshot = current_metrics_;
            metrics_valid = true;
        }
        bool present_success_backend = true;
        DisplayFrameInfo frame_info{pf.frame.image, pf.frame.frame_id,
                                    metrics_valid ? &metrics_snapshot : nullptr,
                                    metrics_valid,
                                    display ? &present_success_backend : nullptr};
        auto disp_start = std::chrono::steady_clock::now();
        bool keep_running = true;
        if (display) {
            keep_running = display->present(frame_info);
        } else {
            present_success_backend = false;
        }
        auto disp_end = std::chrono::steady_clock::now();
        const double e2e_ms = std::chrono::duration_cast<std::chrono::microseconds>(disp_end - pf.frame.timestamp).count() / 1000.0;
        double disp_ms = std::chrono::duration_cast<std::chrono::microseconds>(disp_end - disp_start).count() / 1000.0;
        if (!keep_running) {
            if (config_.display_allow_null) {
                live_gate_block_.store(0, std::memory_order_relaxed);
                if (state && display) {
                    state->last_present_ns.store(now_ns());
                    std::lock_guard<std::mutex> lk(state->display_mutex);
                    state->display.reset();
                    state->driver_name = "null";
                }
                display.reset();
                present_success_backend = false;
            } else {
                if (state && display) {
                    state->last_present_ns.store(to_ns(display->lastPresentMono()));
                }
                stop();
                break;
            }
        }
        const bool has_real_display = display && display->driverName() != "null";
        bool metrics_presented = has_real_display ? present_success_backend : false;
        if (has_real_display && !present_success_backend) {
            drop_display_.fetch_add(1, std::memory_order_relaxed);
        }
        if (config_.latency_mode == LatencyMode::Live) {
            live_gate_block_.store(0, std::memory_order_relaxed);
        } else {
            live_gate_block_.store(has_real_display && !present_success_backend ? 1 : 0, std::memory_order_relaxed);
        }
        last_present_ok_.store(metrics_presented, std::memory_order_relaxed);
        if (state) {
            if (display && present_success_backend) {
                state->last_present_ns.store(to_ns(display->lastPresentMono()));
                save_display_probe(state, pf.frame.image);
            } else if (config_.display_allow_null && !pf.frame.image.empty()) {
                save_display_probe(state, pf.frame.image);
            }
            {
                std::lock_guard<std::mutex> lat_lk(state->lat_mutex);
                state->display_lat.push_back(disp_ms);
            }
        }
        double enc_sample_ms = 0.0;
        bool enc_sample_valid = false;
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
                LOG_STAGE_F(RingStage::ENC_PKT, pf.frame.frame_id, -1, "enc_write_fail");
            } else {
                LOG_STAGE_F(RingStage::ENC_PKT, pf.frame.frame_id, 0, "enc_pkt");
                enc_sample_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
                enc_sample_valid = true;
            }
        }

        {
            std::lock_guard<std::mutex> lk(metrics_mutex_);
            if (enc_sample_valid) {
                enc_lat_.push_back(enc_sample_ms);
            }
            e2e_lat_.push_back(e2e_ms);
        }
        ++processed;
        if (config_.log_level == "debug" && pf.frame.frame_id % 50 == 0) {
            struct mallinfo2 mi = mallinfo2();
            std::cout << "[heap] stage=output frame=" << pf.frame.frame_id
                      << " heap_mb=" << (mi.uordblks / (1024.0 * 1024.0)) << std::endl;
        }
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
    LOG_STAGE(RingStage::ENC_CLOSE, 0, "enc_close");
    if (enc_ok) {
        encoder.close();
    }
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
        size_t reorder_pending = 0;
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
            double e2e_p50 = percentile(e2e_lat_, 0.5);
            double e2e_p95 = percentile(e2e_lat_, 0.95);
            e2e_lat_.clear();
            m.e2e_ms.p50 = (float)e2e_p50;
            m.e2e_ms.p95 = (float)e2e_p95;
            reorder_pending = reorderer_ ? reorderer_->pendingCount() : 0;
            int q_cap = (int)capture_queue_.size();
            int q_pp = (int)preprocess_queue_.size();
            int q_inf = (int)inference_queue_.size();
            int q_post = (int)postprocess_queue_.size();
            int q_overlay = (int)overlay_queue_.size();
            m.queue_sizes = {
                {"cap_pp", q_cap},
                {"pp_sched", q_pp},
                {"sched_inf", q_inf},
                {"inf_post", q_post},
                {"post_reord", q_overlay},
                {"reorder_buf", (int)reorder_pending},
                {"q_cap", q_cap},
                {"q_pp", q_pp},
                {"q_inf", q_inf},
                {"q_post", q_post},
                {"q_ord", (int)reorder_pending}
            };
            m.queue_capacity = {
                {"cap_pp", static_cast<int>(capture_queue_.capacity())},
                {"pp_sched", static_cast<int>(preprocess_queue_.capacity())},
                {"sched_inf", static_cast<int>(inference_queue_.capacity())},
                {"inf_post", static_cast<int>(postprocess_queue_.capacity())},
                {"post_reord", static_cast<int>(overlay_queue_.capacity())},
                {"reorder_buf", static_cast<int>(reorder_capacity_hint_)},
                {"q_cap", static_cast<int>(capture_queue_.capacity())},
                {"q_pp", static_cast<int>(preprocess_queue_.capacity())},
                {"q_inf", static_cast<int>(inference_queue_.capacity())},
                {"q_post", static_cast<int>(postprocess_queue_.capacity())},
                {"q_ord", static_cast<int>(reorder_capacity_hint_)}
            };
            uint64_t drop_cap = drop_cap_.load(std::memory_order_relaxed);
            uint64_t drop_pp = drop_pp_.load(std::memory_order_relaxed);
            uint64_t drop_inf = drop_inf_.load(std::memory_order_relaxed);
            uint64_t drop_post = drop_post_.load(std::memory_order_relaxed);
            uint64_t drop_disp = drop_display_.load(std::memory_order_relaxed);
            uint64_t reorder_drop_backlog = reorderer_ ? reorderer_->dropBacklogCount() : 0;
            uint64_t reorder_drop_ttl = reorderer_ ? reorderer_->dropTTLCount() : 0;
            uint64_t reorder_gap_skips = reorderer_ ? reorderer_->gapSkipCount() : 0;
            m.drop_counts = {
                {"drops_cap", drop_cap},
                {"drops_pp", drop_pp},
                {"drops_inf", drop_inf},
                {"drops_post", drop_post},
                {"drops_ord", reorder_drop_backlog},
                {"drops_ttl", reorder_drop_ttl},
                {"drops_disp", drop_disp}
            };
            m.drop_backlog = drop_cap + drop_pp + drop_inf + drop_post + reorder_drop_backlog + drop_disp;
            m.drop_ttl = reorder_drop_ttl;
            m.gap_skips = reorder_gap_skips;
            m.ttl_drops = reorder_drop_ttl;
            m.live_gate_block = live_gate_block_.load(std::memory_order_relaxed);
            m.reorder_backlog_max = reorderer_ ? reorderer_->backlogMax() : 0;
            m.display_presented = last_present_ok_.load(std::memory_order_relaxed);
            struct mallinfo2 mi = mallinfo2();
            m.heap_bytes = mi.uordblks;
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
        auto qval = [&](const char* k) -> int {
            auto it = current_metrics_.queue_sizes.find(k);
            return (it == current_metrics_.queue_sizes.end()) ? 0 : it->second;
        };
        auto qcap = [&](const char* k) -> int {
            auto it = current_metrics_.queue_capacity.find(k);
            return (it == current_metrics_.queue_capacity.end()) ? 0 : it->second;
        };
        auto dropval = [&](const char* k) -> uint64_t {
            auto it = current_metrics_.drop_counts.find(k);
            return (it == current_metrics_.drop_counts.end()) ? 0ULL : it->second;
        };
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss << std::setprecision(2)
            << "[metrics] in_fps=" << current_metrics_.input_fps
            << " out_fps=" << current_metrics_.output_fps
            << " e2e_p50=" << current_metrics_.e2e_ms.p50
            << " e2e_p95=" << current_metrics_.e2e_ms.p95
            << " drop_pct=" << current_metrics_.drop_percentage
            << " cap_ms=" << current_metrics_.latency_ms.capture
            << " pp_ms=" << current_metrics_.latency_ms.preprocess
            << " inf_p50=" << current_metrics_.latency_ms.inference_p50
            << " inf_p95=" << current_metrics_.latency_ms.inference_p95
            << " post_ms=" << current_metrics_.latency_ms.postprocess
            << " ovl_ms=" << current_metrics_.latency_ms.overlay
            << " enc_ms=" << current_metrics_.latency_ms.encode
            << " q_cap=" << qval("q_cap") << '/' << qcap("q_cap")
            << " q_pp=" << qval("q_pp") << '/' << qcap("q_pp")
            << " q_inf=" << qval("q_inf") << '/' << qcap("q_inf")
            << " q_ord=" << qval("q_ord") << '/' << qcap("q_ord")
            << " drops_cap=" << dropval("drops_cap")
            << " drops_pp=" << dropval("drops_pp")
            << " drops_inf=" << dropval("drops_inf")
            << " drops_post=" << dropval("drops_post")
            << " drops_ord=" << dropval("drops_ord")
            << " drops_ttl=" << dropval("drops_ttl")
            << " drops_disp=" << dropval("drops_disp")
            << " drop_total=" << current_metrics_.drop_backlog
            << " drop_ttl=" << current_metrics_.drop_ttl
            << " present=" << (current_metrics_.display_presented ? 1 : 0)
            << " backlog_max=" << current_metrics_.reorder_backlog_max
            << " heap_mb=" << (current_metrics_.heap_bytes / (1024.0 * 1024.0))
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
