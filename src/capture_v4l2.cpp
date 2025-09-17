#include "capture.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <sstream>

/**
 * @file capture_v4l2.cpp
 * @brief V4L2 capture backend handling device scan, format negotiation, and streaming.
 */

#include <dirent.h>
#include <fcntl.h>
#include <poll.h>
#include <limits.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <linux/videodev2.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/log.h>
#include <libavutil/error.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}

namespace yolov5 {
namespace {
/** @brief Preferred camera payload when parsing CLI query string. */
enum class CamFormatPreference {
    Auto,
    YUYV,
    MJPEG
};

constexpr int kDefaultWidth = 1280;
constexpr int kDefaultHeight = 720;
constexpr int kDefaultFPS = 30;
constexpr int kBufferCount = 4;

/** @brief Lowercase helper used for query string parsing. */
std::string toLowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return value;
}

/**
 * @brief Parse CLI query string to select desired camera format.
 * @param opts Query string after '?'.
 * @param current Existing preference which acts as default.
 */
CamFormatPreference parseFormatPreference(const std::string& opts, CamFormatPreference current) {
    if (opts.empty()) return current;
    std::stringstream ss(opts);
    std::string kv;
    while (std::getline(ss, kv, '&')) {
        auto eq = kv.find('=');
        if (eq == std::string::npos) continue;
        std::string key = toLowerCopy(kv.substr(0, eq));
        std::string val = toLowerCopy(kv.substr(eq + 1));
        if (key == "fmt" || key == "format" || key == "cam_fmt") {
            if (val == "yuyv" || val == "yuyv422") {
                return CamFormatPreference::YUYV;
            }
            if (val == "mjpeg" || val == "jpeg") {
                return CamFormatPreference::MJPEG;
            }
            return CamFormatPreference::Auto;
        }
    }
    return current;
}

/** @brief Convert V4L2 fourcc to printable string. */
std::string fourccToString(uint32_t fourcc) {
    char buf[5];
    buf[0] = static_cast<char>(fourcc & 0xFF);
    buf[1] = static_cast<char>((fourcc >> 8) & 0xFF);
    buf[2] = static_cast<char>((fourcc >> 16) & 0xFF);
    buf[3] = static_cast<char>((fourcc >> 24) & 0xFF);
    buf[4] = '\0';
    return std::string(buf);
}

/** @brief Return true when filesystem path exists. */
bool pathExists(const std::string& path) {
    struct stat st{};
    return stat(path.c_str(), &st) == 0;
}

/** @brief Resolve symlinks to canonical device path. */
std::string canonicalPath(const std::string& path) {
    char resolved[PATH_MAX];
    if (realpath(path.c_str(), resolved)) {
        return std::string(resolved);
    }
    return {};
}

/**
 * @brief Convert V4L2 buffer timestamp to steady_clock when possible.
 *
 * Uses monotonic timestamp when driver exposes it; otherwise falls back to now.
 */
std::chrono::steady_clock::time_point bufferTimestamp(const v4l2_buffer& buf) {
#ifdef V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC
    if ((buf.flags & V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC) && (buf.timestamp.tv_sec || buf.timestamp.tv_usec)) {
        timespec mono_ts{};
        if (clock_gettime(CLOCK_MONOTONIC, &mono_ts) == 0) {
            auto now = std::chrono::steady_clock::now();
            auto buf_ns = std::chrono::seconds(buf.timestamp.tv_sec) +
                          std::chrono::nanoseconds(buf.timestamp.tv_usec * 1000LL);
            auto mono_ns = std::chrono::seconds(mono_ts.tv_sec) + std::chrono::nanoseconds(mono_ts.tv_nsec);
            if (buf_ns > mono_ns) {
                return now;
            }
            auto delta = mono_ns - buf_ns;
            auto delta_steady = std::chrono::duration_cast<std::chrono::steady_clock::duration>(delta);
            return now - delta_steady;
        }
    }
#endif
    return std::chrono::steady_clock::now();
}

/**
 * @brief Group camera nodes by persistent /dev/v4l/by-id symlinks.
 *
 * Ensures stereo devices expose both index0/index1 variants once.
 */
std::vector<std::vector<std::string>> enumerateById(std::set<std::string>& used_real) {
    std::vector<std::vector<std::string>> groups;
    DIR* dir = opendir("/dev/v4l/by-id");
    if (!dir) {
        return groups;
    }
    std::map<std::string, std::vector<std::pair<int, std::string>>> grouped;
    if (dirent* ent; true) {
        while ((ent = readdir(dir)) != nullptr) {
            if (!ent->d_name || ent->d_name[0] == '.') {
                continue;
            }
            std::string entry = ent->d_name;
            std::string full = std::string("/dev/v4l/by-id/") + entry;
            if (!pathExists(full)) {
                continue;
            }
            std::string base = entry;
            int idx = -1;
            auto pos = base.rfind("-index");
            if (pos != std::string::npos) {
                std::string suffix = base.substr(pos + 6);
                try {
                    idx = std::stoi(suffix);
                } catch (...) {
                    idx = -1;
                }
                base = base.substr(0, pos);
            }
            grouped[base].push_back({idx, full});
        }
    }
    closedir(dir);

    for (auto& kv : grouped) {
        auto& candidates = kv.second;
        std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
            if (a.first == b.first) {
                return a.second < b.second;
            }
            if (a.first < 0) {
                return false;
            }
            if (b.first < 0) {
                return true;
            }
            return a.first < b.first;
        });
        std::vector<std::string> group;
        for (const auto& item : candidates) {
            if (!pathExists(item.second)) {
                continue;
            }
            std::string real = canonicalPath(item.second);
            if (!real.empty()) {
                used_real.insert(real);
            }
            group.push_back(item.second);
        }
        if (!group.empty()) {
            groups.push_back(std::move(group));
        }
    }
    return groups;
}

/**
 * @brief Enumerate /dev/video* nodes and pair possible stereo siblings.
 */
std::vector<std::vector<std::string>> enumerateVideoNodes(std::set<std::string>& used_real) {
    std::vector<std::vector<std::string>> groups;
    std::set<int> skip_indices;
    for (int i = 0; i < 64; ++i) {
        if (skip_indices.count(i)) {
            continue;
        }
        std::string node = std::string("/dev/video") + std::to_string(i);
        if (!pathExists(node)) {
            continue;
        }
        std::string real = canonicalPath(node);
        if (!real.empty() && used_real.count(real)) {
            continue;
        }
        std::vector<std::string> group;
        group.push_back(node);
        if (!real.empty()) {
            used_real.insert(real);
        }
        int twin_idx = i ^ 1;
        if (twin_idx > i && twin_idx < 64) {
            std::string twin = std::string("/dev/video") + std::to_string(twin_idx);
            if (pathExists(twin)) {
                std::string twin_real = canonicalPath(twin);
                if (twin_real.empty() || !used_real.count(twin_real)) {
                    group.push_back(twin);
                    if (!twin_real.empty()) {
                        used_real.insert(twin_real);
                    }
                    skip_indices.insert(twin_idx);
                }
            }
        }
        groups.push_back(std::move(group));
    }
    return groups;
}

/**
 * @brief Build candidate list from explicit CLI device specification.
 */
std::vector<std::string> buildExplicitGroup(const std::string& spec) {
    std::vector<std::string> group;
    if (spec.empty()) {
        return group;
    }
    if (!pathExists(spec)) {
        return group;
    }
    group.push_back(spec);
    auto addCandidate = [&](const std::string& candidate) {
        if (!candidate.empty() && candidate != spec && pathExists(candidate)) {
            group.push_back(candidate);
        }
    };
    auto idx_pos = spec.rfind("-index");
    if (idx_pos != std::string::npos && idx_pos + 6 < spec.size()) {
        std::string suffix = spec.substr(idx_pos + 6);
        if (suffix == "0") {
            addCandidate(spec.substr(0, idx_pos) + "-index1");
        } else if (suffix == "1") {
            addCandidate(spec.substr(0, idx_pos) + "-index0");
        }
    } else if (spec.rfind("/dev/video", 0) == 0) {
        const std::string prefix = "/dev/video";
        int idx = -1;
        try {
            idx = std::stoi(spec.substr(prefix.size()));
        } catch (...) {
            idx = -1;
        }
        if (idx >= 0) {
            int twin_idx = idx ^ 1;
            if (twin_idx >= 0 && twin_idx < 64) {
                addCandidate(prefix + std::to_string(twin_idx));
            }
        }
    }
    return group;
}

/**
 * @brief Resolve CLI source into ordered list of device candidates.
 */
std::vector<std::vector<std::string>> buildCandidates(const std::string& source) {
    std::string spec = source;
    if (spec.rfind("v4l2:", 0) == 0) {
        spec = spec.substr(5);
    }
    while (spec.rfind("//", 0) == 0) {
        spec = spec.substr(2);
    }
    if (spec == "auto") {
        std::set<std::string> used_real;
        std::vector<std::vector<std::string>> groups;
        auto by_id = enumerateById(used_real);
        groups.insert(groups.end(), by_id.begin(), by_id.end());
        auto fallback = enumerateVideoNodes(used_real);
        groups.insert(groups.end(), fallback.begin(), fallback.end());
        return groups;
    }
    std::vector<std::vector<std::string>> groups;
    auto explicit_group = buildExplicitGroup(spec);
    if (!explicit_group.empty()) {
        groups.push_back(std::move(explicit_group));
    }
    return groups;
}

} // namespace

/**
 * @brief Encapsulates V4L2 device enumeration, format negotiation, and streaming.
 *
 * All members are used solely by the capture thread. The class manages kernel
 * buffers (mmap), optional MJPEG decoder, and swscale contexts to normalize
 * camera payloads into BGR Mats.
 */
class CaptureV4L2::Impl {
public:
    struct Buffer {
        void* start = nullptr;
        size_t length = 0;
    };

    Impl() : fd_(-1), streaming_(false), width_(kDefaultWidth), height_(kDefaultHeight),
             fps_(kDefaultFPS), frame_counter_(0), sws_ctx_(nullptr), selected_device_(),
             stride_(kDefaultWidth * 2), requested_format_(CamFormatPreference::Auto),
             active_format_fourcc_(0), mjpeg_decoder_ctx_(nullptr), mjpeg_frame_(nullptr) {}

    ~Impl() {
        release();
    }

    /**
     * @brief Probe and start V4L2 device matching CLI source string.
     *
     * Iterates candidate nodes until format negotiation and streaming succeed,
     * respecting requested fmt query (auto/yuyv/mjpeg).
     */
    bool init(const std::string& source) {
        av_log_set_level(AV_LOG_ERROR);
        release();
        frame_counter_ = 0;
        width_ = kDefaultWidth;
        height_ = kDefaultHeight;
        fps_ = kDefaultFPS;

        std::string base = source;
        std::string opts;
        auto qpos = base.find('?');
        if (qpos != std::string::npos) {
            opts = base.substr(qpos + 1);
            base = base.substr(0, qpos);
        }
        requested_format_ = CamFormatPreference::Auto;
        requested_format_ = parseFormatPreference(opts, requested_format_);
        active_format_fourcc_ = 0;
        std::string source_base = base;

        auto candidates = buildCandidates(source_base);
        if (candidates.empty()) {
            std::cerr << "[ERROR] v4l2: no candidates for source '" << source << "'" << std::endl;
            return false;
        }

        for (const auto& group : candidates) {
            for (const auto& dev : group) {
                if (!pathExists(dev)) {
                    continue;
                }
                if (tryStartDevice(dev)) {
                    std::string real = canonicalPath(dev);
                    selected_device_ = real.empty() ? dev : real;
                    std::cout << std::fixed << std::setprecision(2)
                              << "[INFO] v4l2: using " << selected_device_
                              << " " << width_ << "x" << height_
                              << " @ " << fps_ << " fps"
                              << " fmt=" << fourccToString(active_format_fourcc_) << std::endl;
                    return true;
                }
            }
        }
        std::cerr << "[ERROR] v4l2: no working capture device found" << std::endl;
        return false;
    }

    /**
     * @brief Dequeue next V4L2 buffer and convert into BGR Frame.
     *
     * Handles YUYV via swscale and MJPEG by decoding through FFmpeg to maintain
     * consistent downstream BGR layout.
     */
    bool getFrame(Frame& frame) {
        if (fd_ < 0) {
            return false;
        }
        struct pollfd pfd { fd_, POLLIN, 0 };
        int pret;
        for (;;) {
            pret = poll(&pfd, 1, 500);
            if (pret < 0 && errno == EINTR) {
                continue;
            }
            break;
        }
        if (pret < 0) {
            int err = errno;
            std::cerr << "[ERROR] v4l2: poll failed (" << strerror(err) << ")" << std::endl;
            return false;
        }
        if (pret == 0) {
            return false;
        }
        if (pfd.revents & POLLERR) {
            std::cerr << "[ERROR] v4l2: poll reported error" << std::endl;
            return false;
        }
        if (!(pfd.revents & POLLIN)) {
            return false;
        }

        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if (ioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
            if (errno == EAGAIN) {
                return false;
            }
            int err = errno;
            std::cerr << "[ERROR] v4l2: VIDIOC_DQBUF failed (" << strerror(err) << ")" << std::endl;
            return false;
        }
        if (buf.index >= buffers_.size()) {
            std::cerr << "[ERROR] v4l2: buffer index out of range" << std::endl;
            ioctl(fd_, VIDIOC_QBUF, &buf);
            return false;
        }
        uint8_t* src = static_cast<uint8_t*>(buffers_[buf.index].start);
        if (!src) {
            std::cerr << "[ERROR] v4l2: buffer pointer null" << std::endl;
            ioctl(fd_, VIDIOC_QBUF, &buf);
            return false;
        }

        frame.format = PixelFormat::BGR;
        frame.y_plane.clear();
        frame.u_plane.clear();
        frame.v_plane.clear();
        frame.y_stride = 0;
        frame.uv_stride = 0;

        static bool logged_dimensions = false;
        if (!logged_dimensions) {
            std::cerr << "[INFO] v4l2: format=" << fourccToString(active_format_fourcc_)
                      << " w=" << width_ << " h=" << height_
                      << " stride=" << stride_
                      << " bytes=" << buf.bytesused << std::endl;
            logged_dimensions = true;
        }

        if (active_format_fourcc_ == V4L2_PIX_FMT_YUYV) {
            frame.image.create(height_, width_, CV_8UC3);
            const uint8_t* src_planes[4] = { src, nullptr, nullptr, nullptr };
            const int src_strides[4] = { stride_, 0, 0, 0 };
            uint8_t* dst_planes[4] = { frame.image.data, nullptr, nullptr, nullptr };
            const int dst_strides[4] = { static_cast<int>(frame.image.step[0]), 0, 0, 0 };

            sws_ctx_ = sws_getCachedContext(sws_ctx_, width_, height_, AV_PIX_FMT_YUYV422,
                                            width_, height_, AV_PIX_FMT_BGR24,
                                            SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
            if (!sws_ctx_) {
                std::cerr << "[ERROR] v4l2: sws_getCachedContext failed" << std::endl;
                ioctl(fd_, VIDIOC_QBUF, &buf);
                return false;
            }
            sws_scale(sws_ctx_, src_planes, src_strides, 0, height_, dst_planes, dst_strides);
        } else if (active_format_fourcc_ == V4L2_PIX_FMT_MJPEG) {
            if (!decodeMJPEG(src, buf.bytesused, frame)) {
                ioctl(fd_, VIDIOC_QBUF, &buf);
                return false;
            }
        } else {
            std::cerr << "[ERROR] v4l2: unsupported active pixel format" << std::endl;
            ioctl(fd_, VIDIOC_QBUF, &buf);
            return false;
        }

        frame.frame_id = frame_counter_++;
        frame.timestamp = bufferTimestamp(buf);
        frame.source_width = width_;
        frame.source_height = height_;

        if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
            int err = errno;
            std::cerr << "[ERROR] v4l2: VIDIOC_QBUF failed (" << strerror(err) << ")" << std::endl;
            return false;
        }
        return true;
    }

    /**
     * @brief Stop streaming and release all device resources.
     */
    void release() {
        stopStreaming();
        for (auto& buf : buffers_) {
            if (buf.start) {
                munmap(buf.start, buf.length);
                buf.start = nullptr;
                buf.length = 0;
            }
        }
        buffers_.clear();
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
        if (sws_ctx_) {
            sws_freeContext(sws_ctx_);
            sws_ctx_ = nullptr;
        }
        releaseMJPEGDecoder();
        streaming_ = false;
        selected_device_.clear();
        active_format_fourcc_ = 0;
        requested_format_ = CamFormatPreference::Auto;
    }

    bool isOpened() const { return fd_ >= 0; }

    int getWidth() const { return width_; }
    int getHeight() const { return height_; }
    double getFPS() const { return fps_; }
    int getFrameCount() const { return -1; }

private:
    /**
     * @brief Attempt to open and initialize a specific device node.
     *
     * Negotiates formats, allocates mmap buffers, and enables streaming. On
     * failure the device is closed and the next candidate is tried.
     */
    bool tryStartDevice(const std::string& dev) {
        try {
            fd_ = open(dev.c_str(), O_RDWR | O_NONBLOCK, 0);
            if (fd_ < 0) {
                int err = errno;
                std::cerr << "[ERROR] v4l2: " << dev << ": open failed (" << strerror(err) << ")" << std::endl;
                return false;
            }

            auto fail = [&](const std::string& msg, int err) {
                std::cerr << "[ERROR] v4l2: " << dev << ": " << msg;
                if (err) {
                    std::cerr << " (" << strerror(err) << ")";
                }
                std::cerr << std::endl;
                closeDevice();
                return false;
            };

            struct v4l2_capability cap;
            memset(&cap, 0, sizeof(cap));
            if (ioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0) {
                return fail("VIDIOC_QUERYCAP failed", errno);
            }
            uint32_t caps = cap.capabilities;
            if (caps & V4L2_CAP_DEVICE_CAPS) {
                caps = cap.device_caps;
            }
            bool has_capture = (caps & V4L2_CAP_VIDEO_CAPTURE) != 0;
            bool has_mplane = (caps & V4L2_CAP_VIDEO_CAPTURE_MPLANE) != 0;
            if (!has_capture && !has_mplane) {
                return fail("device missing V4L2_CAP_VIDEO_CAPTURE", 0);
            }
            bool has_stream = (caps & V4L2_CAP_STREAMING) != 0;
#ifdef V4L2_CAP_VIDEO_M2M
            if ((caps & V4L2_CAP_VIDEO_M2M) != 0) {
                return fail("skipping mem2mem device", 0);
            }
#endif
#ifdef V4L2_CAP_VIDEO_M2M_MPLANE
            if ((caps & V4L2_CAP_VIDEO_M2M_MPLANE) != 0) {
                return fail("skipping mem2mem device", 0);
            }
#endif
            if (!has_capture && has_mplane) {
                return fail("mplane capture not supported", 0);
            }
            if (!has_stream) {
                return fail("device missing V4L2_CAP_STREAMING", 0);
            }

            bool supports_yuyv = false;
            bool supports_mjpeg = false;
            struct v4l2_fmtdesc fdesc;
            memset(&fdesc, 0, sizeof(fdesc));
            fdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            while (ioctl(fd_, VIDIOC_ENUM_FMT, &fdesc) == 0) {
                if (fdesc.pixelformat == V4L2_PIX_FMT_YUYV) supports_yuyv = true;
                if (fdesc.pixelformat == V4L2_PIX_FMT_MJPEG) supports_mjpeg = true;
                fdesc.index++;
            }

            uint32_t desired_fourcc = 0;
            if (requested_format_ == CamFormatPreference::YUYV) {
                if (!supports_yuyv) {
                    return fail("requested YUYV but device lacks support", 0);
                }
                desired_fourcc = V4L2_PIX_FMT_YUYV;
            } else if (requested_format_ == CamFormatPreference::MJPEG) {
                if (!supports_mjpeg) {
                    if (supports_yuyv) {
                        std::cerr << "[WARN] v4l2: " << dev << ": requested MJPEG but not available, falling back to YUYV" << std::endl;
                        desired_fourcc = V4L2_PIX_FMT_YUYV;
                    } else {
                        return fail("requested MJPEG but no supported fallback", 0);
                    }
                } else {
                    desired_fourcc = V4L2_PIX_FMT_MJPEG;
                }
            } else {
                if (supports_yuyv) desired_fourcc = V4L2_PIX_FMT_YUYV;
                else if (supports_mjpeg) desired_fourcc = V4L2_PIX_FMT_MJPEG;
            }

            if (desired_fourcc == 0) {
                return fail("device missing YUYV/MJPEG support", 0);
            }

            struct v4l2_format fmt;
            memset(&fmt, 0, sizeof(fmt));
            fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            fmt.fmt.pix.width = width_;
            fmt.fmt.pix.height = height_;
            fmt.fmt.pix.pixelformat = desired_fourcc;
            fmt.fmt.pix.field = V4L2_FIELD_NONE;
            if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
                return fail("VIDIOC_S_FMT failed", errno);
            }
            width_ = fmt.fmt.pix.width;
            height_ = fmt.fmt.pix.height;
            active_format_fourcc_ = fmt.fmt.pix.pixelformat;
            if (desired_fourcc == V4L2_PIX_FMT_MJPEG && active_format_fourcc_ == V4L2_PIX_FMT_YUYV) {
                std::cerr << "[WARN] v4l2: " << dev << ": driver forced YUYV despite MJPEG request" << std::endl;
                desired_fourcc = V4L2_PIX_FMT_YUYV;
            }
            if (active_format_fourcc_ != desired_fourcc) {
                return fail("device refused requested pixel format", 0);
            }
            if (active_format_fourcc_ == V4L2_PIX_FMT_YUYV) {
                stride_ = fmt.fmt.pix.bytesperline ? fmt.fmt.pix.bytesperline : width_ * 2;
                releaseMJPEGDecoder();
            } else if (active_format_fourcc_ == V4L2_PIX_FMT_MJPEG) {
                stride_ = 0;
                if (!initMJPEGDecoder()) {
                    return fail("failed to init MJPEG decoder", 0);
                }
            }

            struct v4l2_streamparm parm;
            memset(&parm, 0, sizeof(parm));
            parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            parm.parm.capture.timeperframe.numerator = 1;
            parm.parm.capture.timeperframe.denominator = kDefaultFPS;
            ioctl(fd_, VIDIOC_S_PARM, &parm);
            if (ioctl(fd_, VIDIOC_G_PARM, &parm) == 0 &&
                parm.parm.capture.timeperframe.numerator > 0 &&
                parm.parm.capture.timeperframe.denominator > 0) {
                fps_ = static_cast<double>(parm.parm.capture.timeperframe.denominator) /
                       parm.parm.capture.timeperframe.numerator;
            } else {
                fps_ = kDefaultFPS;
            }

            struct v4l2_requestbuffers req;
            memset(&req, 0, sizeof(req));
            req.count = kBufferCount;
            req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            req.memory = V4L2_MEMORY_MMAP;
            if (ioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
                return fail("VIDIOC_REQBUFS failed", errno);
            }
            if (req.count == 0) {
                return fail("driver returned zero buffers", 0);
            }
            uint32_t alloc_count = req.count;
            if (alloc_count < 2) alloc_count = 2;
            if (alloc_count > 64) {
                std::cerr << "[WARN] v4l2: " << dev << ": driver requested " << alloc_count
                          << " buffers, clamping to 64" << std::endl;
                alloc_count = 64;
            }
            try {
                buffers_.resize(alloc_count);
            } catch (const std::exception&) {
                return fail("buffer allocation failed", 0);
            }
            for (uint32_t i = 0; i < alloc_count; ++i) {
            struct v4l2_buffer buf;
            memset(&buf, 0, sizeof(buf));
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;
            if (ioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
                return fail("VIDIOC_QUERYBUF failed", errno);
            }
            void* mapped = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);
            if (mapped == MAP_FAILED) {
                return fail("mmap failed", errno);
            }
            buffers_[i].start = mapped;
            buffers_[i].length = buf.length;
            if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
                return fail("VIDIOC_QBUF failed", errno);
            }
        }
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
            return fail("VIDIOC_STREAMON failed", errno);
        }
        streaming_ = true;
        return true;
        } catch (const std::exception& ex) {
            std::cerr << "[ERROR] v4l2: " << dev << ": unexpected exception: " << ex.what() << std::endl;
            closeDevice();
            return false;
        }
    }

    /** @brief Issue STREAMOFF to halt capture and preserve buffers. */
    void stopStreaming() {
        if (fd_ >= 0 && streaming_) {
            enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            ioctl(fd_, VIDIOC_STREAMOFF, &type);
            streaming_ = false;
        }
    }

    /** @brief Close device fd and unmap buffers (idempotent). */
    void closeDevice() {
        stopStreaming();
        for (auto& buf : buffers_) {
            if (buf.start) {
                munmap(buf.start, buf.length);
                buf.start = nullptr;
                buf.length = 0;
            }
        }
        buffers_.clear();
        releaseMJPEGDecoder();
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
        active_format_fourcc_ = 0;
        stride_ = kDefaultWidth * 2;
    }

    bool initMJPEGDecoder();
    void releaseMJPEGDecoder();
    bool decodeMJPEG(void* data, size_t size, Frame& frame);

    int fd_;
    bool streaming_;
    std::vector<Buffer> buffers_;
    int width_;
    int height_;
    double fps_;
    uint64_t frame_counter_;
    SwsContext* sws_ctx_;
    std::string selected_device_;
    int stride_;
    CamFormatPreference requested_format_;
    uint32_t active_format_fourcc_;
    AVCodecContext* mjpeg_decoder_ctx_;
    AVFrame* mjpeg_frame_;
};

/** @brief Prepare FFmpeg MJPEG decoder for camera payloads. */
bool CaptureV4L2::Impl::initMJPEGDecoder() {
    releaseMJPEGDecoder();
    const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_MJPEG);
    if (!codec) {
        std::cerr << "[ERROR] v4l2: MJPEG codec not found" << std::endl;
        return false;
    }
    mjpeg_decoder_ctx_ = avcodec_alloc_context3(codec);
    if (!mjpeg_decoder_ctx_) {
        std::cerr << "[ERROR] v4l2: failed to allocate MJPEG decoder context" << std::endl;
        return false;
    }
    if (avcodec_open2(mjpeg_decoder_ctx_, codec, nullptr) < 0) {
        std::cerr << "[ERROR] v4l2: failed to open MJPEG decoder" << std::endl;
        avcodec_free_context(&mjpeg_decoder_ctx_);
        mjpeg_decoder_ctx_ = nullptr;
        return false;
    }
    mjpeg_frame_ = av_frame_alloc();
    if (!mjpeg_frame_) {
        std::cerr << "[ERROR] v4l2: failed to allocate MJPEG frame" << std::endl;
        releaseMJPEGDecoder();
        return false;
    }
    return true;
}

/** @brief Tear down MJPEG decoder resources. */
void CaptureV4L2::Impl::releaseMJPEGDecoder() {
    if (mjpeg_frame_) {
        av_frame_free(&mjpeg_frame_);
        mjpeg_frame_ = nullptr;
    }
    if (mjpeg_decoder_ctx_) {
        avcodec_free_context(&mjpeg_decoder_ctx_);
        mjpeg_decoder_ctx_ = nullptr;
    }
}

/**
 * @brief Decode MJPEG buffer into BGR Mat using FFmpeg.
 * @param data Pointer to MJPEG payload from kernel buffer.
 * @param size Payload length in bytes.
 * @param frame Frame receiving decoded BGR data.
 */
bool CaptureV4L2::Impl::decodeMJPEG(void* data, size_t size, Frame& frame) {
    if (!mjpeg_decoder_ctx_ || !data || size == 0) {
        std::cerr << "[ERROR] v4l2: MJPEG decoder not ready" << std::endl;
        return false;
    }
    AVPacket* packet = av_packet_alloc();
    if (!packet) {
        std::cerr << "[ERROR] v4l2: failed to allocate AVPacket" << std::endl;
        return false;
    }
    packet->data = static_cast<uint8_t*>(data);
    packet->size = static_cast<int>(size);

    int ret = avcodec_send_packet(mjpeg_decoder_ctx_, packet);
    if (ret < 0) {
        std::cerr << "[ERROR] v4l2: avcodec_send_packet failed" << std::endl;
        av_packet_free(&packet);
        return false;
    }
    ret = avcodec_receive_frame(mjpeg_decoder_ctx_, mjpeg_frame_);
    av_packet_free(&packet);
    if (ret < 0) {
        if (ret != AVERROR(EAGAIN) && ret != AVERROR_EOF) {
            std::cerr << "[ERROR] v4l2: avcodec_receive_frame failed" << std::endl;
        }
        return false;
    }

    frame.image.create(height_, width_, CV_8UC3);
    uint8_t* dst_planes[4] = { frame.image.data, nullptr, nullptr, nullptr };
    const int dst_strides[4] = { static_cast<int>(frame.image.step[0]), 0, 0, 0 };

    sws_ctx_ = sws_getCachedContext(sws_ctx_, mjpeg_frame_->width, mjpeg_frame_->height,
                                    static_cast<AVPixelFormat>(mjpeg_frame_->format),
                                    width_, height_, AV_PIX_FMT_BGR24,
                                    SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!sws_ctx_) {
        std::cerr << "[ERROR] v4l2: sws_getCachedContext failed for MJPEG" << std::endl;
        return false;
    }
    sws_scale(sws_ctx_, mjpeg_frame_->data, mjpeg_frame_->linesize, 0, mjpeg_frame_->height,
              dst_planes, dst_strides);
    av_frame_unref(mjpeg_frame_);
    return true;
}

CaptureV4L2::CaptureV4L2() : pImpl(std::make_unique<Impl>()) {}
CaptureV4L2::~CaptureV4L2() = default;

bool CaptureV4L2::init(const std::string& source) {
    return pImpl->init(source);
}

bool CaptureV4L2::getFrame(Frame& frame) {
    return pImpl->getFrame(frame);
}

void CaptureV4L2::release() {
    pImpl->release();
}

int CaptureV4L2::getWidth() const {
    return pImpl->getWidth();
}

int CaptureV4L2::getHeight() const {
    return pImpl->getHeight();
}

double CaptureV4L2::getFPS() const {
    return pImpl->getFPS();
}

int CaptureV4L2::getFrameCount() const {
    return pImpl->getFrameCount();
}

bool CaptureV4L2::isOpened() const {
    return pImpl->isOpened();
}

/** @brief Factory selecting capture backend by URI scheme. */
std::unique_ptr<ICapture> createCapture(const std::string& source) {
    if (source.rfind("v4l2:", 0) == 0 || source.rfind("/dev/video", 0) == 0 ||
        source.rfind("/dev/v4l/", 0) == 0) {
        return std::make_unique<CaptureV4L2>();
    }
    if (source.rfind("file:", 0) == 0 || source.find('.') != std::string::npos) {
        return std::make_unique<CaptureFile>();
    }
    return nullptr;
}

} // namespace yolov5
