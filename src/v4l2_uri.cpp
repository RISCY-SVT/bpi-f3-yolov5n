#include "v4l2_uri.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <sstream>

namespace yolov5 {
namespace {

inline std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool parse_int(const std::string& text, int min_value, int max_value, int& out) {
    if (text.empty()) {
        return false;
    }
    const char* start = text.c_str();
    char* end = nullptr;
    errno = 0;
    long val = std::strtol(start, &end, 10);
    if (start == end || *end != '\0' || errno == ERANGE) {
        return false;
    }
    if (val < min_value || val > max_value) {
        return false;
    }
    out = static_cast<int>(val);
    return true;
}

void set_error(std::string* error, const std::string& message) {
    if (!error || message.empty()) {
        return;
    }
    if (!error->empty()) {
        error->append("; ");
    }
    error->append(message);
}

} // namespace

V4L2UriOptions parse_v4l2_uri(const std::string& uri, std::string* error) {
    V4L2UriOptions out;
    if (uri.empty()) {
        return out;
    }

    std::string work = uri;
    auto colon = work.find(':');
    if (colon == std::string::npos) {
        return out;
    }
    std::string scheme = to_lower(work.substr(0, colon));
    if (scheme != "v4l2") {
        return out;
    }
    out.is_v4l2 = true;
    work = work.substr(colon + 1);

    while (work.rfind("//", 0) == 0) {
        work = work.substr(2);
    }

    std::string query;
    auto qpos = work.find('?');
    if (qpos != std::string::npos) {
        query = work.substr(qpos + 1);
        work = work.substr(0, qpos);
    }

    if (work.empty() || work == "auto") {
        out.auto_device = true;
        out.device = "auto";
    } else {
        out.device = work;
    }

    std::stringstream ss(query);
    std::string token;
    while (std::getline(ss, token, '&')) {
        if (token.empty()) {
            continue;
        }
        std::string key;
        std::string value;
        auto eq = token.find('=');
        if (eq == std::string::npos) {
            key = token;
            value.clear();
        } else {
            key = token.substr(0, eq);
            value = token.substr(eq + 1);
        }
        key = to_lower(key);
        out.query[key] = value;

        if (key == "fmt" || key == "format" || key == "cam_fmt") {
            std::string fmt = to_lower(value);
            if (fmt == "yuyv" || fmt == "yuyv422" || fmt == "yuv422") {
                out.fmt = "yuyv";
                out.fmt_specified = true;
            } else if (fmt == "mjpeg" || fmt == "jpeg") {
                out.fmt = "mjpeg";
                out.fmt_specified = true;
            } else if (fmt == "h264" || fmt == "h.264") {
                out.fmt = "h264";
                out.fmt_specified = true;
            } else {
                set_error(error, "unknown fmt=" + value);
            }
        } else if (key == "fps") {
            int val = out.fps;
            if (parse_int(value, 1, 120, val)) {
                out.fps = val;
                out.fps_specified = true;
            } else {
                set_error(error, "invalid fps=" + value);
            }
        } else if (key == "width" || key == "w") {
            int val = out.width;
            if (parse_int(value, 16, 7680, val)) {
                out.width = val;
                out.width_specified = true;
            } else {
                set_error(error, "invalid width=" + value);
            }
        } else if (key == "height" || key == "h") {
            int val = out.height;
            if (parse_int(value, 16, 4320, val)) {
                out.height = val;
                out.height_specified = true;
            } else {
                set_error(error, "invalid height=" + value);
            }
        } else if (key == "buffers" || key == "bufs" || key == "cam_bufs") {
            int val = out.buffers;
            if (parse_int(value, 1, 16, val)) {
                out.buffers = val;
                out.buffers_specified = true;
            } else {
                set_error(error, "invalid buffers=" + value);
            }
        } else if (key == "poll" || key == "poll_ms" || key == "timeout") {
            int val = out.poll_ms;
            if (parse_int(value, 1, 1000, val)) {
                out.poll_ms = val;
                out.poll_specified = true;
            } else {
                set_error(error, "invalid poll_ms=" + value);
            }
        } else if (key == "ttl" || key == "ttl_ms") {
            int val = out.ttl_ms;
            if (parse_int(value, 10, 3000, val)) {
                out.ttl_ms = val;
                out.ttl_specified = true;
            } else {
                set_error(error, "invalid ttl_ms=" + value);
            }
        }
    }

    return out;
}

} // namespace yolov5
