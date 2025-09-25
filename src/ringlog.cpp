#include "ringlog.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <ctime>

namespace yolov5 {
namespace {
RingLogger g_logger{std::atomic<uint64_t>{0}, {}};
std::mutex g_dump_mutex;
std::string g_run_label;

uint64_t load_index() {
    return g_logger.index.load(std::memory_order_relaxed);
}

RingLogRecord snapshot_record(uint64_t i) {
    return g_logger.buffer[i % kRingCapacity];
}

size_t note_length(const RingLogRecord& rec) {
    size_t len = 0;
    while (len < sizeof(rec.note) && rec.note[len] != '\0') {
        ++len;
    }
    return len;
}

std::string sanitize_note(const RingLogRecord& rec) {
    return std::string(rec.note, note_length(rec));
}

std::string build_stage_string(RingStage stage) {
    switch (stage) {
        case RingStage::CAP_ENQ: return "CAP_ENQ";
        case RingStage::PP_DONE: return "PP_DONE";
        case RingStage::INF_START: return "INF_START";
        case RingStage::INF_DONE: return "INF_DONE";
        case RingStage::POST_DONE: return "POST_DONE";
        case RingStage::ORD_ENQ: return "ORD_ENQ";
        case RingStage::ORD_DROP_TTL: return "ORD_DROP_TTL";
        case RingStage::ORD_GAP_SKIP: return "ORD_GAP_SKIP";
        case RingStage::DISP_CREATE_OK: return "DISP_CREATE_OK";
        case RingStage::DISP_CREATE_FAIL: return "DISP_CREATE_FAIL";
        case RingStage::DISP_PRESENT_OK: return "DISP_PRESENT_OK";
        case RingStage::DISP_PRESENT_FAIL: return "DISP_PRESENT_FAIL";
        case RingStage::ENC_OPEN: return "ENC_OPEN";
        case RingStage::ENC_PKT: return "ENC_PKT";
        case RingStage::ENC_CLOSE: return "ENC_CLOSE";
        case RingStage::WDG_TRIGGER: return "WDG_TRIGGER";
        case RingStage::CAP_TIMEOUT: return "CAP_TIMEOUT";
        case RingStage::CAP_DROP: return "CAP_DROP";
        case RingStage::METRIC_FLUSH: return "METRIC_FLUSH";
        case RingStage::CUSTOM: return "CUSTOM";
    }
    return "UNKNOWN";
}

std::string active_label() {
    if (g_run_label.empty()) {
        return "ring";
    }
    return g_run_label;
}

void ensure_parent(const std::string& path) {
    if (path.empty()) return;
    try {
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    } catch (...) {
    }
}

} // namespace

RingLogger& ring_logger() {
    return g_logger;
}

void ring_set_run_label(const std::string& label) {
    std::lock_guard<std::mutex> lk(g_dump_mutex);
    g_run_label = label;
}

void ring_log(RingStage stage, int rc, const char* note, int64_t frame_id) {
    const uint64_t idx = g_logger.index.fetch_add(1, std::memory_order_relaxed);
    RingLogRecord rec{};
    rec.ts_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    rec.frame_id = frame_id;
    rec.stage = stage;
    rec.rc = rc;
    if (note) {
        std::snprintf(rec.note, sizeof(rec.note), "%s", note);
    } else {
        rec.note[0] = '\0';
    }
    g_logger.buffer[idx % kRingCapacity] = rec;
}

std::string ring_stage_name(RingStage stage) {
    return build_stage_string(stage);
}

void ring_dump_text(const std::string& path) {
    if (path.empty()) return;
    std::lock_guard<std::mutex> lk(g_dump_mutex);
    ensure_parent(path);
    std::ofstream ofs(path, std::ios::out | std::ios::trunc);
    if (!ofs) {
        return;
    }
    const uint64_t end = load_index();
    const uint64_t start = (end > kRingCapacity) ? end - kRingCapacity : 0;
    ofs << "# ring dump label=" << active_label() << " entries=" << (end - start) << "\n";
    ofs << "ts_ns frame_id stage rc note\n";
    for (uint64_t i = start; i < end; ++i) {
        RingLogRecord rec = snapshot_record(i);
        ofs << rec.ts_ns << ' '
            << rec.frame_id << ' '
            << build_stage_string(rec.stage) << ' '
            << rec.rc << ' ' << sanitize_note(rec) << '\n';
    }
}

void ring_dump_jsonl(const std::string& path) {
    if (path.empty()) return;
    std::lock_guard<std::mutex> lk(g_dump_mutex);
    ensure_parent(path);
    std::ofstream ofs(path, std::ios::out | std::ios::trunc);
    if (!ofs) {
        return;
    }
    const uint64_t end = load_index();
    const uint64_t start = (end > kRingCapacity) ? end - kRingCapacity : 0;
    for (uint64_t i = start; i < end; ++i) {
        RingLogRecord rec = snapshot_record(i);
        ofs << "{\"ts_ns\":" << rec.ts_ns
            << ",\"frame_id\":" << rec.frame_id
            << ",\"stage\":\"" << build_stage_string(rec.stage) << "\""
            << ",\"rc\":" << rec.rc
            << ",\"note\":\"";
        const std::string note = sanitize_note(rec);
        for (char c : note) {
            if (c == '"' || c == '\\') {
                ofs << '\\' << c;
            } else if (c == '\n' || c == '\r') {
                ofs << ' ';
            } else {
                ofs << c;
            }
        }
        ofs << "\"}" << '\n';
    }
}

void ring_dump_artifacts(const std::string& reason) {
    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    char ts_buf[32];
    std::snprintf(ts_buf, sizeof(ts_buf), "%04d-%02d-%02d_%02d-%02d-%02d",
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec);
    std::string sanitized_reason;
    sanitized_reason.reserve(reason.size());
    for (char c : reason) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) {
            sanitized_reason.push_back(c);
        } else if (c == '_' || c == '-') {
            sanitized_reason.push_back(c);
        } else {
            sanitized_reason.push_back('_');
        }
    }
    if (sanitized_reason.empty()) {
        sanitized_reason = "ring";
    }
    std::string text_path = std::string("/data/Work_Logs/ringdump_") + ts_buf + "_" + sanitized_reason + ".log";
    std::string json_path = std::string("artifacts/ring_") + active_label() + "_" + ts_buf + ".jsonl";
    ring_dump_text(text_path);
    ring_dump_jsonl(json_path);
}

} // namespace yolov5
