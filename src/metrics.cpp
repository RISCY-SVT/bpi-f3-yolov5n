#include "metrics.hpp"
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>

/**
 * @file metrics.cpp
 * @brief JSONL metrics writer implementation with deterministic schema.
 */

namespace yolov5 {

/** @brief Escape string for safe inclusion in JSON output. */
static std::string json_escape(const std::string& s) {
    std::ostringstream o;
    for (auto c : s) {
        switch (c) {
            case '"': o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    o << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                      << (int)(unsigned char)c;
                } else {
                    o << c;
                }
        }
    }
    return o.str();
}

/**
 * @brief Open JSONL file for append; parent directories should exist already.
 */
JSONLMetricsWriter::JSONLMetricsWriter(const std::string& path) {
    ofs_.open(path, std::ios::out | std::ios::app);
}

JSONLMetricsWriter::~JSONLMetricsWriter() {
    if (ofs_.is_open()) ofs_.close();
}

/**
 * @brief Serialize metrics snapshot to JSONL using fixed key order.
 */
void JSONLMetricsWriter::write(const PerfMetrics& m) {
    if (!ofs_.is_open()) return;
    std::lock_guard<std::mutex> lock(mu_);

    ofs_ << '{'
         << "\"ts_ms\":" << m.timestamp_ms << ','
         << "\"in_fps\":" << std::fixed << std::setprecision(2) << m.input_fps << ','
         << "\"out_fps\":" << std::fixed << std::setprecision(2) << m.output_fps << ','
         << "\"drop_pct\":" << std::fixed << std::setprecision(2) << m.drop_percentage << ',';

    // latency_ms object
    ofs_ << "\"latency_ms\":{"
         << "\"cap\":" << std::fixed << std::setprecision(2) << m.latency_ms.capture << ','
         << "\"pp\":" << std::fixed << std::setprecision(2) << m.latency_ms.preprocess << ','
         << "\"inf_p50\":" << std::fixed << std::setprecision(2) << m.latency_ms.inference_p50 << ','
         << "\"inf_p95\":" << std::fixed << std::setprecision(2) << m.latency_ms.inference_p95 << ','
         << "\"post\":" << std::fixed << std::setprecision(2) << m.latency_ms.postprocess << ','
         << "\"overlay\":" << std::fixed << std::setprecision(2) << m.latency_ms.overlay << ','
         << "\"enc\":" << std::fixed << std::setprecision(2) << m.latency_ms.encode
         << "},";

    // qsize object (ensure fixed set of keys)
    auto qv = [&](const char* k) -> int {
        auto it = m.queue_sizes.find(k);
        return (it == m.queue_sizes.end()) ? 0 : it->second;
    };
    ofs_ << "\"qsize\":{"
         << "\"cap_pp\":" << qv("cap_pp") << ','
         << "\"pp_sched\":" << qv("pp_sched") << ','
         << "\"sched_inf\":" << qv("sched_inf") << ','
         << "\"inf_post\":" << qv("inf_post") << ','
         << "\"post_reord\":" << qv("post_reord") << ','
         << "\"reorder_buf\":" << qv("reorder_buf")
         << "},"
         << "\"heap_bytes\":" << m.heap_bytes << ',';

    // workers_busy_pct array
    ofs_ << "\"workers_busy_pct\":[";
    for (size_t i = 0; i < m.worker_busy_pct.size(); ++i) {
        if (i) ofs_ << ',';
        ofs_ << std::fixed << std::setprecision(1) << m.worker_busy_pct[i];
    }
    ofs_ << "]}" << '\n';
    ofs_.flush();
}

} // namespace yolov5
