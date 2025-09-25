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
         << "\"enc\":" << std::fixed << std::setprecision(2) << m.latency_ms.encode << ','
         << "\"display\":" << std::fixed << std::setprecision(2) << m.latency_ms.display
         << "},";

    ofs_ << "\"e2e_ms\":{"
         << "\"p50\":" << std::fixed << std::setprecision(2) << m.e2e_ms.p50 << ','
         << "\"p95\":" << std::fixed << std::setprecision(2) << m.e2e_ms.p95
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
         << "\"reorder_buf\":" << qv("reorder_buf") << ','
         << "\"q_cap\":" << qv("q_cap") << ','
         << "\"q_pp\":" << qv("q_pp") << ','
         << "\"q_inf\":" << qv("q_inf") << ','
         << "\"q_post\":" << qv("q_post") << ','
         << "\"q_ord\":" << qv("q_ord")
         << "},";

    auto qc = [&](const char* k) -> int {
        auto it = m.queue_capacity.find(k);
        return (it == m.queue_capacity.end()) ? 0 : it->second;
    };
    ofs_ << "\"qcap\":{"
         << "\"cap_pp\":" << qc("cap_pp") << ','
         << "\"pp_sched\":" << qc("pp_sched") << ','
         << "\"sched_inf\":" << qc("sched_inf") << ','
         << "\"inf_post\":" << qc("inf_post") << ','
         << "\"post_reord\":" << qc("post_reord") << ','
         << "\"reorder_buf\":" << qc("reorder_buf") << ','
         << "\"q_cap\":" << qc("q_cap") << ','
         << "\"q_pp\":" << qc("q_pp") << ','
         << "\"q_inf\":" << qc("q_inf") << ','
         << "\"q_post\":" << qc("q_post") << ','
         << "\"q_ord\":" << qc("q_ord")
         << "},";

    ofs_ << "\"drops\":{";
    bool first = true;
    for (const auto& kv : m.drop_counts) {
        if (!first) {
            ofs_ << ',';
        }
        first = false;
        ofs_ << '"' << json_escape(kv.first) << "\":" << kv.second;
    }
    ofs_ << "},"
         << "\"drop_backlog\":" << m.drop_backlog << ','
         << "\"drop_ttl\":" << m.drop_ttl << ','
         << "\"gap_skips\":" << m.gap_skips << ','
         << "\"ttl_drops\":" << m.ttl_drops << ','
         << "\"live_gate_block\":" << m.live_gate_block << ','
         << "\"reorder_backlog_max\":" << m.reorder_backlog_max << ','
         << "\"display_presented\":" << (m.display_presented ? "true" : "false") << ','
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
