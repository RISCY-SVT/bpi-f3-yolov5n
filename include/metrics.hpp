#ifndef METRICS_HPP
#define METRICS_HPP

#include "types.hpp"
#include <fstream>
#include <mutex>
#include <string>

namespace yolov5 {

/**
 * @file metrics.hpp
 * @brief Declares JSONL metrics writer used by the pipeline.
 *
 * Metrics are emitted by the metrics thread at a fixed cadence and flushed to
 * disk so they can be collected post-run. Thread safety protects against
 * concurrent writes from multiple producers.
 */

/**
 * @brief Writes metrics snapshots into a JSON Lines file.
 * @threading Safe for concurrent calls; guards with an internal mutex.
 * @lifecycle Construct once per pipeline when `--perf-json` is provided.
 */
class JSONLMetricsWriter {
public:
    /**
     * @brief Open output file and prepare stream.
     * @param path Destination JSONL file path (created or truncated).
     */
    explicit JSONLMetricsWriter(const std::string& path);
    ~JSONLMetricsWriter();

    /**
     * @brief Append one metrics record as a single JSON line.
     * @param m Metrics snapshot; function serializes and flushes.
     */
    void write(const PerfMetrics& m);

private:
    std::ofstream ofs_; //!< Owned JSONL stream.
    std::mutex mu_;     //!< Protects interleaved write() calls.
};

} // namespace yolov5

#endif // METRICS_HPP
