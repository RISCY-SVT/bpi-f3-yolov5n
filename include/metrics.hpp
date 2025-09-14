#ifndef METRICS_HPP
#define METRICS_HPP

#include "types.hpp"
#include <fstream>
#include <mutex>
#include <string>

namespace yolov5 {

// Simple JSONL metrics writer. One JSON object per line.
class JSONLMetricsWriter {
public:
    explicit JSONLMetricsWriter(const std::string& path);
    ~JSONLMetricsWriter();

    // Append one metrics record as JSON line.
    void write(const PerfMetrics& m);

private:
    std::ofstream ofs_;
    std::mutex mu_;
};

} // namespace yolov5

#endif // METRICS_HPP

