#ifndef YOLONV_RINGLOG_HPP
#define YOLONV_RINGLOG_HPP

#include <atomic>
#include <array>
#include <cstdint>
#include <string>
#include <ostream>

namespace yolov5 {

constexpr int64_t kRingNoFrame = -1;
constexpr size_t kRingCapacity = 4096;

/**
 * @brief Enumerates pipeline stages recorded into the live ring logger.
 */
enum class RingStage : uint16_t {
    CAP_ENQ = 0,
    PP_DONE,
    INF_START,
    INF_DONE,
    POST_DONE,
    ORD_ENQ,
    ORD_DROP_TTL,
    ORD_GAP_SKIP,
    DISP_CREATE_OK,
    DISP_CREATE_FAIL,
    DISP_PRESENT_OK,
    DISP_PRESENT_FAIL,
    ENC_OPEN,
    ENC_PKT,
    ENC_CLOSE,
    WDG_TRIGGER,
    CAP_TIMEOUT,
    CAP_DROP,
    METRIC_FLUSH,
    CUSTOM
};

struct RingLogRecord {
    uint64_t ts_ns;
    int64_t frame_id;
    RingStage stage;
    int rc;
    char note[32];
};

struct RingLogger {
    std::atomic<uint64_t> index;
    std::array<RingLogRecord, kRingCapacity> buffer;
};

RingLogger& ring_logger();
void ring_log(RingStage stage, int rc, const char* note, int64_t frame_id);
std::string ring_stage_name(RingStage stage);
void ring_dump_text(const std::string& path);
void ring_dump_jsonl(const std::string& path);
void ring_set_run_label(const std::string& label);
void ring_dump_artifacts(const std::string& reason);

#define LOG_STAGE(stage_value, rc_value, note_literal) \
    ::yolov5::ring_log((stage_value), (rc_value), (note_literal), ::yolov5::kRingNoFrame)

#define LOG_STAGE_F(stage_value, frame_id_value, rc_value, note_literal) \
    ::yolov5::ring_log((stage_value), (rc_value), (note_literal), static_cast<int64_t>(frame_id_value))

} // namespace yolov5

#endif
