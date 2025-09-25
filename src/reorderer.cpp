#include "pipeline.hpp"

#include <algorithm>
#include <chrono>

namespace yolov5 {

namespace {
size_t live_limit(size_t cap_hint, size_t backlog_limit) {
    if (backlog_limit > 0) {
        return backlog_limit;
    }
    if (cap_hint > 0) {
        return cap_hint;
    }
    return 8;
}
}

FrameReorderer::FrameReorderer()
    : expected_id_(0),
      mode_(LatencyMode::Normal),
      live_ttl_(std::chrono::milliseconds(300)),
      live_cap_hint_(0),
      live_backlog_limit_(0),
      drop_backlog_(0),
      drop_ttl_(0),
      gap_skips_(0),
      backlog_max_(0),
      live_latest_(std::nullopt),
      live_latest_id_(0),
      live_expected_id_(0),
      stopped_(false) {}

void FrameReorderer::configureLive(LatencyMode mode, int ttl_ms, size_t cap_hint, size_t backlog_limit) {
    std::lock_guard<std::mutex> lock(mutex_);
    mode_ = mode;
    live_ttl_ = std::chrono::milliseconds(std::max(1, ttl_ms));
    live_cap_hint_ = cap_hint;
    live_backlog_limit_ = backlog_limit;
    drop_backlog_.store(0, std::memory_order_relaxed);
    drop_ttl_.store(0, std::memory_order_relaxed);
    gap_skips_.store(0, std::memory_order_relaxed);
    backlog_max_ = 0;
    live_latest_.reset();
    live_latest_id_ = 0;
    live_expected_id_ = 0;
}

void FrameReorderer::addFrame(const ProcessedFrame& frame) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_ == LatencyMode::Live) {
        const uint64_t fid = frame.frame.frame_id;
        if (live_latest_) {
            if (fid < live_latest_id_) {
                drop_backlog_.fetch_add(1, std::memory_order_relaxed);
                LOG_STAGE_F(RingStage::ORD_GAP_SKIP, fid, -1, "live_old");
                return;
            }
            if (fid > live_latest_id_ + 1) {
                gap_skips_.fetch_add(fid - (live_latest_id_ + 1), std::memory_order_relaxed);
                LOG_STAGE_F(RingStage::ORD_GAP_SKIP, fid, 0, "live_gap");
            }
        } else if (fid > live_expected_id_) {
            gap_skips_.fetch_add(fid - live_expected_id_, std::memory_order_relaxed);
            LOG_STAGE_F(RingStage::ORD_GAP_SKIP, fid, 0, "live_gap_init");
        }
        live_latest_ = frame;
        live_latest_id_ = fid;
        live_expected_id_ = fid + 1;
        backlog_max_ = std::max(backlog_max_, static_cast<size_t>(live_latest_ ? 1 : 0));
        LOG_STAGE_F(RingStage::ORD_ENQ, fid, 0, "ord_enq_live");
        cv_.notify_all();
        return;
    }
    buffer_[frame.frame.frame_id] = frame;
    backlog_max_ = std::max(backlog_max_, buffer_.size());
    LOG_STAGE_F(RingStage::ORD_ENQ, frame.frame.frame_id, 0, "ord_enq");

    if (mode_ == LatencyMode::Live) {
        const size_t limit = live_limit(live_cap_hint_, live_backlog_limit_);
        while (limit > 0 && buffer_.size() > limit) {
            auto it = buffer_.begin();
            if (it == buffer_.end()) {
                break;
            }
            const uint64_t victim = it->first;
            buffer_.erase(it);
            drop_backlog_.fetch_add(1, std::memory_order_relaxed);
            gap_skips_.fetch_add(1, std::memory_order_relaxed);
            LOG_STAGE_F(RingStage::ORD_GAP_SKIP, victim, 1, "reorder_trim");
            if (victim == expected_id_) {
                ++expected_id_;
            } else if (victim > expected_id_) {
                dropped_ids_.insert(victim);
            }
        }
    }
    cv_.notify_all();
}

void FrameReorderer::markDropped(uint64_t frame_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_ == LatencyMode::Live) {
        if (frame_id >= live_expected_id_) {
            gap_skips_.fetch_add(1, std::memory_order_relaxed);
        }
        if (live_latest_ && frame_id == live_latest_id_) {
            live_latest_.reset();
        }
        cv_.notify_all();
        return;
    }
    dropped_ids_.insert(frame_id);
    cv_.notify_all();
}

bool FrameReorderer::getNextFrame(ProcessedFrame& out) {
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
        if (stopped_) {
            return false;
        }

        // Skip frames explicitly marked as dropped upstream.
        while (!dropped_ids_.empty()) {
            auto it = dropped_ids_.find(expected_id_);
            if (it == dropped_ids_.end()) {
                break;
            }
            dropped_ids_.erase(it);
            gap_skips_.fetch_add(1, std::memory_order_relaxed);
            LOG_STAGE_F(RingStage::ORD_GAP_SKIP, expected_id_, 0, "marked_drop");
            ++expected_id_;
        }

        if (mode_ == LatencyMode::Live) {
            auto now = std::chrono::steady_clock::now();
            bool ttl_purged = false;
            while (!buffer_.empty()) {
                auto it = buffer_.begin();
                const auto& pf = it->second;
                if (now - pf.frame.timestamp <= live_ttl_) {
                    break;
                }
                const uint64_t expired = it->first;
                buffer_.erase(it);
                drop_ttl_.fetch_add(1, std::memory_order_relaxed);
                LOG_STAGE_F(RingStage::ORD_DROP_TTL, expired, 0, "ttl");
                if (expired <= expected_id_) {
                    ++expected_id_;
                } else {
                    dropped_ids_.insert(expired);
                }
                ttl_purged = true;
            }
            if (ttl_purged) {
                continue;
            }
        }

        auto ready = buffer_.find(expected_id_);
        if (ready != buffer_.end()) {
            out = std::move(ready->second);
            buffer_.erase(ready);
            ++expected_id_;
            return true;
        }

        if (mode_ == LatencyMode::Live) {
            if (live_latest_) {
                auto now = std::chrono::steady_clock::now();
                if (now - live_latest_->frame.timestamp > live_ttl_) {
                    drop_ttl_.fetch_add(1, std::memory_order_relaxed);
                    LOG_STAGE_F(RingStage::ORD_DROP_TTL, live_latest_id_, 0, "live_ttl_get");
                }
                out = std::move(*live_latest_);
                live_latest_.reset();
                return true;
            }
            cv_.wait_for(lock, std::chrono::milliseconds(5));
            continue;
        } else {
            cv_.wait(lock);
        }
    }
}

bool FrameReorderer::popLatestNonBlocking(ProcessedFrame& out) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_ != LatencyMode::Live) {
        return false;
    }
    if (!live_latest_) {
        return false;
    }
    auto now = std::chrono::steady_clock::now();
    if (now - live_latest_->frame.timestamp > live_ttl_) {
        drop_ttl_.fetch_add(1, std::memory_order_relaxed);
        LOG_STAGE_F(RingStage::ORD_DROP_TTL, live_latest_id_, 0, "live_ttl");
    }
    out = std::move(*live_latest_);
    live_latest_.reset();
    return true;
}

void FrameReorderer::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_.clear();
    dropped_ids_.clear();
    expected_id_ = 0;
    drop_backlog_.store(0, std::memory_order_relaxed);
    drop_ttl_.store(0, std::memory_order_relaxed);
    gap_skips_.store(0, std::memory_order_relaxed);
    backlog_max_ = 0;
    live_latest_.reset();
    live_latest_id_ = 0;
    live_expected_id_ = 0;
}

void FrameReorderer::stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    stopped_ = true;
    cv_.notify_all();
}

size_t FrameReorderer::pendingCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_ == LatencyMode::Live) {
        return live_latest_ ? 1 : 0;
    }
    return buffer_.size();
}

uint64_t FrameReorderer::dropBacklogCount() const {
    return drop_backlog_.load(std::memory_order_relaxed);
}

uint64_t FrameReorderer::dropTTLCount() const {
    return drop_ttl_.load(std::memory_order_relaxed);
}

uint64_t FrameReorderer::gapSkipCount() const {
    return gap_skips_.load(std::memory_order_relaxed);
}

size_t FrameReorderer::backlogMax() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return backlog_max_;
}

} // namespace yolov5
