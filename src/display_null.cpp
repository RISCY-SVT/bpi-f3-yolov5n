#include "display.hpp"
#include <iostream>

/**
 * @file display_null.cpp
 * @brief Headless display backend implementing no-op presentation.
 */

namespace yolov5 {

/**
 * @brief Placeholder display used when SDL2 is unavailable or disabled.
 */
class NullDisplay : public IDisplay {
public:
    /** @brief Report that display is disabled but keep watchdog alive. */
    bool init(const DisplayConfig& cfg) override {
        std::cout << "[WARN] Display disabled or unavailable (" << cfg.title << ")" << std::endl;
        last_present_ = std::chrono::steady_clock::now();
        return true;
    }
    bool present(const DisplayFrameInfo&) override {
        last_present_ = std::chrono::steady_clock::now();
        return true;
    }
    void updateMetrics(const PerfMetrics&) override {}
    std::chrono::steady_clock::time_point lastPresentMono() const override {
        return last_present_;
    }
    std::string driverName() const override { return "null"; }
    void close() override {}

private:
    std::chrono::steady_clock::time_point last_present_{};
};

std::unique_ptr<IDisplay> createNullDisplay() {
    return std::make_unique<NullDisplay>();
}

#ifndef HAVE_SDL2
std::unique_ptr<IDisplay> createDisplay(const std::string& mode, const std::string& driver_hint) {
    (void)driver_hint;
    if (mode == "sdl") {
        return nullptr;
    }
    return createNullDisplay();
}
#endif

} // namespace yolov5
