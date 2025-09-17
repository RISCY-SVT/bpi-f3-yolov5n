#ifndef DISPLAY_HPP
#define DISPLAY_HPP

#include "types.hpp"
#include <memory>

namespace yolov5 {

/**
 * @file display.hpp
 * @brief Declares display abstraction covering SDL2 and headless modes.
 *
 * Display implementations are owned by the overlay/output thread and are
 * responsible for presenting annotated frames, tracking watchdog timestamps,
 * and exposing driver metadata for diagnostics.
 */

/**
 * @brief Configuration for initializing a display backend.
 * @ownership Owned by caller; copied into display implementations during init().
 */
struct DisplayConfig {
    int width;              //!< Frame width in pixels.
    int height;             //!< Frame height in pixels.
    std::string title;      //!< Window title or identifier string.
    std::string driver;     //!< SDL driver hint (auto, wayland, kmsdrm, x11, dummy).
};

/**
 * @brief Metadata passed to present() while rendering annotated frames.
 * @threading Produced by overlay thread, consumed by display thread.
 */
struct DisplayFrameInfo {
    const cv::Mat& image;           //!< Annotated frame in BGR24 layout.
    uint64_t frame_id;              //!< Monotonic frame identifier for diagnostics.
    const PerfMetrics* metrics;     //!< Pointer to latest metrics snapshot (may be null).
    bool metrics_valid;             //!< Indicates that `metrics` points to fresh data.
};

/**
 * @brief Abstract display interface bridging overlay and presentation stages.
 * @threading Owned by the output/display thread. No concurrent access expected.
 * @lifecycle init() → present()/updateMetrics()* → close().
 */
class IDisplay {
public:
    virtual ~IDisplay() = default;

    /**
     * @brief Initialize backend with negotiated configuration.
     * @param cfg Dimensions, title, and driver hints from CLI/autodetect.
     * @return True when a visible or headless display is ready to present.
     */
    virtual bool init(const DisplayConfig& cfg) = 0;

    /**
     * @brief Present the next annotated frame to the user.
     * @param frame Frame metadata and pixel buffer produced by overlay stage.
     * @return True if the frame made it to the backend; false triggers pipeline stop.
     */
    virtual bool present(const DisplayFrameInfo& frame) = 0;

    /**
     * @brief Update metrics overlay state; some drivers draw HUDs from metrics.
     */
    virtual void updateMetrics(const PerfMetrics& metrics) = 0;

    /**
     * @brief Return monotonic timestamp of the most recent successful present().
     */
    virtual std::chrono::steady_clock::time_point lastPresentMono() const = 0;

    /**
     * @brief Provide backend identifier used for logging and watchdog messages.
     */
    virtual std::string driverName() const = 0;

    /**
     * @brief Shutdown the backend and release associated resources.
     */
    virtual void close() = 0;
};

/**
 * @brief Build display object for requested mode (sdl/off/auto maps).
 * @param mode CLI mode string (auto/off/sdl).
 * @param driver_hint SDL driver priority string.
 */
std::unique_ptr<IDisplay> createDisplay(const std::string& mode, const std::string& driver_hint);

/**
 * @brief Build a placeholder display that simply swallows present() calls.
 */
std::unique_ptr<IDisplay> createNullDisplay();

} // namespace yolov5

#endif // DISPLAY_HPP
