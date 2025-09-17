#ifndef CAPTURE_HPP
#define CAPTURE_HPP

#include "types.hpp"
#include <memory>
#include <string>
#include <atomic>

namespace yolov5 {

/**
 * @file capture.hpp
 * @brief Declares capture interfaces for file-based and V4L2 video sources.
 *
 * Capture objects operate on the capture thread owned by the pipeline and feed
 * decoded frames into the capture queue. Implementations translate raw media
 * streams into `Frame` structures, keeping timestamps and frame ids monotonic.
 */

/**
 * @brief Abstract interface for all capture backends used by the pipeline.
 * @threading Owned and accessed exclusively by the capture thread after
 *            construction. No concurrent calls are expected.
 * @lifecycle init() → repeated getFrame() → release().
 */
class ICapture {
public:
    virtual ~ICapture() = default;

    /**
     * @brief Initialize the capture backend and open the source.
     * @param source Canonicalized CLI string (e.g. file:/path or v4l2:/dev/video0?).
     * @return True when the source is ready for frame retrieval.
     */
    virtual bool init(const std::string& source) = 0;

    /**
     * @brief Fetch the next decoded frame from the source.
     * @param frame Output frame populated on success.
     * @return True when a frame was produced, false on EOF or error.
     */
    virtual bool getFrame(Frame& frame) = 0;

    /**
     * @brief Release all resources and close the capture source.
     */
    virtual void release() = 0;

    /**
     * @brief Report negotiated width in pixels.
     */
    virtual int getWidth() const = 0;
    /**
     * @brief Report negotiated height in pixels.
     */
    virtual int getHeight() const = 0;
    /**
     * @brief Report estimated frames per second if available.
     */
    virtual double getFPS() const = 0;
    /**
     * @brief Report total frame count when the container provides it.
     */
    virtual int getFrameCount() const = 0;

    /**
     * @brief Check whether the capture backend is still operational.
     */
    virtual bool isOpened() const = 0;
};

/**
 * @brief FFmpeg-backed file capture that parses demuxed video streams.
 * @threading Owned by the capture thread; FFmpeg contexts stay confined to it.
 * @ownership Holds decoder state and reuses internal AVFrame buffers between calls.
 */
class CaptureFile : public ICapture {
public:
    CaptureFile();
    ~CaptureFile() override;

    bool init(const std::string& source) override;
    bool getFrame(Frame& frame) override;
    void release() override;

    int getWidth() const override;
    int getHeight() const override;
    double getFPS() const override;
    int getFrameCount() const override;
    bool isOpened() const override;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl; //!< Hidden FFmpeg state managed via PIMPL.
};

/**
 * @brief V4L2 capture implementation for USB cameras.
 * @threading Owned by the capture thread; file descriptors are not shared.
 * @ownership Manages kernel buffers and converts camera YUV/MJPEG payloads to Frame.
 */
class CaptureV4L2 : public ICapture {
public:
    CaptureV4L2();
    ~CaptureV4L2() override;

    bool init(const std::string& source) override;
    bool getFrame(Frame& frame) override;
    void release() override;

    int getWidth() const override;
    int getHeight() const override;
    double getFPS() const override;
    int getFrameCount() const override;
    bool isOpened() const override;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl; //!< Internal V4L2 state wrapper.
};

/**
 * @brief Factory selecting capture backend based on CLI source scheme.
 * @param source Expanded source string (with scheme prefixes resolved).
 * @return Concrete capture implementation ready for init().
 */
std::unique_ptr<ICapture> createCapture(const std::string& source);

} // namespace yolov5

#endif // CAPTURE_HPP
