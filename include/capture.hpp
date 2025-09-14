#ifndef CAPTURE_HPP
#define CAPTURE_HPP

#include "types.hpp"
#include <memory>
#include <string>
#include <atomic>

namespace yolov5 {

// Abstract base class for video capture
class ICapture {
public:
    virtual ~ICapture() = default;
    
    // Initialize the capture source
    virtual bool init(const std::string& source) = 0;
    
    // Get next frame
    virtual bool getFrame(Frame& frame) = 0;
    
    // Release resources
    virtual void release() = 0;
    
    // Get capture properties
    virtual int getWidth() const = 0;
    virtual int getHeight() const = 0;
    virtual double getFPS() const = 0;
    virtual int getFrameCount() const = 0;
    
    // Check if capture is still active
    virtual bool isOpened() const = 0;
};

// File-based video capture using FFmpeg
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
    std::unique_ptr<Impl> pImpl;
};

// V4L2 USB camera capture
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
    std::unique_ptr<Impl> pImpl;
};

// Factory function to create appropriate capture
std::unique_ptr<ICapture> createCapture(const std::string& source);

} // namespace yolov5

#endif // CAPTURE_HPP