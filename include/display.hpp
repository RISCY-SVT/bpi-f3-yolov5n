#ifndef DISPLAY_HPP
#define DISPLAY_HPP

#include "types.hpp"
#include <memory>

namespace yolov5 {

class IDisplay {
public:
    virtual ~IDisplay() = default;
    virtual bool init(int width, int height, const std::string& title) = 0;
    virtual void show(const cv::Mat& frame) = 0;
    virtual void close() = 0;
};

// Factory: returns SDL display when available and requested, otherwise no-op
std::unique_ptr<IDisplay> createDisplay(const std::string& mode);

} // namespace yolov5

#endif // DISPLAY_HPP

