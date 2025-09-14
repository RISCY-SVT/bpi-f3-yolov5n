#include "display.hpp"
#include <iostream>

namespace yolov5 {

class NullDisplay : public IDisplay {
public:
    bool init(int, int, const std::string& title) override {
        std::cout << "[WARN] Display disabled or unavailable (" << title << ")" << std::endl;
        return true;
    }
    void show(const cv::Mat&) override {}
    void close() override {}
};

std::unique_ptr<IDisplay> createDisplay(const std::string& mode) {
    (void)mode;
    return std::make_unique<NullDisplay>();
}

} // namespace yolov5

