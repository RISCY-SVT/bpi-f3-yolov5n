#include "v4l2_uri.hpp"
#include <cassert>
#include <iostream>

using namespace yolov5;

int main() {
    {
        std::string err;
        auto opts = parse_v4l2_uri("v4l2:auto?fmt=yuyv", &err);
        assert(opts.is_v4l2);
        assert(opts.auto_device);
        assert(opts.device == "auto");
        assert(opts.fmt == "yuyv");
        assert(opts.fmt_specified);
        assert(err.empty());
    }
    {
        std::string err;
        auto opts = parse_v4l2_uri("v4l2:/dev/video21?fmt=yuyv&fps=30", &err);
        assert(opts.is_v4l2);
        assert(!opts.auto_device);
        assert(opts.device == "/dev/video21");
        assert(opts.fps == 30);
        assert(opts.fps_specified);
        assert(opts.buffers == 3);
        assert(err.empty());
    }
    {
        std::string err;
        auto opts = parse_v4l2_uri("v4l2:auto?fmt=mjpeg&fps=25&width=640&height=480", &err);
        assert(opts.is_v4l2);
        assert(opts.auto_device);
        assert(opts.fmt == "mjpeg");
        assert(opts.width == 640);
        assert(opts.height == 480);
        assert(opts.width_specified);
        assert(opts.height_specified);
        assert(opts.fps == 25);
        assert(opts.fps_specified);
        assert(err.empty());
    }
    std::cout << "test_v4l2_uri: OK" << std::endl;
    return 0;
}
