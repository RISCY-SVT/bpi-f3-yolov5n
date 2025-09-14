#include "engine.hpp"
#include <opencv2/imgproc.hpp>

namespace yolov5 {

static cv::Scalar color_for(int id) {
    static const cv::Scalar palette[] = {
        {255, 0, 0}, {0,255,0}, {0,0,255}, {255,255,0}, {255,0,255}, {0,255,255}
    };
    return palette[id % (sizeof(palette)/sizeof(palette[0]))];
}

void draw_detections(cv::Mat& frame_bgr, const std::vector<Detection>& dets) {
    for (const auto& d : dets) {
        cv::Rect r(cv::Point((int)d.x1, (int)d.y1), cv::Point((int)d.x2, (int)d.y2));
        cv::rectangle(frame_bgr, r, color_for(d.class_id), 2);
        char buf[128];
        snprintf(buf, sizeof(buf), "%s %.2f", d.label.c_str(), d.confidence);
        int base=0; cv::Size ts = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
        cv::rectangle(frame_bgr, cv::Rect(r.x, std::max(0, r.y - ts.height - 4), ts.width + 6, ts.height + 4), color_for(d.class_id), -1);
        cv::putText(frame_bgr, buf, cv::Point(r.x + 3, std::max(0, r.y - 3)), cv::FONT_HERSHEY_SIMPLEX, 0.5, {0,0,0}, 1);
    }
}

} // namespace yolov5

