#ifndef PREPROCESS_HPP
#define PREPROCESS_HPP

#include "types.hpp"
#include <opencv2/core.hpp>
#include <vector>

namespace yolov5 {

// Preprocessing functions for YOLOv5 model input
class Preprocessor {
public:
    Preprocessor();
    ~Preprocessor();
    
    // Letterbox preprocessing - maintains aspect ratio with padding
    void letterbox(const cv::Mat& src, cv::Mat& dst, 
                   float& scale, int& dx, int& dy,
                   int target_w = MODEL_WIDTH, int target_h = MODEL_HEIGHT);
    
    // Convert BGR image to NCHW FP16 format for model input
    void convertToNCHW_FP16(const cv::Mat& src, void* dst);
    
    // Convert BGR image to NCHW FP32 format (for compatibility)
    void convertToNCHW_FP32(const cv::Mat& src, float* dst);
    
    // Complete preprocessing pipeline
    void preprocess(const Frame& input, void* model_input,
                    float& scale, int& dx, int& dy);
    
    // Check if RVV acceleration is available
    bool hasRVVSupport() const;
    
    // YUV to RGB conversion (for V4L2 YUYV format)
    void yuyvToBGR(const uint8_t* yuyv, cv::Mat& bgr, int width, int height);
    
    // RVV-accelerated YUV to RGB
    void yuyvToBGR_RVV(const uint8_t* yuyv, cv::Mat& bgr, int width, int height);
    
private:
    // Simple resize (scalar implementation)
    void simpleResize(const cv::Mat& src, cv::Mat& dst, int new_w, int new_h);
    
    // RVV-accelerated resize (if available)
    void rvvResize(const cv::Mat& src, cv::Mat& dst, int new_w, int new_h);
    
    bool rvv_available_;
};

// Utility functions
namespace utils {
    // Check RVV availability at runtime
    bool checkRVVSupport();
    
    // Aligned memory allocation for SIMD operations
    void* alignedAlloc(size_t size, size_t alignment = 64);
    void alignedFree(void* ptr);
}

} // namespace yolov5

#endif // PREPROCESS_HPP