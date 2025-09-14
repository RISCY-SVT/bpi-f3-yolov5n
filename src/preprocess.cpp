#include "preprocess.hpp"
#include <opencv2/imgproc.hpp>
#include <cstring>
#include <algorithm>

#ifdef __riscv_vector
// RVV compatibility shim — comments in English only
extern "C" {
  #include <riscv_vector.h>
}
static inline size_t vl_e8m1(size_t n) {
# if defined(__clang__) || defined(RVV_USE_SHORT_NAMES)
  return vsetvl_e8m1(n);
# else
  return __riscv_vsetvl_e8m1(n);
# endif
}
static inline size_t vl_e16m1(size_t n) {
# if defined(__clang__) || defined(RVV_USE_SHORT_NAMES)
  return vsetvl_e16m1(n);
# else
  return __riscv_vsetvl_e16m1(n);
# endif
}
static inline size_t vl_e32m1(size_t n) {
# if defined(__clang__) || defined(RVV_USE_SHORT_NAMES)
  return vsetvl_e32m1(n);
# else
  return __riscv_vsetvl_e32m1(n);
# endif
}
#endif

namespace yolov5 {

Preprocessor::Preprocessor() {
    rvv_available_ = utils::checkRVVSupport();
}

Preprocessor::~Preprocessor() = default;

void Preprocessor::simpleResize(const cv::Mat& src, cv::Mat& dst, int new_w, int new_h) {
    dst = cv::Mat::zeros(new_h, new_w, CV_8UC3);
    
    float scale_x = (float)src.cols / new_w;
    float scale_y = (float)src.rows / new_h;
    
    for (int y = 0; y < new_h; y++) {
        for (int x = 0; x < new_w; x++) {
            int src_x = (int)(x * scale_x);
            int src_y = (int)(y * scale_y);
            
            if (src_x < src.cols && src_y < src.rows) {
                cv::Vec3b pixel = src.at<cv::Vec3b>(src_y, src_x);
                dst.at<cv::Vec3b>(y, x) = pixel;
            }
        }
    }
}

void Preprocessor::rvvResize(const cv::Mat& src, cv::Mat& dst, int new_w, int new_h) {
#ifdef __riscv_vector
    if (rvv_available_) {
        // RVV accelerated nearest-neighbor placeholder (bilinear TBD)
        dst = cv::Mat::zeros(new_h, new_w, CV_8UC3);
        float sx = (float)src.cols / new_w;
        float sy = (float)src.rows / new_h;
        for (int y = 0; y < new_h; y++) {
            int syi = (int)(y * sy);
            const uint8_t* srow = src.ptr<uint8_t>(syi);
            uint8_t* drow = dst.ptr<uint8_t>(y);
            int x = 0;
            while (x < new_w) {
                size_t vl = vl_e32m1((size_t)(new_w - x));
                for (size_t i = 0; i < vl; i++) {
                    int sxi = (int)((x + (int)i) * sx);
                    const uint8_t* sp = srow + sxi * 3;
                    uint8_t* dp = drow + (x + (int)i) * 3;
                    dp[0] = sp[0]; dp[1] = sp[1]; dp[2] = sp[2];
                }
                x += vl;
            }
        }
        return;
    }
#endif
    simpleResize(src, dst, new_w, new_h);
}

void Preprocessor::letterbox(const cv::Mat& src, cv::Mat& dst,
                              float& scale, int& dx, int& dy,
                              int target_w, int target_h) {
    // Create output with gray padding
    dst = cv::Mat::zeros(target_h, target_w, CV_8UC3);
    dst.setTo(cv::Scalar(128, 128, 128));
    
    // Calculate scale to fit image
    scale = std::min(float(target_w) / src.cols, float(target_h) / src.rows);
    int new_w = int(src.cols * scale);
    int new_h = int(src.rows * scale);
    
    // Resize image
    cv::Mat resized;
    if (rvv_available_) {
        rvvResize(src, resized, new_w, new_h);
    } else {
        simpleResize(src, resized, new_w, new_h);
    }
    
    // Calculate padding
    dx = (target_w - new_w) / 2;
    dy = (target_h - new_h) / 2;
    
    // Copy resized image to center of output
    for (int y = 0; y < new_h; y++) {
        for (int x = 0; x < new_w; x++) {
            dst.at<cv::Vec3b>(dy + y, dx + x) = resized.at<cv::Vec3b>(y, x);
        }
    }
}

void Preprocessor::convertToNCHW_FP32(const cv::Mat& src, float* dst) {
    const int h = src.rows;
    const int w = src.cols;
    const int c = src.channels();
    
    // Convert HWC to NCHW and normalize to [0, 1]
    for (int ch = 0; ch < c; ch++) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);
                dst[ch * h * w + y * w + x] = pixel[ch] / 255.0f;
            }
        }
    }
}

void Preprocessor::convertToNCHW_FP16(const cv::Mat& src, void* dst) {
    // Convert to FP32 then to FP16 (placeholder); rely on engine to handle FP16 properly.
    float* temp = new float[src.channels() * src.rows * src.cols];
    convertToNCHW_FP32(src, temp);
    uint16_t* fp16_dst = (uint16_t*)dst;
    const int total = src.channels() * src.rows * src.cols;
    for (int i = 0; i < total; i++) {
        fp16_dst[i] = (uint16_t)(std::min(std::max(temp[i], 0.0f), 1.0f) * 65535.0f);
    }
    delete[] temp;
}

void Preprocessor::preprocess(const Frame& input, void* model_input,
                               float& scale, int& dx, int& dy) {
    // Apply letterbox
    cv::Mat letterboxed;
    letterbox(input.image, letterboxed, scale, dx, dy);
    
    // Convert to model input format (FP32 for now, will be converted to FP16 in engine)
    convertToNCHW_FP32(letterboxed, (float*)model_input);
}

bool Preprocessor::hasRVVSupport() const {
    return rvv_available_;
}

void Preprocessor::yuyvToBGR(const uint8_t* yuyv, cv::Mat& bgr, int width, int height) {
    bgr = cv::Mat(height, width, CV_8UC3);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x += 2) {
            int idx = y * width * 2 + x * 2;
            
            uint8_t y0 = yuyv[idx];
            uint8_t u = yuyv[idx + 1];
            uint8_t y1 = yuyv[idx + 2];
            uint8_t v = yuyv[idx + 3];
            
            // YUV to RGB conversion
            int c = y0 - 16;
            int d = u - 128;
            int e = v - 128;
            
            int r0 = std::max(0, std::min(255, (298 * c + 409 * e + 128) >> 8));
            int g0 = std::max(0, std::min(255, (298 * c - 100 * d - 208 * e + 128) >> 8));
            int b0 = std::max(0, std::min(255, (298 * c + 516 * d + 128) >> 8));
            
            c = y1 - 16;
            int r1 = std::max(0, std::min(255, (298 * c + 409 * e + 128) >> 8));
            int g1 = std::max(0, std::min(255, (298 * c - 100 * d - 208 * e + 128) >> 8));
            int b1 = std::max(0, std::min(255, (298 * c + 516 * d + 128) >> 8));
            
            bgr.at<cv::Vec3b>(y, x) = cv::Vec3b(b0, g0, r0);
            bgr.at<cv::Vec3b>(y, x + 1) = cv::Vec3b(b1, g1, r1);
        }
    }
}

void Preprocessor::yuyvToBGR_RVV(const uint8_t* yuyv, cv::Mat& bgr, int width, int height) {
    // For now, fall back to scalar implementation
    // RVV implementation would go here
    yuyvToBGR(yuyv, bgr, width, height);
}

namespace utils {

bool checkRVVSupport() {
#ifdef __riscv_vector
    return true;
#else
    return false;
#endif
}

void* alignedAlloc(size_t size, size_t alignment) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
}

void alignedFree(void* ptr) {
    free(ptr);
}

} // namespace utils

} // namespace yolov5
