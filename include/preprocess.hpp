#ifndef PREPROCESS_HPP
#define PREPROCESS_HPP

#include "types.hpp"
#include <opencv2/core.hpp>
#include <vector>

// Forward declare to avoid leaking FFmpeg headers here
struct SwsContext;

namespace yolov5 {

/**
 * @file preprocess.hpp
 * @brief Declares preprocessing helpers producing CSI-NN2 ready tensors.
 *
 * The preprocessor performs letterboxing, colorspace conversion, and optional
 * RVV acceleration. Outputs align with YOLOv5n HHB expectations: FP16/FP32
 * NCHW tensors sized 1×3×384×640.
 */

/**
 * @brief Preprocessing pipeline bridging capture frames to model tensors.
 * @threading Owned by the preprocess thread exclusively.
 * @ownership Reuses cached SwsContext and temporary buffers internally.
 */
class Preprocessor {
public:
    /**
     * @brief Construct preprocessor using requested backend.
     * @param mode Scalar (SW) or RVV accelerated mode.
     */
    Preprocessor(PreprocMode mode = PreprocMode::SW);
    ~Preprocessor();

    /**
     * @brief Apply letterbox resize with padding to match model input.
     * @param src Input BGR image.
     * @param dst Output image resized/padded to target resolution.
     * @param scale Scale factor used later to remap detection boxes.
     * @param dx Horizontal padding applied during letterbox.
     * @param dy Vertical padding applied during letterbox.
     * @param target_w Target width (defaults to 640).
     * @param target_h Target height (defaults to 384).
     */
    void letterbox(const cv::Mat& src, cv::Mat& dst,
                   float& scale, int& dx, int& dy,
                   int target_w = MODEL_WIDTH, int target_h = MODEL_HEIGHT);

    /**
     * @brief Convert BGR image to FP16 NCHW buffer (scalar path).
     */
    void convertToNCHW_FP16(const cv::Mat& src, void* dst);

    /**
     * @brief Convert BGR image to FP32 NCHW using RVV kernels.
     */
    void convertToNCHW_FP32_RVV(const cv::Mat& src, float* dst);

    /**
     * @brief Convert BGR image to FP32 NCHW (scalar implementation).
     */
    void convertToNCHW_FP32(const cv::Mat& src, float* dst);

    /**
     * @brief Execute full preprocessing pipeline for a single frame.
     * @param input Frame containing capture data (possibly planar YUV).
     * @param model_input Output tensor buffer.
     * @param scale Output scale factor matching applied letterbox.
     * @param dx Horizontal padding offset for back-projection.
     * @param dy Vertical padding offset for back-projection.
     */
    void preprocess(Frame& input, void* model_input,
                    float& scale, int& dx, int& dy);

    /**
     * @brief Report whether RVV kernels are available on current CPU.
     */
    bool hasRVVSupport() const;

    /**
     * @brief Convert YUYV422 payloads into BGR Mat (scalar fallback).
     */
    void yuyvToBGR(const uint8_t* yuyv, cv::Mat& bgr, int width, int height);

    /**
     * @brief Convert YUYV422 using RVV acceleration.
     */
    void yuyvToBGR_RVV(const uint8_t* yuyv, cv::Mat& bgr, int width, int height);

private:
    PreprocMode mode_;
    SwsContext* sws_ctx_ = nullptr;  //!< Cached swscale context for planar YUV.

    /** @brief Scalar bilinear resize fallback. */
    void bilinearResizeScalar(const cv::Mat& src, cv::Mat& dst, int new_w, int new_h);

    /** @brief RVV accelerated bilinear resize. */
    void rvvResizeBilinear(const cv::Mat& src, cv::Mat& dst, int new_w, int new_h);

    // RVV kernels (with scalar fallbacks when RVV is not available)
    void resize_bilinear_bgr_rvv(const uint8_t* src, int sw, int sh, int sstride,
                                 uint8_t* dst, int dw, int dh, int dstride);

    void hwc_bgr8_to_nchw_fp16_rvv(const uint8_t* src, int w, int h, uint16_t* dst);

    void yuv420_to_bgr_rvv(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                            int w, int h, int y_stride, int uv_stride,
                            uint8_t* dst_bgr, int dst_stride);
    // SW path: convert planar YUV420p to BGR via swscale
    void yuv420_to_bgr_sw(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                          int w, int h, int y_stride, int uv_stride,
                          uint8_t* dst_bgr, int dst_stride);

    bool rvv_available_;
};

// Utility functions
namespace utils {
    /** @brief Detect RVV support using HWCAP probing. */
    bool checkRVVSupport();

    /** @brief Allocate aligned memory for SIMD-friendly operations. */
    void* alignedAlloc(size_t size, size_t alignment = 64);
    void alignedFree(void* ptr);
}

} // namespace yolov5

#endif // PREPROCESS_HPP
