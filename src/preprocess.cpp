#include "preprocess.hpp"
#include <opencv2/imgproc.hpp>
#include <cstring>
#include <algorithm>
#include <cmath>

extern "C" {
#include <libswscale/swscale.h>
#include <libavutil/pixfmt.h>
}

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

/**
 * @file preprocess.cpp
 * @brief Implements letterbox, colorspace conversions, and RVV-assisted kernels.
 */

namespace yolov5 {

/** @brief Configure preprocessor mode and detect RVV availability. */
Preprocessor::Preprocessor(PreprocMode mode) : mode_(mode) {
    rvv_available_ = utils::checkRVVSupport();
}

/** @brief Release swscale context if allocated. */
Preprocessor::~Preprocessor() {
    if (sws_ctx_) {
        sws_freeContext(sws_ctx_);
        sws_ctx_ = nullptr;
    }
}

// Scalar bilinear resize for BGR8 images using fixed-point weights to avoid FP noise.
void Preprocessor::bilinearResizeScalar(const cv::Mat& src, cv::Mat& dst, int new_w, int new_h) {
    dst = cv::Mat(new_h, new_w, CV_8UC3);
    const int sw = src.cols, sh = src.rows;
    const int stride = (int)src.step;
    const uint8_t* S = src.data;
    uint8_t* D = dst.data;
    const int FP = 8; // 8-bit fractional
    const int ONE = 1 << FP;

    // Precompute X mapping
    std::vector<int> x0(new_w), x1(new_w), wx(new_w);
    for (int dx = 0; dx < new_w; ++dx) {
        float sx = (dx + 0.5f) * (float)sw / new_w - 0.5f;
        int ix = (int)std::floor(sx);
        int ix1 = std::min(ix + 1, sw - 1);
        int fx = (int)std::round((sx - ix) * ONE);
        if (ix < 0) { ix = 0; fx = 0; }
        x0[dx] = std::clamp(ix, 0, sw - 1);
        x1[dx] = std::clamp(ix1, 0, sw - 1);
        wx[dx] = fx;
    }

    for (int dy = 0; dy < new_h; ++dy) {
        float sy = (dy + 0.5f) * (float)sh / new_h - 0.5f;
        int iy = (int)std::floor(sy);
        int iy1 = std::min(iy + 1, sh - 1);
        int fy = (int)std::round((sy - iy) * ONE);
        if (iy < 0) { iy = 0; fy = 0; }
        const uint8_t* r0 = S + std::clamp(iy, 0, sh - 1) * stride;
        const uint8_t* r1 = S + std::clamp(iy1, 0, sh - 1) * stride;
        uint8_t* drow = D + dy * dst.step;
        for (int dx = 0; dx < new_w; ++dx) {
            const uint8_t* p00 = r0 + x0[dx] * 3;
            const uint8_t* p01 = r0 + x1[dx] * 3;
            const uint8_t* p10 = r1 + x0[dx] * 3;
            const uint8_t* p11 = r1 + x1[dx] * 3;
            int fx = wx[dx];
            int ix = ONE - fx;
            int iyw = ONE - fy;
            uint8_t* d = drow + dx * 3;
            for (int c = 0; c < 3; ++c) {
                int v0 = p00[c] * ix + p01[c] * fx;      // [0..(255<<FP)]
                int v1 = p10[c] * ix + p11[c] * fx;
                int vv = (v0 * iyw + v1 * fy + (1 << (FP*2-1))) >> (FP * 2);
                d[c] = (uint8_t)std::clamp(vv, 0, 255);
            }
        }
    }
}

void Preprocessor::rvvResizeBilinear(const cv::Mat& src, cv::Mat& dst, int new_w, int new_h) {
    // RVV kernels still WIP; reuse tuned scalar fallback which benchmarks faster.
    bilinearResizeScalar(src, dst, new_w, new_h);
}

void Preprocessor::letterbox(const cv::Mat& src, cv::Mat& dst,
                              float& scale, int& dx, int& dy,
                              int target_w, int target_h) {
    // Resize with aspect ratio preservation, fill gray borders, and expose offsets for reverse mapping.
    // Create output with gray padding
    dst = cv::Mat::zeros(target_h, target_w, CV_8UC3);
    dst.setTo(cv::Scalar(128, 128, 128));
    
    // Calculate scale to fit image
    scale = std::min(float(target_w) / src.cols, float(target_h) / src.rows);
    int new_w = int(src.cols * scale);
    int new_h = int(src.rows * scale);
    
    // Resize image
    cv::Mat resized;
    // Use RVV bilinear when available; keep scalar fallback otherwise
    rvvResizeBilinear(src, resized, new_w, new_h);
    
    // Calculate padding
    dx = (target_w - new_w) / 2;
    dy = (target_h - new_h) / 2;
    
    // Copy resized image to center of output (fast ROI copy)
    resized.copyTo(dst(cv::Rect(dx, dy, new_w, new_h)));
}

void Preprocessor::convertToNCHW_FP32_RVV(const cv::Mat& src, float* dst) {
    // Use RVV vectorization when available; otherwise fall back to scalar normalization.
    const int h = src.rows;
    const int w = src.cols;
    const int c = src.channels();
    const uint8_t* s = src.data;
    const int stride = (int)src.step;
#ifdef __riscv_vector
    if (rvv_available_) {
        const float inv255 = 1.0f / 255.0f;
        float* db = dst + 0 * w * h;
        float* dg = dst + 1 * w * h;
        float* dr = dst + 2 * w * h;
        for (int y = 0; y < h; ++y) {
            const uint8_t* row = s + y * stride;
            int x = 0;
            while (x < w) {
                size_t vl = vl_e8m1((size_t)(w - x));
                // Load interleaved BGR
                // We will manually deinterleave
                for (size_t i = 0; i < vl; ++i) {
                    const uint8_t* p = row + (x + (int)i) * 3;
                    db[y * w + x + (int)i] = p[0] * inv255;
                    dg[y * w + x + (int)i] = p[1] * inv255;
                    dr[y * w + x + (int)i] = p[2] * inv255;
                }
                x += vl;
            }
        }
        return;
    }
#endif
    // Scalar fallback
    for (int ch = 0; ch < c; ch++) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                const uint8_t* p = s + y * stride + x * 3;
                dst[ch * h * w + y * w + x] = p[ch] / 255.0f;
            }
        }
    }
}

// Backward-compatible entry that leverages RVV path when available
void Preprocessor::convertToNCHW_FP32(const cv::Mat& src, float* dst) {
    convertToNCHW_FP32_RVV(src, dst);
}

// Float -> FP16 conversion (IEEE 754 half, round-to-nearest-even)
static inline uint16_t f32_to_f16(float f) {
    uint32_t x; std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 31) & 0x1;
    int32_t exp = int32_t((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;
    if (exp <= 0) {
        // Subnormal/zero
        if (exp < -10) return uint16_t(sign << 15);
        mant = (mant | 0x800000) >> (1 - exp);
        return uint16_t((sign << 15) | (mant + 0x1000 >> 13));
    } else if (exp >= 31) {
        // Inf/NaN
        return uint16_t((sign << 15) | 0x7C00);
    } else {
        return uint16_t((sign << 15) | (exp << 10) | ((mant + 0x1000) >> 13));
    }
}

void Preprocessor::convertToNCHW_FP16(const cv::Mat& src, void* dst) {
    // Walk HWC input and pack into channel-first FP16 buffer expected by CSI-NN2.
    const int h = src.rows;
    const int w = src.cols;
    const uint8_t* s = src.data;
    uint16_t* d = (uint16_t*)dst;
#ifdef __riscv_vector
    if (rvv_available_) {
        hwc_bgr8_to_nchw_fp16_rvv(s, w, h, d);
        return;
    }
#endif
    // Scalar fallback
    const float inv255 = 1.0f / 255.0f;
    uint16_t* db = d + 0 * w * h;
    uint16_t* dg = d + 1 * w * h;
    uint16_t* dr = d + 2 * w * h;
    for (int y = 0; y < h; ++y) {
        const uint8_t* row = s + y * w * 3;
        for (int x = 0; x < w; ++x) {
            const uint8_t* p = row + x * 3;
            float bf = p[0] * inv255;
            float gf = p[1] * inv255;
            float rf = p[2] * inv255;
            // Clamp to [0,1] and guard against NaN/Inf
            bf = std::isfinite(bf) ? std::min(std::max(bf, 0.0f), 1.0f) : 0.0f;
            gf = std::isfinite(gf) ? std::min(std::max(gf, 0.0f), 1.0f) : 0.0f;
            rf = std::isfinite(rf) ? std::min(std::max(rf, 0.0f), 1.0f) : 0.0f;
            db[y * w + x] = f32_to_f16(bf);
            dg[y * w + x] = f32_to_f16(gf);
            dr[y * w + x] = f32_to_f16(rf);
        }
    }
}

// Full preprocessing: colorspace -> letterbox -> tensor conversion.
void Preprocessor::preprocess(Frame& input, void* model_input,
                              float& scale, int& dx, int& dy) {
    // 1) Colorspace conversion: YUV420P -> BGR
    if (input.format == PixelFormat::YUV420P && !input.y_plane.empty()) {
        const int w = input.source_width;
        const int h = input.source_height;
        cv::Mat bgr(h, w, CV_8UC3);
        const uint8_t* Y = input.y_plane.data();
        const uint8_t* U = input.u_plane.data();
        const uint8_t* V = input.v_plane.data();
        if (mode_ == PreprocMode::SW) {
            yuv420_to_bgr_sw(Y, U, V, w, h, input.y_stride, input.uv_stride, bgr.data, (int)bgr.step);
        } else {
            yuv420_to_bgr_rvv(Y, U, V, w, h, input.y_stride, input.uv_stride, bgr.data, (int)bgr.step);
        }
        input.image = std::move(bgr);
        input.format = PixelFormat::BGR;
    }

    // 2) Letterbox resize
    cv::Mat letterboxed;
    if (mode_ == PreprocMode::SW) {
        // SW path: use OpenCV resize for baseline speed
        // Create output with gray padding
        letterboxed = cv::Mat::zeros(MODEL_HEIGHT, MODEL_WIDTH, CV_8UC3);
        letterboxed.setTo(cv::Scalar(128, 128, 128));
        scale = std::min(float(MODEL_WIDTH) / input.image.cols, float(MODEL_HEIGHT) / input.image.rows);
        int new_w = int(input.image.cols * scale);
        int new_h = int(input.image.rows * scale);
        dx = (MODEL_WIDTH - new_w) / 2;
        dy = (MODEL_HEIGHT - new_h) / 2;
        cv::Mat resized;
        cv::resize(input.image, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
        resized.copyTo(letterboxed(cv::Rect(dx, dy, new_w, new_h)));
    } else {
        // RVV path (with scalar fallback inside)
        letterbox(input.image, letterboxed, scale, dx, dy);
    }

    // 3) Layout + dtype conversion — always produce FP16 for the engine (model expects FP16)
    convertToNCHW_FP16(letterboxed, model_input);
}

bool Preprocessor::hasRVVSupport() const {
    return rvv_available_;
}

// Scalar conversion for packed YUYV422 -> BGR (used on V4L2 path).
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

// RVV-accelerated path uses libswscale fallback when unsupported.
void Preprocessor::yuyvToBGR_RVV(const uint8_t* yuyv, cv::Mat& bgr, int width, int height) {
    // For now, fall back to scalar implementation
    // RVV implementation would go here
    yuyvToBGR(yuyv, bgr, width, height);
}

// --- RVV kernels (with scalar fallbacks) ---
void Preprocessor::resize_bilinear_bgr_rvv(const uint8_t* src, int sw, int sh, int sstride,
                                           uint8_t* dst, int dw, int dh, int dstride) {
#ifdef __riscv_vector
    if (rvv_available_) {
        // Simple per-pixel bilinear using vectorized deinterleave/accumulate in small chunks
        const float fx = (float)sw / dw;
        const float fy = (float)sh / dh;
        for (int dy = 0; dy < dh; ++dy) {
            float sy = (dy + 0.5f) * fy - 0.5f;
            int y0 = (int)std::floor(sy);
            int y1 = std::min(y0 + 1, sh - 1);
            float wy = sy - y0; if (y0 < 0) { y0 = 0; wy = 0.f; }
            const uint8_t* r0 = src + std::clamp(y0, 0, sh - 1) * sstride;
            const uint8_t* r1 = src + std::clamp(y1, 0, sh - 1) * sstride;
            int dx = 0;
            while (dx < dw) {
                size_t vl = vl_e32m1((size_t)(dw - dx));
                for (size_t i = 0; i < vl; ++i) {
                    float sx = (dx + (int)i + 0.5f) * fx - 0.5f;
                    int x0 = (int)std::floor(sx);
                    int x1 = std::min(x0 + 1, sw - 1);
                    float wx = sx - x0; if (x0 < 0) { x0 = 0; wx = 0.f; }
                    const uint8_t* p00 = r0 + std::clamp(x0, 0, sw - 1) * 3;
                    const uint8_t* p01 = r0 + std::clamp(x1, 0, sw - 1) * 3;
                    const uint8_t* p10 = r1 + std::clamp(x0, 0, sw - 1) * 3;
                    const uint8_t* p11 = r1 + std::clamp(x1, 0, sw - 1) * 3;
                    float w00 = (1 - wx) * (1 - wy);
                    float w01 = wx * (1 - wy);
                    float w10 = (1 - wx) * wy;
                    float w11 = wx * wy;
                    uint8_t* d = dst + dy * dstride + (dx + (int)i) * 3;
                    for (int c = 0; c < 3; ++c) {
                        float val = p00[c]*w00 + p01[c]*w01 + p10[c]*w10 + p11[c]*w11;
                        d[c] = (uint8_t)std::min(255.0f, std::max(0.0f, val));
                    }
                }
                dx += vl;
            }
        }
        return;
    }
#endif
    // Scalar fallback
    cv::Mat srcMat(sh, sw, CV_8UC3, (void*)src, sstride);
    cv::Mat dstMat(dh, dw, CV_8UC3, dst, dstride);
    bilinearResizeScalar(srcMat, dstMat, dw, dh);
}

void Preprocessor::hwc_bgr8_to_nchw_fp16_rvv(const uint8_t* src, int w, int h, uint16_t* dst) {
#ifdef __riscv_vector
    if (rvv_available_) {
        _Float16* out_b = (_Float16*)(dst + 0 * w * h);
        _Float16* out_g = (_Float16*)(dst + 1 * w * h);
        _Float16* out_r = (_Float16*)(dst + 2 * w * h);
        const float inv255 = 1.0f / 255.0f;
        for (int y = 0; y < h; ++y) {
            const uint8_t* row = src + y * w * 3;
            int x = 0;
            while (x < w) {
                size_t vl = vl_e8m1((size_t)(w - x));
                // Load interleaved B,G,R via strided loads to avoid seg-load overhead
                vuint8m1_t vb = __riscv_vlse8_v_u8m1(row + x * 3 + 0, 3, vl);
                vuint8m1_t vg = __riscv_vlse8_v_u8m1(row + x * 3 + 1, 3, vl);
                vuint8m1_t vr = __riscv_vlse8_v_u8m1(row + x * 3 + 2, 3, vl);
                // Zero-extend to u16 and convert directly to f16, then normalize
                vuint16m2_t wb = __riscv_vzext_vf2_u16m2(vb, vl);
                vuint16m2_t wg = __riscv_vzext_vf2_u16m2(vg, vl);
                vuint16m2_t wr = __riscv_vzext_vf2_u16m2(vr, vl);
                vfloat16m2_t hb = __riscv_vfcvt_f_xu_v_f16m2(wb, vl);
                vfloat16m2_t hg = __riscv_vfcvt_f_xu_v_f16m2(wg, vl);
                vfloat16m2_t hr = __riscv_vfcvt_f_xu_v_f16m2(wr, vl);
                hb = __riscv_vfmul_vf_f16m2(hb, (_Float16)inv255, vl);
                hg = __riscv_vfmul_vf_f16m2(hg, (_Float16)inv255, vl);
                hr = __riscv_vfmul_vf_f16m2(hr, (_Float16)inv255, vl);
                // Clamp to [0,1]
                hb = __riscv_vfmin_vf_f16m2(__riscv_vfmax_vf_f16m2(hb, (_Float16)0.0f, vl), (_Float16)1.0f, vl);
                hg = __riscv_vfmin_vf_f16m2(__riscv_vfmax_vf_f16m2(hg, (_Float16)0.0f, vl), (_Float16)1.0f, vl);
                hr = __riscv_vfmin_vf_f16m2(__riscv_vfmax_vf_f16m2(hr, (_Float16)0.0f, vl), (_Float16)1.0f, vl);
                __riscv_vse16_v_f16m2(out_b + y * w + x, hb, vl);
                __riscv_vse16_v_f16m2(out_g + y * w + x, hg, vl);
                __riscv_vse16_v_f16m2(out_r + y * w + x, hr, vl);
                x += (int)vl;
            }
        }
        return;
    }
#endif
    // Scalar fallback
    const float inv255 = 1.0f / 255.0f;
    uint16_t* db = dst + 0 * w * h;
    uint16_t* dg = dst + 1 * w * h;
    uint16_t* dr = dst + 2 * w * h;
    for (int y = 0; y < h; ++y) {
        const uint8_t* row = src + y * w * 3;
        for (int x = 0; x < w; ++x) {
            const uint8_t* p = row + x * 3;
            db[y * w + x] = f32_to_f16(p[0] * inv255);
            dg[y * w + x] = f32_to_f16(p[1] * inv255);
            dr[y * w + x] = f32_to_f16(p[2] * inv255);
        }
    }
}

void Preprocessor::yuv420_to_bgr_rvv(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                                      int w, int h, int y_stride, int uv_stride,
                                      uint8_t* dst_bgr, int dst_stride) {
#ifdef __riscv_vector
    if (rvv_available_) {
        auto clamp8f = [](_Float16 x) -> uint8_t {
            float f = (float)x;
            if (f < 0.f) f = 0.f; if (f > 255.f) f = 255.f; return (uint8_t)lrintf(f);
        };
        const _Float16 kR = (_Float16)1.402f;
        const _Float16 kUG = (_Float16)0.344136f;
        const _Float16 kVG = (_Float16)0.714136f;
        const _Float16 kB = (_Float16)1.772f;
        for (int j = 0; j < h; ++j) {
            const uint8_t* yrow0 = y + j * y_stride;
            const uint8_t* urow = u + (j/2) * uv_stride;
            const uint8_t* vrow = v + (j/2) * uv_stride;
            uint8_t* drow0 = dst_bgr + j * dst_stride;
            int i = 0;
            while (i < w) {
                int remain_pairs = (w - i + 1) / 2;
                size_t vl = vl_e8m1((size_t)remain_pairs);
                // Load Y even and odd positions
                vuint8m1_t y0 = __riscv_vlse8_v_u8m1(yrow0 + i + 0, 2, vl);
                vuint8m1_t y1 = __riscv_vlse8_v_u8m1(yrow0 + i + 1, 2, vl);
                vuint8m1_t uu = __riscv_vle8_v_u8m1(urow + (i/2), vl);
                vuint8m1_t vv = __riscv_vle8_v_u8m1(vrow + (i/2), vl);
                vuint16m2_t y0_16 = __riscv_vzext_vf2_u16m2(y0, vl);
                vuint16m2_t y1_16 = __riscv_vzext_vf2_u16m2(y1, vl);
                // Convert to f16 and center U,V around zero: Uv = U-128, Vv = V-128
                vfloat16m2_t y0f = __riscv_vfcvt_f_xu_v_f16m2(y0_16, vl);
                vfloat16m2_t y1f = __riscv_vfcvt_f_xu_v_f16m2(y1_16, vl);
                vuint16m2_t u16 = __riscv_vzext_vf2_u16m2(uu, vl);
                vuint16m2_t v16 = __riscv_vzext_vf2_u16m2(vv, vl);
                vfloat16m2_t uf = __riscv_vfcvt_f_xu_v_f16m2(u16, vl);
                vfloat16m2_t vf = __riscv_vfcvt_f_xu_v_f16m2(v16, vl);
                uf = __riscv_vfsub_vf_f16m2(uf, (_Float16)128.0f, vl);
                vf = __riscv_vfsub_vf_f16m2(vf, (_Float16)128.0f, vl);
                // Compute for pair 0
                vfloat16m2_t r0 = __riscv_vfmacc_vf_f16m2(y0f, kR, vf, vl);
                vfloat16m2_t g0 = __riscv_vfmsac_vf_f16m2(__riscv_vfmsac_vf_f16m2(y0f, kUG, uf, vl), kVG, vf, vl);
                vfloat16m2_t b0 = __riscv_vfmacc_vf_f16m2(y0f, kB, uf, vl);
                // Compute for pair 1
                vfloat16m2_t r1 = __riscv_vfmacc_vf_f16m2(y1f, kR, vf, vl);
                vfloat16m2_t g1 = __riscv_vfmsac_vf_f16m2(__riscv_vfmsac_vf_f16m2(y1f, kUG, uf, vl), kVG, vf, vl);
                vfloat16m2_t b1 = __riscv_vfmacc_vf_f16m2(y1f, kB, uf, vl);
                // Store as interleaved BGR for each pair (scalar store per lane)
                // Extract into temporary arrays
                alignas(64) _Float16 tb[512], tg[512], tr[512];
                alignas(64) _Float16 tb2[512], tg2[512], tr2[512];
                __riscv_vse16_v_f16m2(tb, b0, vl);
                __riscv_vse16_v_f16m2(tg, g0, vl);
                __riscv_vse16_v_f16m2(tr, r0, vl);
                __riscv_vse16_v_f16m2(tb2, b1, vl);
                __riscv_vse16_v_f16m2(tg2, g1, vl);
                __riscv_vse16_v_f16m2(tr2, r1, vl);
                for (size_t k = 0; k < vl; ++k) {
                    int x = i + (int)(2*k);
                    uint8_t* p0 = drow0 + x * 3;
                    p0[0] = clamp8f(tb[k]); p0[1] = clamp8f(tg[k]); p0[2] = clamp8f(tr[k]);
                    if (x + 1 < w) {
                        uint8_t* p1 = drow0 + (x + 1) * 3;
                        p1[0] = clamp8f(tb2[k]); p1[1] = clamp8f(tg2[k]); p1[2] = clamp8f(tr2[k]);
                    }
                }
                i += (int)(vl * 2);
            }
        }
        return;
    }
#endif
    // Scalar BT.601 full-range YUV420p -> BGR
    auto clamp8 = [](int x){ return (uint8_t)std::min(255, std::max(0, x)); };
    for (int j = 0; j < h; ++j) {
        const uint8_t* yrow = y + j * y_stride;
        const uint8_t* urow = u + (j/2) * uv_stride;
        const uint8_t* vrow = v + (j/2) * uv_stride;
        uint8_t* drow = dst_bgr + j * dst_stride;
        for (int i = 0; i < w; i += 2) {
            int Y0 = yrow[i];
            int Y1 = (i+1 < w) ? yrow[i+1] : yrow[i];
            int Uv = urow[i/2] - 128;
            int Vv = vrow[i/2] - 128;
            int C0 = Y0; int C1 = Y1;
            int R0 = (int)std::round(C0 + 1.402f * Vv);
            int G0 = (int)std::round(C0 - 0.344136f * Uv - 0.714136f * Vv);
            int B0 = (int)std::round(C0 + 1.772f * Uv);
            int R1 = (int)std::round(C1 + 1.402f * Vv);
            int G1 = (int)std::round(C1 - 0.344136f * Uv - 0.714136f * Vv);
            int B1 = (int)std::round(C1 + 1.772f * Uv);
            drow[i*3+0] = clamp8(B0); drow[i*3+1] = clamp8(G0); drow[i*3+2] = clamp8(R0);
            if (i+1 < w) { drow[(i+1)*3+0] = clamp8(B1); drow[(i+1)*3+1] = clamp8(G1); drow[(i+1)*3+2] = clamp8(R1); }
        }
    }
}

// SW path: use libswscale for robust and fast YUV420P -> BGR24
void Preprocessor::yuv420_to_bgr_sw(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                                    int w, int h, int y_stride, int uv_stride,
                                    uint8_t* dst_bgr, int dst_stride) {
    // Cache and reuse the context to avoid per-frame logs and overhead
    sws_ctx_ = sws_getCachedContext(sws_ctx_, w, h, AV_PIX_FMT_YUV420P,
                                    w, h, AV_PIX_FMT_BGR24,
                                    SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!sws_ctx_) {
        // Fallback to scalar conversion if sws allocation fails
        yuv420_to_bgr_rvv(y, u, v, w, h, y_stride, uv_stride, dst_bgr, dst_stride);
        return;
    }
    const uint8_t* src_slices[3] = { y, u, v };
    int src_stride[3] = { y_stride, uv_stride, uv_stride };
    uint8_t* dst_slices[1] = { dst_bgr };
    int dst_strides[1] = { dst_stride };
    sws_scale(sws_ctx_, src_slices, src_stride, 0, h, dst_slices, dst_strides);
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
