#include "capture.hpp"
#include "preprocess.hpp"
#include <iostream>
#include <cstring>
#include <cerrno>
#include <set>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <chrono>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

namespace yolov5 {

class CaptureV4L2::Impl {
public:
    struct Buffer {
        void* start;
        size_t length;
    };
    
    Impl() : fd_(-1), buffers_(nullptr), n_buffers_(0), 
             width_(640), height_(480), fps_(30), frame_counter_(0),
             format_(0), mjpeg_decoder_ctx_(nullptr), mjpeg_frame_(nullptr),
             mjpeg_frame_rgb_(nullptr), sws_ctx_(nullptr) {}
    
    ~Impl() {
        release();
    }
    
    bool init(const std::string& source) {
        // Extract device path from v4l2:// URL
        std::string device = source;
        if (device.substr(0, 6) == "v4l2:") {
            device = device.substr(6);
        }
        
        // Open device
        fd_ = open(device.c_str(), O_RDWR | O_NONBLOCK);
        if (fd_ == -1) {
            std::cerr << "Failed to open V4L2 device: " << device << std::endl;
            return false;
        }
        
        // Query capabilities
        struct v4l2_capability cap;
        if (ioctl(fd_, VIDIOC_QUERYCAP, &cap) == -1) {
            std::cerr << "Failed to query V4L2 capabilities" << std::endl;
            release();
            return false;
        }
        
        if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
            std::cerr << "Device does not support video capture" << std::endl;
            release();
            return false;
        }
        
        if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
            std::cerr << "Device does not support streaming" << std::endl;
            release();
            return false;
        }
        
        // List supported pixel formats (debug)
        {
            std::set<uint32_t> fmts;
            struct v4l2_fmtdesc fdesc;
            memset(&fdesc, 0, sizeof(fdesc));
            fdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            while (ioctl(fd_, VIDIOC_ENUM_FMT, &fdesc) == 0) {
                fmts.insert(fdesc.pixelformat);
                fdesc.index++;
            }
            std::cout << "[DEBUG] V4L2 supported formats:";
            for (auto f : fmts) {
                char fourcc[5] = { (char)(f & 0xFF), (char)((f >> 8) & 0xFF), (char)((f >> 16) & 0xFF), (char)((f >> 24) & 0xFF), 0 };
                std::cout << " " << fourcc;
            }
            std::cout << std::endl;
        }

        // Try to set format - prefer YUYV, fall back to MJPEG
        if (!setFormat(V4L2_PIX_FMT_YUYV)) {
            if (!setFormat(V4L2_PIX_FMT_MJPEG)) {
                std::cerr << "Failed to set video format (tried YUYV and MJPEG)" << std::endl;
                release();
                return false;
            }
            format_ = V4L2_PIX_FMT_MJPEG;
            if (!initMJPEGDecoder()) {
                release();
                return false;
            }
        } else {
            format_ = V4L2_PIX_FMT_YUYV;
        }
        
        // Initialize buffers
        if (!initBuffers()) {
            release();
            return false;
        }
        
        // Start streaming
        if (!startStreaming()) {
            release();
            return false;
        }
        
        return true;
    }
    
    bool getFrame(Frame& frame) {
        if (fd_ == -1) {
            return false;
        }
        
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        
        // Dequeue buffer
        if (ioctl(fd_, VIDIOC_DQBUF, &buf) == -1) {
            if (errno == EAGAIN) {
                return false;  // No frame available
            }
            std::cerr << "Failed to dequeue buffer" << std::endl;
            return false;
        }
        
        // Process frame based on format
        cv::Mat bgr_image;
        if (format_ == V4L2_PIX_FMT_YUYV) {
            // Convert YUYV to BGR
            Preprocessor proc;
            proc.yuyvToBGR((uint8_t*)buffers_[buf.index].start, bgr_image, width_, height_);
        } else if (format_ == V4L2_PIX_FMT_MJPEG) {
            // Decode MJPEG to BGR
            if (!decodeMJPEG(buffers_[buf.index].start, buf.bytesused, bgr_image)) {
                // Queue buffer back
                ioctl(fd_, VIDIOC_QBUF, &buf);
                return false;
            }
        }
        
        // Fill Frame structure
        frame.frame_id = frame_counter_++;
        frame.image = bgr_image;
        frame.timestamp = std::chrono::steady_clock::now();
        frame.source_width = width_;
        frame.source_height = height_;
        
        // Queue buffer back
        if (ioctl(fd_, VIDIOC_QBUF, &buf) == -1) {
            std::cerr << "Failed to queue buffer" << std::endl;
            return false;
        }
        
        return true;
    }
    
    void release() {
        stopStreaming();
        
        if (buffers_) {
            for (unsigned int i = 0; i < n_buffers_; i++) {
                munmap(buffers_[i].start, buffers_[i].length);
            }
            delete[] buffers_;
            buffers_ = nullptr;
        }
        
        if (fd_ != -1) {
            close(fd_);
            fd_ = -1;
        }
        
        releaseMJPEGDecoder();
    }
    
    bool isOpened() const {
        return fd_ != -1;
    }
    
    int getWidth() const { return width_; }
    int getHeight() const { return height_; }
    double getFPS() const { return fps_; }
    int getFrameCount() const { return -1; }  // Live stream
    
private:
    bool setFormat(uint32_t pixel_format) {
        struct v4l2_format fmt;
        memset(&fmt, 0, sizeof(fmt));
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = width_;
        fmt.fmt.pix.height = height_;
        fmt.fmt.pix.pixelformat = pixel_format;
        fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;
        
        if (ioctl(fd_, VIDIOC_S_FMT, &fmt) == -1) {
            return false;
        }
        
        // Update actual dimensions
        width_ = fmt.fmt.pix.width;
        height_ = fmt.fmt.pix.height;
        
        return true;
    }
    
    bool initBuffers() {
        struct v4l2_requestbuffers req;
        memset(&req, 0, sizeof(req));
        req.count = 4;  // Request 4 buffers
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
        
        if (ioctl(fd_, VIDIOC_REQBUFS, &req) == -1) {
            std::cerr << "Failed to request buffers" << std::endl;
            return false;
        }
        
        if (req.count < 2) {
            std::cerr << "Insufficient buffer memory" << std::endl;
            return false;
        }
        
        buffers_ = new Buffer[req.count];
        n_buffers_ = req.count;
        
        for (unsigned int i = 0; i < n_buffers_; i++) {
            struct v4l2_buffer buf;
            memset(&buf, 0, sizeof(buf));
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;
            
            if (ioctl(fd_, VIDIOC_QUERYBUF, &buf) == -1) {
                std::cerr << "Failed to query buffer" << std::endl;
                return false;
            }
            
            buffers_[i].length = buf.length;
            buffers_[i].start = mmap(nullptr, buf.length,
                                     PROT_READ | PROT_WRITE, MAP_SHARED,
                                     fd_, buf.m.offset);
            
            if (buffers_[i].start == MAP_FAILED) {
                std::cerr << "Failed to map buffer" << std::endl;
                return false;
            }
            
            // Queue buffer
            if (ioctl(fd_, VIDIOC_QBUF, &buf) == -1) {
                std::cerr << "Failed to queue buffer" << std::endl;
                return false;
            }
        }
        
        return true;
    }
    
    bool startStreaming() {
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(fd_, VIDIOC_STREAMON, &type) == -1) {
            std::cerr << "Failed to start streaming" << std::endl;
            return false;
        }
        return true;
    }
    
    void stopStreaming() {
        if (fd_ != -1) {
            enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            ioctl(fd_, VIDIOC_STREAMOFF, &type);
        }
    }
    
    bool initMJPEGDecoder() {
        const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_MJPEG);
        if (!codec) {
            std::cerr << "MJPEG codec not found" << std::endl;
            return false;
        }
        
        mjpeg_decoder_ctx_ = avcodec_alloc_context3(codec);
        if (!mjpeg_decoder_ctx_) {
            std::cerr << "Failed to allocate MJPEG decoder context" << std::endl;
            return false;
        }
        
        if (avcodec_open2(mjpeg_decoder_ctx_, codec, nullptr) < 0) {
            std::cerr << "Failed to open MJPEG decoder" << std::endl;
            avcodec_free_context(&mjpeg_decoder_ctx_);
            mjpeg_decoder_ctx_ = nullptr;
            return false;
        }
        
        mjpeg_frame_ = av_frame_alloc();
        mjpeg_frame_rgb_ = av_frame_alloc();
        
        if (!mjpeg_frame_ || !mjpeg_frame_rgb_) {
            releaseMJPEGDecoder();
            return false;
        }
        
        // Allocate RGB buffer
        int num_bytes = av_image_get_buffer_size(AV_PIX_FMT_BGR24, width_, height_, 1);
        uint8_t* buffer = (uint8_t*)av_malloc(num_bytes);
        av_image_fill_arrays(mjpeg_frame_rgb_->data, mjpeg_frame_rgb_->linesize, buffer,
                             AV_PIX_FMT_BGR24, width_, height_, 1);
        
        return true;
    }
    
    void releaseMJPEGDecoder() {
        if (sws_ctx_) {
            sws_freeContext(sws_ctx_);
            sws_ctx_ = nullptr;
        }
        
        if (mjpeg_frame_rgb_) {
            if (mjpeg_frame_rgb_->data[0]) {
                av_free(mjpeg_frame_rgb_->data[0]);
            }
            av_frame_free(&mjpeg_frame_rgb_);
            mjpeg_frame_rgb_ = nullptr;
        }
        
        if (mjpeg_frame_) {
            av_frame_free(&mjpeg_frame_);
            mjpeg_frame_ = nullptr;
        }
        
        if (mjpeg_decoder_ctx_) {
            avcodec_free_context(&mjpeg_decoder_ctx_);
            mjpeg_decoder_ctx_ = nullptr;
        }
    }
    
    bool decodeMJPEG(void* data, size_t size, cv::Mat& bgr_image) {
        if (!mjpeg_decoder_ctx_) {
            return false;
        }
        
        AVPacket* packet = av_packet_alloc();
        if (!packet) {
            return false;
        }
        packet->data = (uint8_t*)data;
        packet->size = size;
        
        // Send packet to decoder
        int ret = avcodec_send_packet(mjpeg_decoder_ctx_, packet);
        if (ret < 0) {
            av_packet_free(&packet);
            return false;
        }
        
        // Receive frame
        ret = avcodec_receive_frame(mjpeg_decoder_ctx_, mjpeg_frame_);
        av_packet_free(&packet);
        if (ret < 0) {
            return false;
        }
        
        // Create or update scaling context if needed
        if (!sws_ctx_ || mjpeg_frame_->width != width_ || mjpeg_frame_->height != height_) {
            if (sws_ctx_) {
                sws_freeContext(sws_ctx_);
            }
            sws_ctx_ = sws_getContext(mjpeg_frame_->width, mjpeg_frame_->height, 
                                       (AVPixelFormat)mjpeg_frame_->format,
                                       width_, height_, AV_PIX_FMT_BGR24,
                                       SWS_BILINEAR, nullptr, nullptr, nullptr);
        }
        
        // Convert to BGR
        sws_scale(sws_ctx_, mjpeg_frame_->data, mjpeg_frame_->linesize, 0, mjpeg_frame_->height,
                  mjpeg_frame_rgb_->data, mjpeg_frame_rgb_->linesize);
        
        // Create OpenCV Mat
        bgr_image = cv::Mat(height_, width_, CV_8UC3, mjpeg_frame_rgb_->data[0],
                            mjpeg_frame_rgb_->linesize[0]).clone();
        
        return true;
    }
    
    int fd_;
    Buffer* buffers_;
    unsigned int n_buffers_;
    int width_;
    int height_;
    double fps_;
    uint64_t frame_counter_;
    uint32_t format_;
    
    // MJPEG decoding
    AVCodecContext* mjpeg_decoder_ctx_;
    AVFrame* mjpeg_frame_;
    AVFrame* mjpeg_frame_rgb_;
    SwsContext* sws_ctx_;
};

// CaptureV4L2 implementation
CaptureV4L2::CaptureV4L2() : pImpl(std::make_unique<Impl>()) {}
CaptureV4L2::~CaptureV4L2() = default;

bool CaptureV4L2::init(const std::string& source) {
    return pImpl->init(source);
}

bool CaptureV4L2::getFrame(Frame& frame) {
    return pImpl->getFrame(frame);
}

void CaptureV4L2::release() {
    pImpl->release();
}

int CaptureV4L2::getWidth() const {
    return pImpl->getWidth();
}

int CaptureV4L2::getHeight() const {
    return pImpl->getHeight();
}

double CaptureV4L2::getFPS() const {
    return pImpl->getFPS();
}

int CaptureV4L2::getFrameCount() const {
    return pImpl->getFrameCount();
}

bool CaptureV4L2::isOpened() const {
    return pImpl->isOpened();
}

// Factory function
std::unique_ptr<ICapture> createCapture(const std::string& source) {
    if (source.substr(0, 5) == "file:" || source.find(".") != std::string::npos) {
        return std::make_unique<CaptureFile>();
    } else if (source.substr(0, 5) == "v4l2:" || source.find("/dev/video") == 0) {
        return std::make_unique<CaptureV4L2>();
    }
    return nullptr;
}

} // namespace yolov5
