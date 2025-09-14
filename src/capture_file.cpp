#include "capture.hpp"
#include <iostream>
#include <chrono>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

namespace yolov5 {

class CaptureFile::Impl {
public:
    Impl() : format_ctx_(nullptr), codec_ctx_(nullptr), sws_ctx_(nullptr),
             video_stream_index_(-1), frame_(nullptr), frame_rgb_(nullptr),
             frame_counter_(0), width_(0), height_(0), fps_(0), frame_count_(0) {}
    
    ~Impl() {
        release();
    }
    
    bool init(const std::string& source) {
        // Extract filename from file:// URL
        std::string filename = source;
        if (filename.substr(0, 5) == "file:") {
            filename = filename.substr(5);
        }
        
        // Open input file
        if (avformat_open_input(&format_ctx_, filename.c_str(), nullptr, nullptr) < 0) {
            std::cerr << "Failed to open input file: " << filename << std::endl;
            return false;
        }
        
        // Find stream information
        if (avformat_find_stream_info(format_ctx_, nullptr) < 0) {
            std::cerr << "Failed to find stream info" << std::endl;
            release();
            return false;
        }
        
        // Find video stream
        for (unsigned int i = 0; i < format_ctx_->nb_streams; i++) {
            if (format_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                video_stream_index_ = i;
                break;
            }
        }
        
        if (video_stream_index_ == -1) {
            std::cerr << "No video stream found" << std::endl;
            release();
            return false;
        }
        
        AVStream* video_stream = format_ctx_->streams[video_stream_index_];
        AVCodecParameters* codecpar = video_stream->codecpar;
        
        // Find decoder
        const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
        if (!codec) {
            std::cerr << "Codec not found" << std::endl;
            release();
            return false;
        }
        
        // Create codec context
        codec_ctx_ = avcodec_alloc_context3(codec);
        if (!codec_ctx_) {
            std::cerr << "Failed to allocate codec context" << std::endl;
            release();
            return false;
        }
        
        // Copy codec parameters
        if (avcodec_parameters_to_context(codec_ctx_, codecpar) < 0) {
            std::cerr << "Failed to copy codec parameters" << std::endl;
            release();
            return false;
        }
        
        // Open codec
        if (avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
            std::cerr << "Failed to open codec" << std::endl;
            release();
            return false;
        }
        
        // Store video properties
        width_ = codec_ctx_->width;
        height_ = codec_ctx_->height;
        
        // Calculate FPS
        AVRational fps = av_guess_frame_rate(format_ctx_, video_stream, nullptr);
        fps_ = fps.num / (double)fps.den;
        
        // Get frame count (may be inaccurate for some formats)
        frame_count_ = video_stream->nb_frames;
        if (frame_count_ <= 0) {
            // Estimate from duration
            if (video_stream->duration > 0 && fps_ > 0) {
                frame_count_ = (int)(video_stream->duration * fps_ / AV_TIME_BASE);
            }
        }
        
        // Allocate frames
        frame_ = av_frame_alloc();
        frame_rgb_ = av_frame_alloc();
        
        if (!frame_ || !frame_rgb_) {
            std::cerr << "Failed to allocate frames" << std::endl;
            release();
            return false;
        }
        
        // Allocate buffer for RGB frame
        int num_bytes = av_image_get_buffer_size(AV_PIX_FMT_BGR24, width_, height_, 1);
        uint8_t* buffer = (uint8_t*)av_malloc(num_bytes);
        av_image_fill_arrays(frame_rgb_->data, frame_rgb_->linesize, buffer,
                             AV_PIX_FMT_BGR24, width_, height_, 1);
        
        // Create scaling context
        sws_ctx_ = sws_getContext(width_, height_, codec_ctx_->pix_fmt,
                                   width_, height_, AV_PIX_FMT_BGR24,
                                   SWS_BILINEAR, nullptr, nullptr, nullptr);
        
        if (!sws_ctx_) {
            std::cerr << "Failed to create scaling context" << std::endl;
            release();
            return false;
        }
        
        return true;
    }
    
    bool getFrame(Frame& frame) {
        if (!format_ctx_ || !codec_ctx_) {
            return false;
        }
        
        AVPacket* packet = av_packet_alloc();
        if (!packet) {
            return false;
        }
        
        while (av_read_frame(format_ctx_, packet) >= 0) {
            if (packet->stream_index == video_stream_index_) {
                // Send packet to decoder
                int ret = avcodec_send_packet(codec_ctx_, packet);
                if (ret < 0) {
                    av_packet_unref(packet);
                    continue;
                }
                
                // Receive frame from decoder
                ret = avcodec_receive_frame(codec_ctx_, frame_);
                if (ret == 0) {
                    // Convert to BGR
                    sws_scale(sws_ctx_, frame_->data, frame_->linesize, 0, height_,
                              frame_rgb_->data, frame_rgb_->linesize);
                    
                    // Create OpenCV Mat from frame data
                    cv::Mat bgr_mat(height_, width_, CV_8UC3, frame_rgb_->data[0],
                                    frame_rgb_->linesize[0]);
                    
                    // Fill Frame structure
                    frame.frame_id = frame_counter_++;
                    frame.image = bgr_mat.clone();  // Clone to avoid data dependency
                    frame.timestamp = std::chrono::steady_clock::now();
                    frame.source_width = width_;
                    frame.source_height = height_;
                    
                    av_packet_unref(packet);
                    av_packet_free(&packet);
                    return true;
                }
            }
            av_packet_unref(packet);
        }
        
        av_packet_free(&packet);
        
        return false;
    }
    
    void release() {
        if (frame_rgb_) {
            if (frame_rgb_->data[0]) {
                av_free(frame_rgb_->data[0]);
            }
            av_frame_free(&frame_rgb_);
            frame_rgb_ = nullptr;
        }
        
        if (frame_) {
            av_frame_free(&frame_);
            frame_ = nullptr;
        }
        
        if (sws_ctx_) {
            sws_freeContext(sws_ctx_);
            sws_ctx_ = nullptr;
        }
        
        if (codec_ctx_) {
            avcodec_free_context(&codec_ctx_);
            codec_ctx_ = nullptr;
        }
        
        if (format_ctx_) {
            avformat_close_input(&format_ctx_);
            format_ctx_ = nullptr;
        }
    }
    
    bool isOpened() const {
        return format_ctx_ != nullptr && codec_ctx_ != nullptr;
    }
    
    int getWidth() const { return width_; }
    int getHeight() const { return height_; }
    double getFPS() const { return fps_; }
    int getFrameCount() const { return frame_count_; }
    
private:
    AVFormatContext* format_ctx_;
    AVCodecContext* codec_ctx_;
    SwsContext* sws_ctx_;
    int video_stream_index_;
    AVFrame* frame_;
    AVFrame* frame_rgb_;
    uint64_t frame_counter_;
    int width_;
    int height_;
    double fps_;
    int frame_count_;
};

// CaptureFile implementation
CaptureFile::CaptureFile() : pImpl(std::make_unique<Impl>()) {}
CaptureFile::~CaptureFile() = default;

bool CaptureFile::init(const std::string& source) {
    return pImpl->init(source);
}

bool CaptureFile::getFrame(Frame& frame) {
    return pImpl->getFrame(frame);
}

void CaptureFile::release() {
    pImpl->release();
}

int CaptureFile::getWidth() const {
    return pImpl->getWidth();
}

int CaptureFile::getHeight() const {
    return pImpl->getHeight();
}

double CaptureFile::getFPS() const {
    return pImpl->getFPS();
}

int CaptureFile::getFrameCount() const {
    return pImpl->getFrameCount();
}

bool CaptureFile::isOpened() const {
    return pImpl->isOpened();
}

} // namespace yolov5