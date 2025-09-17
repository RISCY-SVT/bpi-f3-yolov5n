#include "capture.hpp"
#include <iostream>
#include <chrono>

/**
 * @file capture_file.cpp
 * @brief FFmpeg-backed file capture implementation feeding pipeline frames.
 */

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
}

namespace yolov5 {

/**
 * @brief Wraps FFmpeg demux/decoder state for file capture.
 *
 * The object lives solely on the capture thread and decodes frames into Frame
 * structures backed by planar YUV data so preprocessing can handle colorspace.
 */
class CaptureFile::Impl {
public:
    Impl() : format_ctx_(nullptr), codec_ctx_(nullptr),
             video_stream_index_(-1), frame_(nullptr),
             frame_counter_(0), width_(0), height_(0), fps_(0), frame_count_(0) {}
    
    ~Impl() {
        release();
    }
    
    /**
     * @brief Open media file and prepare decoder.
     * @param source CLI-style source string (file:/path or bare path).
     */
    bool init(const std::string& source) {
        // Extract filename from file:// URL; bare paths already work.
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
        
        // Allocate frames used for decoder output reuse.
        frame_ = av_frame_alloc();

        if (!frame_) {
            std::cerr << "Failed to allocate frames" << std::endl;
            release();
            return false;
        }
        // We keep decoder's native pixel format and will handle colorspace in preprocess
        return true;
    }

    /**
     * @brief Decode next frame and populate Frame object.
     *
     * Frames are exported as planar YUV420 when available. Preprocess stage is
     * responsible for colorspace conversion.
     */
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
                    // Fill Frame structure with YUV420P planes; BGR produced later in preprocess.
                    frame.frame_id = frame_counter_++;
                    // Assume YUV420P layout; if not, we silently skip for now
                    frame.source_width = width_;
                    frame.source_height = height_;
                    frame.timestamp = std::chrono::steady_clock::now();
                    if (frame_->format == AV_PIX_FMT_YUV420P || frame_->format == AV_PIX_FMT_YUVJ420P) {
                        int y_stride = frame_->linesize[0];
                        int u_stride = frame_->linesize[1];
                        int v_stride = frame_->linesize[2];
                        frame.format = PixelFormat::YUV420P;
                        frame.y_stride = width_;
                        frame.uv_stride = width_ / 2;
                        frame.y_plane.resize(width_ * height_);
                        frame.u_plane.resize((width_/2) * (height_/2));
                        frame.v_plane.resize((width_/2) * (height_/2));
                        // Copy with stride handling
                        for (int j = 0; j < height_; ++j) {
                            memcpy(&frame.y_plane[j*width_], frame_->data[0] + j * y_stride, width_);
                        }
                        for (int j = 0; j < height_/2; ++j) {
                            memcpy(&frame.u_plane[j*(width_/2)], frame_->data[1] + j * u_stride, width_/2);
                            memcpy(&frame.v_plane[j*(width_/2)], frame_->data[2] + j * v_stride, width_/2);
                        }
                        // Do not set frame.image here to avoid extra conversion work; preprocess will populate if needed
                    } else {
                        // Fallback: unsupported pix_fmt; create empty image to avoid downstream crashes.
                        frame.format = PixelFormat::BGR;
                        frame.image = cv::Mat(height_, width_, CV_8UC3);
                        frame.image.setTo(cv::Scalar(0,0,0));
                    }
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
    
    /**
     * @brief Tear down FFmpeg objects.
     */
    void release() {
        if (frame_) {
            av_frame_free(&frame_);
            frame_ = nullptr;
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
    int video_stream_index_;
    AVFrame* frame_;
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
