#include "engine.hpp"
#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <cerrno>
#include <cstdio>
#include <cstdlib>

/**
 * @file engine_csi.cpp
 * @brief CSI-NN2 engine integration for HHB-compiled YOLOv5n model.
 */

extern "C" {
#include "csi_nn.h"
#include "shl_utils.h"
#include "shl_c920v2.h"

// External functions from model.c
void *csinn_(char *params_base);
void csinn_update_input_and_run(struct csinn_tensor **input_tensors, void *sess);
}

namespace yolov5 {

// YOLOv5 configuration
static const int STRIDES[3] = {8, 16, 32};
static const float ANCHORS[18] = {
    10,13, 16,30, 33,23, 30,61, 62,45, 59,119,
    116,90, 156,198, 373,326
};

// COCO class names
const std::vector<std::string> COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

// Helper function to load binary file
/** @brief Read entire binary model file into aligned buffer. */
static char* get_binary_from_file(const char* filename, int* size) {
    std::FILE* fp = std::fopen(filename, "rb");
    if (!fp) {
        std::cerr << "[ERROR] Cannot open file: " << filename 
                  << " (errno=" << errno << ": " << std::strerror(errno) << ")" << std::endl;
        return nullptr;
    }
    
    std::fseek(fp, 0, SEEK_END);
    int file_size = std::ftell(fp);
    std::rewind(fp);
    
    char* buffer = nullptr;
    int ret = posix_memalign((void**)&buffer, 4096, file_size);
    if (ret || !buffer) {
        std::cerr << "[ERROR] posix_memalign failed" << std::endl;
        std::fclose(fp);
        return nullptr;
    }
    
    ret = std::fread(buffer, 1, file_size, fp);
    if (ret != file_size) {
        std::cerr << "[ERROR] fread failed (expected " << file_size << ", got " << ret << ")" << std::endl;
        std::free(buffer);
        std::fclose(fp);
        return nullptr;
    }
    
    std::fclose(fp);
    if (size) *size = file_size;
    return buffer;
}

// Create graph function - properly loads .bm file
/**
 * @brief Load HHB-produced binary model entirely from memory.
 *
 * Follows AGENTS.md mandate: never call csinn_* with file paths.
 */
static void* create_graph(const char* params_path) {
    int file_size = 0;
    char* params = get_binary_from_file(params_path, &file_size);
    if (!params) {
        std::cerr << "[ERROR] Failed to load file: " << params_path << std::endl;
        return nullptr;
    }
    
    // Check file type
    const char* suffix = params_path + (std::strlen(params_path) - 3);
    if (std::strcmp(suffix, ".bm") == 0) {
        // Process .bm file
        if (file_size < 4128 + sizeof(struct shl_bm_sections)) {
            std::cerr << "[ERROR] File too small for .bm format (size=" << file_size << ")" << std::endl;
            std::free(params);
            return nullptr;
        }
        
        struct shl_bm_sections* section = (struct shl_bm_sections*)(params + 4128);
        
        if (section->graph_offset) {
            // HHB integration: import YOLOv5n graph from in-memory BM blob (1x3x384x640 FP16 NCHW).
            // Model binaries are staged under cpu_model/hhb.bm as mandated by AGENTS.md.
            void* result = csinn_import_binary_model(params);
            if (!result) {
                std::cerr << "[ERROR] csinn_import_binary_model failed" << std::endl;
                std::free(params);
                return nullptr;
            }
            std::free(params);
            return result;
        } else {
            // Use csinn with offset
            void* result = csinn_(params + section->params_offset * 4096);
            // Note: params will be managed by CSI-NN2
            return result;
        }
    } else {
        // For .params files, call csinn_ directly
        suffix = params_path + (std::strlen(params_path) - 7);
        if (std::strcmp(suffix, ".params") == 0) {
            void* result = csinn_(params);
            return result;
        }
    }
    
    std::cerr << "[ERROR] Unknown file type: " << params_path << std::endl;
    std::free(params);
    return nullptr;
}

/**
 * @brief Wraps CSI-NN2 session lifecycle and YOLOv5 post-processing.
 *
 * One instance lives per inference worker. Manages session init, tensor
 * conversions, and delegates detection post-process to SHL helpers.
 */
class EngineCSI::Impl {
public:
    Impl() : session_(nullptr), input_tensor_(nullptr), initialized_(false), 
             conf_thresh_(0.25f), nms_thresh_(0.45f) {}
    
    ~Impl() {
        release();
    }
    
    /**
     * @brief Initialize CSI-NN2 session from HHB weights.
     * @param weights_path Path to cpu_model/hhb.bm (or .params fallback).
     */
    bool init(const std::string& weights_path) {
        if (initialized_) {
            return true;
        }
        
        // Load model using create_graph
        void* sess = create_graph(weights_path.c_str());
        if (!sess) {
            std::cerr << "[ERROR] Failed to load model from: " << weights_path << std::endl;
            return false;
        }
        
        session_ = (struct csinn_session*)sess;
        
        // Setup input tensor
        struct csinn_tensor* temp_input = csinn_alloc_tensor(nullptr);
        csinn_get_input(0, temp_input, session_);
        
        // Allocate aligned memory for input
        int input_size = csinn_tensor_byte_size(temp_input);
        void* input_data = shl_mem_alloc_aligned(input_size, 4096);
        if (!input_data) {
            std::cerr << "[ERROR] Failed to allocate input memory" << std::endl;
            csinn_free_tensor(temp_input);
            release();
            return false;
        }
        
        input_tensor_ = csinn_alloc_tensor(nullptr);
        csinn_tensor_copy(input_tensor_, temp_input);
        input_tensor_->data = input_data;
        csinn_free_tensor(temp_input);
        
        initialized_ = true;
        std::cout << "[INFO] Model initialized successfully" << std::endl;
        return true;
    }
    
    /** @brief Release session and allocated tensors. */
    void release() {
        if (input_tensor_) {
            if (input_tensor_->data) {
                shl_mem_free(input_tensor_->data);
            }
            csinn_free_tensor(input_tensor_);
            input_tensor_ = nullptr;
        }
        
        if (session_) {
            csinn_session_deinit(session_);
            csinn_free_session(session_);
            session_ = nullptr;
        }
        initialized_ = false;
    }
    
    bool isInitialized() const {
        return initialized_;
    }

    /**
     * @brief Run single inference pass and collect detections.
     * @param input_data Preprocessed tensor (FP16 NCHW for optimal path).
     * @param conf_thresh Confidence threshold applied before NMS.
     * @param nms_thresh IOU threshold for NMS.
     */
    std::vector<Detection> infer(void* input_data, float conf_thresh, float nms_thresh) {
        if (!initialized_ || !session_) {
            return {};
        }
        
        // Copy/convert input according to model's expected dtype
        int input_size = csinn_tensor_byte_size(input_tensor_);
        if (input_tensor_->dtype == CSINN_DTYPE_FLOAT16) {
            // Preprocessor provided FP16 buffer
            std::memcpy(input_tensor_->data, input_data, input_size);
        } else if (input_tensor_->dtype == CSINN_DTYPE_FLOAT32) {
            std::memcpy(input_tensor_->data, input_data, input_size);
        } else {
            // Fallback: convert from FP32 buffer
            void* converted_input = shl_c920v2_f32_to_input_dtype(0, (float*)input_data, session_);
            std::memcpy(input_tensor_->data, converted_input, input_size);
            shl_mem_free(converted_input);
        }
        
        // Run inference
        csinn_update_input_and_run(&input_tensor_, session_);
        
        // Get outputs and convert to FP32
        int output_num = csinn_get_output_number(session_);
        struct csinn_tensor** outputs = new struct csinn_tensor*[output_num];
        
        for (int i = 0; i < output_num; i++) {
            // Get original output tensor
            struct csinn_tensor* output = csinn_alloc_tensor(nullptr);
            csinn_get_output(i, output, session_);
            
            // Create new tensor for float32 conversion
            struct csinn_tensor* ret = csinn_alloc_tensor(nullptr);
            csinn_tensor_copy(ret, output);
            
            // Remove quantization info and set to float32
            if (ret->qinfo) {
                shl_mem_free(ret->qinfo);
                ret->qinfo = nullptr;
            }
            ret->quant_channel = 0;
            ret->dtype = CSINN_DTYPE_FLOAT32;
            
            // Convert data to float32
            ret->data = shl_c920v2_output_to_f32_dtype(i, output->data, session_);
            outputs[i] = ret;
            
            // Free original output tensor
            csinn_free_tensor(output);
        }
        
        // Post-process
        std::vector<Detection> detections = postprocess(outputs, output_num, conf_thresh, nms_thresh);
        
        // Free outputs
        for (int i = 0; i < output_num; i++) {
            if (outputs[i]->data) {
                shl_mem_free(outputs[i]->data);
            }
            csinn_free_tensor(outputs[i]);
        }
        delete[] outputs;
        
        return detections;
    }
    
private:
    /**
     * @brief Convert CSI-NN2 raw outputs into Detection list via SHL helpers.
     */
    std::vector<Detection> postprocess(struct csinn_tensor** outputs, int output_num,
                                        float conf_thresh, float nms_thresh) {
        // Use CSI-NN2's built-in YOLOv5 postprocessing
        struct shl_yolov5_params* params = (struct shl_yolov5_params*)shl_mem_alloc(sizeof(struct shl_yolov5_params));
        if (!params) {
            std::cerr << "[ERROR] Failed to allocate YOLO params" << std::endl;
            return {};
        }
        
        params->conf_thres = conf_thresh;
        params->iou_thres = nms_thresh;
        params->strides[0] = STRIDES[0];
        params->strides[1] = STRIDES[1];
        params->strides[2] = STRIDES[2];
        
        for (int i = 0; i < 18; i++) {
            params->anchors[i] = ANCHORS[i];
        }
        
        // Allocate boxes
        struct shl_yolov5_box* boxes = new shl_yolov5_box[1000];
        int box_count = shl_c920v2_detect_yolov5_postprocess(outputs, boxes, params);
        
        // Convert to Detection format
        std::vector<Detection> detections;
        for (int i = 0; i < box_count; i++) {
            Detection det;
            det.x1 = boxes[i].x1;
            det.y1 = boxes[i].y1;
            det.x2 = boxes[i].x2;
            det.y2 = boxes[i].y2;
            det.confidence = boxes[i].score;
            det.class_id = boxes[i].label;
            if (det.class_id < COCO_CLASSES.size()) {
                det.label = COCO_CLASSES[det.class_id];
            }
            detections.push_back(det);
        }
        
        delete[] boxes;
        shl_mem_free(params);
        
        return detections;
    }
    
    struct csinn_session* session_;
    struct csinn_tensor* input_tensor_;
    bool initialized_;
    float conf_thresh_;
    float nms_thresh_;
};

// EngineCSI implementation
EngineCSI::EngineCSI() : pImpl(std::make_unique<Impl>()) {}

EngineCSI::~EngineCSI() = default;

bool EngineCSI::init(const std::string& weights_path) {
    return pImpl->init(weights_path);
}

std::vector<Detection> EngineCSI::infer(void* input_data) {
    return pImpl->infer(input_data, 0.25f, 0.45f);  // Default thresholds
}

void EngineCSI::getInputDims(int& batch, int& channels, int& height, int& width) const {
    batch = 1;
    channels = MODEL_CHANNELS;
    height = MODEL_HEIGHT;
    width = MODEL_WIDTH;
}

void EngineCSI::release() {
    pImpl->release();
}

bool EngineCSI::isInitialized() const {
    return pImpl->isInitialized();
}

// Factory function
std::unique_ptr<IEngine> createEngine(const std::string& type) {
    if (type == "csi") {
        return std::make_unique<EngineCSI>();
    }
    return nullptr;
}

} // namespace yolov5
