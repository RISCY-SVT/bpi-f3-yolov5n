#ifndef ENGINE_HPP
#define ENGINE_HPP

#include "types.hpp"
#include <memory>
#include <vector>
#include <string>

// Forward declarations for CSI-NN2 types
struct csinn_session;
struct csinn_tensor;

namespace yolov5 {

// Abstract inference engine interface
class IEngine {
public:
    virtual ~IEngine() = default;
    
    // Initialize model from weights file
    virtual bool init(const std::string& weights_path) = 0;
    
    // Run inference on preprocessed input
    // Input: NCHW FP16 data (1x3x384x640)
    // Output: Vector of detections
    virtual std::vector<Detection> infer(void* input_data) = 0;
    
    // Get model input dimensions
    virtual void getInputDims(int& batch, int& channels, int& height, int& width) const = 0;
    
    // Release resources
    virtual void release() = 0;
    
    // Check if engine is initialized
    virtual bool isInitialized() const = 0;
};

// CSI-NN2 implementation of inference engine
class EngineCSI : public IEngine {
public:
    EngineCSI();
    ~EngineCSI() override;
    
    bool init(const std::string& weights_path) override;
    std::vector<Detection> infer(void* input_data) override;
    void getInputDims(int& batch, int& channels, int& height, int& width) const override;
    void release() override;
    bool isInitialized() const override;
    
private:
    // CSI-NN2 specific implementation
    class Impl;
    std::unique_ptr<Impl> pImpl;
    
    // YOLOv5 specific post-processing
    std::vector<Detection> postprocess(void* output_data);
    
    // Non-maximum suppression
    std::vector<Detection> nms(std::vector<Detection>& detections, 
                                float conf_thresh, float iou_thresh);
};

// Factory function to create engine
std::unique_ptr<IEngine> createEngine(const std::string& type = "csi");

} // namespace yolov5

#endif // ENGINE_HPP