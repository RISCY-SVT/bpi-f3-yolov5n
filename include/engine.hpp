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

/**
 * @file engine.hpp
 * @brief Declares the inference engine abstraction and CSI-NN2 implementation.
 *
 * Engines convert preprocessed model inputs into detection vectors and are
 * owned by dedicated inference worker threads. CSI-NN2 bridges to HHB-generated
 * YOLOv5n networks running entirely from in-memory binaries.
 */

/**
 * @brief Abstract inference engine used by pipeline workers.
 * @threading One instance per inference worker thread.
 * @lifecycle init(weights) → repeated infer() → release().
 */
class IEngine {
public:
    virtual ~IEngine() = default;

    /**
     * @brief Initialize engine state from compiled HHB weights.
     * @param weights_path Path to `cpu_model/hhb.bm` staged on device/host.
     * @return True if the session is ready for inference.
     */
    virtual bool init(const std::string& weights_path) = 0;

    /**
     * @brief Run inference on a single preprocessed tensor.
     * @param input_data Pointer to FP16/FP32 NCHW buffer sized for 1x3x384x640.
     * @return Vector of filtered detections (already NMSed).
     */
    virtual std::vector<Detection> infer(void* input_data) = 0;

    /**
     * @brief Report static model input dimensions.
     */
    virtual void getInputDims(int& batch, int& channels, int& height, int& width) const = 0;

    /**
     * @brief Release engine-owned buffers and sessions.
     */
    virtual void release() = 0;

    /**
     * @brief Signal whether init() succeeded and resources are live.
     */
    virtual bool isInitialized() const = 0;
};

/**
 * @brief CSI-NN2 backed engine for YOLOv5n HHB binaries.
 * @threading One per pipeline worker thread; manages a dedicated `csinn_session`.
 * @ownership Owns model session, input tensor buffers, and conversions.
 */
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
    class Impl;
    std::unique_ptr<Impl> pImpl; //!< Hides CSI-NN2 specific state.
};

/**
 * @brief Factory for engine implementations (currently CSI-NN2 only).
 * @param type Engine identifier such as "csi".
 */
std::unique_ptr<IEngine> createEngine(const std::string& type = "csi");

} // namespace yolov5

#endif // ENGINE_HPP
