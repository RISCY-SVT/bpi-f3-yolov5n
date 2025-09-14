// Simple test for modular pipeline without video codecs
#include "types.hpp"
#include "preprocess.hpp"
#include "engine.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace yolov5;

// Simple PPM image reader
bool readPPM(const std::string& filename, cv::Mat& image) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        return false;
    }
    
    std::string magic;
    int width, height, maxval;
    file >> magic >> width >> height >> maxval;
    
    if (magic != "P6" || maxval != 255) {
        return false;
    }
    
    // Skip whitespace
    file.get();
    
    // Read RGB data
    image = cv::Mat(height, width, CV_8UC3);
    file.read((char*)image.data, width * height * 3);
    
    // Convert RGB to BGR
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    
    return true;
}

// Simple PPM writer
bool writePPM(const std::string& filename, const cv::Mat& image) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        return false;
    }
    
    // Convert BGR to RGB
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    
    // Write PPM header
    file << "P6\n" << rgb.cols << " " << rgb.rows << "\n255\n";
    
    // Write pixel data
    file.write((char*)rgb.data, rgb.cols * rgb.rows * 3);
    
    return true;
}

// Draw bounding box
void drawBox(cv::Mat& image, const Detection& det, float scale, int dx, int dy) {
    // Scale back to original image coordinates
    int x1 = (det.x1 - dx) / scale;
    int y1 = (det.y1 - dy) / scale;
    int x2 = (det.x2 - dx) / scale;
    int y2 = (det.y2 - dy) / scale;
    
    // Clip to image bounds
    x1 = std::max(0, std::min(image.cols - 1, x1));
    y1 = std::max(0, std::min(image.rows - 1, y1));
    x2 = std::max(0, std::min(image.cols - 1, x2));
    y2 = std::max(0, std::min(image.rows - 1, y2));
    
    // Draw rectangle (green color)
    for (int x = x1; x <= x2; x++) {
        if (y1 >= 0 && y1 < image.rows) {
            image.at<cv::Vec3b>(y1, x) = cv::Vec3b(0, 255, 0);
        }
        if (y2 >= 0 && y2 < image.rows) {
            image.at<cv::Vec3b>(y2, x) = cv::Vec3b(0, 255, 0);
        }
    }
    for (int y = y1; y <= y2; y++) {
        if (x1 >= 0 && x1 < image.cols) {
            image.at<cv::Vec3b>(y, x1) = cv::Vec3b(0, 255, 0);
        }
        if (x2 >= 0 && x2 < image.cols) {
            image.at<cv::Vec3b>(y, x2) = cv::Vec3b(0, 255, 0);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <input.ppm> <output.ppm>" << std::endl;
        return 1;
    }
    
    std::string input_file = argv[1];
    std::string output_file = argv[2];
    
    std::cout << "YOLOv5n Modular Pipeline Test\n";
    std::cout << "==============================\n";
    
    // Load image
    cv::Mat image;
    if (!readPPM(input_file, image)) {
        std::cerr << "Failed to read input image: " << input_file << std::endl;
        return 1;
    }
    std::cout << "Loaded image: " << image.cols << "x" << image.rows << std::endl;
    
    // Create engine
    auto engine = createEngine("csi");
    if (!engine || !engine->init("cpu_model/hhb.bm")) {
        std::cerr << "Failed to initialize engine" << std::endl;
        return 1;
    }
    std::cout << "Engine initialized\n";
    
    // Create preprocessor
    Preprocessor preprocessor;
    std::cout << "RVV support: " << (preprocessor.hasRVVSupport() ? "Yes" : "No") << std::endl;
    
    // Create frame
    Frame frame;
    frame.frame_id = 0;
    frame.image = image;
    frame.timestamp = std::chrono::steady_clock::now();
    
    // Allocate model input
    const size_t input_size = MODEL_CHANNELS * MODEL_HEIGHT * MODEL_WIDTH * sizeof(float);
    float* model_input = new float[MODEL_CHANNELS * MODEL_HEIGHT * MODEL_WIDTH];
    
    // Preprocess
    float scale;
    int dx, dy;
    auto prep_start = std::chrono::steady_clock::now();
    preprocessor.preprocess(frame, model_input, scale, dx, dy);
    auto prep_end = std::chrono::steady_clock::now();
    
    std::cout << "Preprocessing done. Scale: " << scale 
              << ", offset: (" << dx << ", " << dy << ")" << std::endl;
    
    // Run inference
    auto inf_start = std::chrono::steady_clock::now();
    std::vector<Detection> detections = engine->infer(model_input);
    auto inf_end = std::chrono::steady_clock::now();
    
    // Calculate times
    auto prep_ms = std::chrono::duration_cast<std::chrono::milliseconds>(prep_end - prep_start).count();
    auto inf_ms = std::chrono::duration_cast<std::chrono::milliseconds>(inf_end - inf_start).count();
    
    std::cout << "\nTiming:\n";
    std::cout << "  Preprocessing: " << prep_ms << "ms\n";
    std::cout << "  Inference: " << inf_ms << "ms\n";
    std::cout << "  Total: " << (prep_ms + inf_ms) << "ms\n";
    std::cout << "  FPS: " << std::fixed << std::setprecision(2) 
              << 1000.0 / (prep_ms + inf_ms) << "\n";
    
    std::cout << "\nFound " << detections.size() << " detections:\n";
    
    // Draw detections
    cv::Mat output = image.clone();
    for (const auto& det : detections) {
        // Rescale coordinates
        Detection scaled_det = det;
        scaled_det.x1 = (det.x1 - dx) / scale;
        scaled_det.y1 = (det.y1 - dy) / scale;
        scaled_det.x2 = (det.x2 - dx) / scale;
        scaled_det.y2 = (det.y2 - dy) / scale;
        
        std::cout << "  [" << (int)scaled_det.x1 << "," << (int)scaled_det.y1 << ","
                  << (int)scaled_det.x2 << "," << (int)scaled_det.y2 << "] "
                  << det.label << ": " << std::fixed << std::setprecision(2) 
                  << det.confidence << std::endl;
        
        drawBox(output, det, scale, dx, dy);
    }
    
    // Save output
    if (!writePPM(output_file, output)) {
        std::cerr << "Failed to write output image: " << output_file << std::endl;
        return 1;
    }
    std::cout << "\nOutput saved to: " << output_file << std::endl;
    
    // Cleanup
    delete[] model_input;
    
    return 0;
}
