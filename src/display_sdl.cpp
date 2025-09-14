#ifdef HAVE_SDL2
#include "display.hpp"
#include <SDL.h>
#include <iostream>

namespace yolov5 {

class SdlDisplay : public IDisplay {
public:
    SdlDisplay() : window_(nullptr), renderer_(nullptr), texture_(nullptr), w_(0), h_(0) {}
    ~SdlDisplay() override { close(); }

    bool init(int width, int height, const std::string& title) override {
        if (SDL_Init(SDL_INIT_VIDEO) != 0) {
            std::cerr << "[WARN] SDL_Init failed: " << SDL_GetError() << std::endl;
            return false;
        }
        w_ = width; h_ = height;
        window_ = SDL_CreateWindow(title.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, 0);
        if (!window_) { std::cerr << "[WARN] SDL_CreateWindow failed: " << SDL_GetError() << std::endl; return false; }
        renderer_ = SDL_CreateRenderer(window_, -1, SDL_RENDERER_ACCELERATED);
        if (!renderer_) { std::cerr << "[WARN] SDL_CreateRenderer failed: " << SDL_GetError() << std::endl; return false; }
        texture_ = SDL_CreateTexture(renderer_, SDL_PIXELFORMAT_BGR24, SDL_TEXTUREACCESS_STREAMING, width, height);
        if (!texture_) { std::cerr << "[WARN] SDL_CreateTexture failed: " << SDL_GetError() << std::endl; return false; }
        return true;
    }

    void show(const cv::Mat& frame) override {
        if (!texture_) return;
        SDL_UpdateTexture(texture_, nullptr, frame.data, frame.step);
        SDL_RenderClear(renderer_);
        SDL_RenderCopy(renderer_, texture_, nullptr, nullptr);
        SDL_RenderPresent(renderer_);
        // Pump events to keep window responsive
        SDL_Event e; while (SDL_PollEvent(&e)) { if (e.type == SDL_QUIT) {} }
    }

    void close() override {
        if (texture_) { SDL_DestroyTexture(texture_); texture_ = nullptr; }
        if (renderer_) { SDL_DestroyRenderer(renderer_); renderer_ = nullptr; }
        if (window_) { SDL_DestroyWindow(window_); window_ = nullptr; }
        if (SDL_WasInit(SDL_INIT_VIDEO)) SDL_QuitSubSystem(SDL_INIT_VIDEO);
        if (SDL_WasInit(0)) SDL_Quit();
    }

private:
    SDL_Window* window_;
    SDL_Renderer* renderer_;
    SDL_Texture* texture_;
    int w_, h_;
};

std::unique_ptr<IDisplay> createDisplay(const std::string& mode) {
    if (mode == "off") {
        return std::unique_ptr<IDisplay>();
    }
    auto d = std::make_unique<SdlDisplay>();
    return d;
}

} // namespace yolov5
#endif // HAVE_SDL2

