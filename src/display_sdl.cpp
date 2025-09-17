#ifdef HAVE_SDL2
#include "display.hpp"
#include <SDL.h>
#include <unistd.h>

#include <array>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

/**
 * @file display_sdl.cpp
 * @brief SDL2-backed display implementation with HUD rendering and watchdog support.
 */

namespace yolov5 {
namespace {

using Glyph = std::array<uint8_t, 8>;

Glyph makeGlyph(std::initializer_list<const char*> rows) {
    Glyph g{};
    size_t r = 0;
    for (const char* row : rows) {
        if (r >= g.size()) break;
        uint8_t bits = 0;
        for (int c = 0; row && row[c] != '\0' && c < 8; ++c) {
            if (row[c] != ' ') {
                bits |= static_cast<uint8_t>(1u << c);
            }
        }
        g[r++] = bits;
    }
    return g;
}

const Glyph GLYPH_SPACE = makeGlyph({
    "        ",
    "        ",
    "        ",
    "        ",
    "        ",
    "        ",
    "        ",
    "        "
});

const Glyph GLYPH_DOT = makeGlyph({
    "        ",
    "        ",
    "        ",
    "        ",
    "        ",
    "  XX   ",
    "  XX   ",
    "        "
});

const Glyph GLYPH_ZERO = makeGlyph({
    "  XXXX ",
    " XX  XX",
    " XX  XX",
    " XX  XX",
    " XX  XX",
    " XX  XX",
    "  XXXX ",
    "        "
});

const Glyph GLYPH_ONE = makeGlyph({
    "   XX  ",
    "  XXX  ",
    "   XX  ",
    "   XX  ",
    "   XX  ",
    "   XX  ",
    " XXXXXX",
    "        "
});

const Glyph GLYPH_TWO = makeGlyph({
    "  XXXX ",
    " XX  XX",
    "     XX",
    "   XXX ",
    "  XX   ",
    " XX    ",
    " XXXXXX",
    "        "
});

const Glyph GLYPH_THREE = makeGlyph({
    " XXXXXX",
    "     XX",
    "    XX ",
    "  XXXX ",
    "     XX",
    " XX  XX",
    "  XXXX ",
    "        "
});

const Glyph GLYPH_FOUR = makeGlyph({
    "   XXX ",
    "  XXXX ",
    " XX XX ",
    " XX XX ",
    " XXXXXX",
    "    XX ",
    "    XX ",
    "        "
});

const Glyph GLYPH_FIVE = makeGlyph({
    " XXXXXX",
    " XX    ",
    " XXXXX ",
    "     XX",
    "     XX",
    " XX  XX",
    "  XXXX ",
    "        "
});

const Glyph GLYPH_SIX = makeGlyph({
    "  XXXX ",
    " XX    ",
    " XX    ",
    " XXXXX ",
    " XX  XX",
    " XX  XX",
    "  XXXX ",
    "        "
});

const Glyph GLYPH_SEVEN = makeGlyph({
    " XXXXXX",
    "     XX",
    "    XX ",
    "   XX  ",
    "  XX   ",
    "  XX   ",
    "  XX   ",
    "        "
});

const Glyph GLYPH_EIGHT = makeGlyph({
    "  XXXX ",
    " XX  XX",
    " XX  XX",
    "  XXXX ",
    " XX  XX",
    " XX  XX",
    "  XXXX ",
    "        "
});

const Glyph GLYPH_NINE = makeGlyph({
    "  XXXX ",
    " XX  XX",
    " XX  XX",
    "  XXXXX",
    "     XX",
    "     XX",
    "  XXXX ",
    "        "
});

const Glyph GLYPH_A = makeGlyph({
    "   XX  ",
    "  XXXX ",
    " XX  XX",
    " XX  XX",
    " XXXXXX",
    " XX  XX",
    " XX  XX",
    "        "
});

const Glyph GLYPH_B = makeGlyph({
    " XXXXX ",
    " XX  XX",
    " XX  XX",
    " XXXXX ",
    " XX  XX",
    " XX  XX",
    " XXXXX ",
    "        "
});

const Glyph GLYPH_C = makeGlyph({
    "  XXXX ",
    " XX  XX",
    " XX    ",
    " XX    ",
    " XX    ",
    " XX  XX",
    "  XXXX ",
    "        "
});

const Glyph GLYPH_D = makeGlyph({
    " XXXXX ",
    " XX  XX",
    " XX  XX",
    " XX  XX",
    " XX  XX",
    " XX  XX",
    " XXXXX ",
    "        "
});

const Glyph GLYPH_E = makeGlyph({
    " XXXXXX",
    " XX    ",
    " XX    ",
    " XXXXX ",
    " XX    ",
    " XX    ",
    " XXXXXX",
    "        "
});

const Glyph GLYPH_F = makeGlyph({
    " XXXXXX",
    " XX    ",
    " XX    ",
    " XXXXX ",
    " XX    ",
    " XX    ",
    " XX    ",
    "        "
});

const Glyph GLYPH_I = makeGlyph({
    " XXXXXX",
    "   XX  ",
    "   XX  ",
    "   XX  ",
    "   XX  ",
    "   XX  ",
    " XXXXXX",
    "        "
});

const Glyph GLYPH_L = makeGlyph({
    " XX    ",
    " XX    ",
    " XX    ",
    " XX    ",
    " XX    ",
    " XX    ",
    " XXXXXX",
    "        "
});

const Glyph GLYPH_N = makeGlyph({
    " XX  XX",
    " XXX XX",
    " XXX XX",
    " XX XXXX",
    " XX XXXX",
    " XX  XX",
    " XX  XX",
    "        "
});

const Glyph GLYPH_O = makeGlyph({
    "  XXXX ",
    " XX  XX",
    " XX  XX",
    " XX  XX",
    " XX  XX",
    " XX  XX",
    "  XXXX ",
    "        "
});

const Glyph GLYPH_P = makeGlyph({
    " XXXXX ",
    " XX  XX",
    " XX  XX",
    " XXXXX ",
    " XX    ",
    " XX    ",
    " XX    ",
    "        "
});

const Glyph GLYPH_Q = makeGlyph({
    "  XXXX ",
    " XX  XX",
    " XX  XX",
    " XX  XX",
    " XX XX ",
    "  XXXX ",
    "     XX",
    "        "
});

const Glyph GLYPH_R = makeGlyph({
    " XXXXX ",
    " XX  XX",
    " XX  XX",
    " XXXXX ",
    " XX XX ",
    " XX  XX",
    " XX  XX",
    "        "
});

const Glyph GLYPH_S = makeGlyph({
    "  XXXX ",
    " XX  XX",
    " XX    ",
    "  XXX  ",
    "     XX",
    " XX  XX",
    "  XXXX ",
    "        "
});

const Glyph GLYPH_T = makeGlyph({
    " XXXXXX",
    "   XX  ",
    "   XX  ",
    "   XX  ",
    "   XX  ",
    "   XX  ",
    "   XX  ",
    "        "
});

const Glyph GLYPH_U = makeGlyph({
    " XX  XX",
    " XX  XX",
    " XX  XX",
    " XX  XX",
    " XX  XX",
    " XX  XX",
    "  XXXX ",
    "        "
});

const Glyph GLYPH_V = makeGlyph({
    " XX  XX",
    " XX  XX",
    " XX  XX",
    " XX  XX",
    "  XXXX ",
    "  XXXX ",
    "   XX  ",
    "        "
});

const Glyph GLYPH_Y = makeGlyph({
    " XX  XX",
    " XX  XX",
    "  XXXX ",
    "   XX  ",
    "   XX  ",
    "   XX  ",
    "   XX  ",
    "        "
});

const Glyph& glyphFor(char c) {
    switch (c) {
        case '0': return GLYPH_ZERO;
        case '1': return GLYPH_ONE;
        case '2': return GLYPH_TWO;
        case '3': return GLYPH_THREE;
        case '4': return GLYPH_FOUR;
        case '5': return GLYPH_FIVE;
        case '6': return GLYPH_SIX;
        case '7': return GLYPH_SEVEN;
        case '8': return GLYPH_EIGHT;
        case '9': return GLYPH_NINE;
        case 'A': return GLYPH_A;
        case 'B': return GLYPH_B;
        case 'C': return GLYPH_C;
        case 'D': return GLYPH_D;
        case 'E': return GLYPH_E;
        case 'F': return GLYPH_F;
        case 'I': return GLYPH_I;
        case 'L': return GLYPH_L;
        case 'N': return GLYPH_N;
        case 'O': return GLYPH_O;
        case 'P': return GLYPH_P;
        case 'Q': return GLYPH_Q;
        case 'R': return GLYPH_R;
        case 'S': return GLYPH_S;
        case 'T': return GLYPH_T;
        case 'U': return GLYPH_U;
        case 'V': return GLYPH_V;
        case 'Y': return GLYPH_Y;
        case '.': return GLYPH_DOT;
        case ' ': return GLYPH_SPACE;
        default:  return GLYPH_SPACE;
    }
}

std::string chooseDriver(const std::string& hint) {
    if (hint != "auto" && !hint.empty()) {
        return hint;
    }
    if (const char* forced = std::getenv("SDL_VIDEODRIVER")) {
        if (forced[0] != '\0') {
            return forced;
        }
    }
    if (std::getenv("WAYLAND_DISPLAY")) {
        return "wayland";
    }
    if (::access("/dev/dri/card0", R_OK | W_OK) == 0) {
        return "kmsdrm";
    }
    if (std::getenv("DISPLAY")) {
        return "x11";
    }
    return "dummy";
}

} // namespace

/**
 * @brief SDL2-backed display with manual text overlay and watchdog hooks.
 *
 * Lives on output/display thread. Probes driver, renders HUD glyphs using
 * manual bitmap fonts, and exports probe snapshots when requested.
 */
class DisplaySDL : public IDisplay {
public:
    explicit DisplaySDL(std::string driver_hint)
        : driver_hint_(std::move(driver_hint)) {}

    ~DisplaySDL() override { close(); }

    /**
     * @brief Initialize SDL window/renderer and choose best driver.
     *
     * Applies CLI driver hint, falls back gracefully, and logs selected backend.
     */
    bool init(const DisplayConfig& cfg) override {
        width_ = cfg.width;
        height_ = cfg.height;
        const std::string requested = driver_hint_.empty() ? std::string("auto") : driver_hint_;
        std::string desired = (requested == "auto") ? chooseDriver("auto") : requested;
        const std::string fallback = chooseDriver("auto");
        bool fallback_notified = false;

        SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "nearest");
        SDL_SetHint(SDL_HINT_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR, "0");
        SDL_setenv("SDL_VIDEO_ALLOW_SCREENSAVER", "1", 1);

        auto apply_driver = [&](const std::string& driver) -> bool {
            SDL_setenv("SDL_VIDEODRIVER", driver.c_str(), 1);
            if (SDL_WasInit(SDL_INIT_VIDEO) == 0) {
                if (SDL_InitSubSystem(SDL_INIT_VIDEO) != 0) {
                    return false;
                }
                owns_sdl_ = true;
            }
            return true;
        };

        if (!apply_driver(desired)) {
            if (requested != "auto") {
                fallback_notified = true;
                std::cerr << "[display] requested driver '" << requested
                          << "' not available, using '" << fallback << "'" << std::endl;
                if (!apply_driver(fallback)) {
                    std::cerr << "[display] SDL_InitSubSystem failed: " << SDL_GetError() << std::endl;
                    return false;
                }
                desired = fallback;
            } else {
                std::cerr << "[display] SDL_InitSubSystem failed: " << SDL_GetError() << std::endl;
                return false;
            }
        }

        driver_selected_ = desired;

        window_ = SDL_CreateWindow(cfg.title.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                   width_, height_, SDL_WINDOW_SHOWN);
        if (!window_) {
            std::cerr << "[display] SDL_CreateWindow failed: " << SDL_GetError() << std::endl;
            close();
            return false;
        }
        renderer_ = SDL_CreateRenderer(window_, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
        if (!renderer_) {
            std::cerr << "[display] SDL_CreateRenderer failed: " << SDL_GetError() << std::endl;
            close();
            return false;
        }
        vsync_enabled_ = true;
#if SDL_VERSION_ATLEAST(2,0,18)
        if (SDL_RenderSetVSync(renderer_, 0) == 0) {
            vsync_enabled_ = false;
        }
#else
        vsync_enabled_ = false;
#endif
        texture_ = SDL_CreateTexture(renderer_, SDL_PIXELFORMAT_BGR24, SDL_TEXTUREACCESS_STREAMING,
                                     width_, height_);
        if (!texture_) {
            std::cerr << "[display] SDL_CreateTexture failed: " << SDL_GetError() << std::endl;
            close();
            return false;
        }
        SDL_SetRenderDrawBlendMode(renderer_, SDL_BLENDMODE_BLEND);
        int actual_w = 0;
        int actual_h = 0;
        SDL_GetWindowSize(window_, &actual_w, &actual_h);
        if (actual_w > 0 && actual_h > 0) {
            width_ = actual_w;
            height_ = actual_h;
        }
        const char* drv = SDL_GetCurrentVideoDriver();
        if (drv) {
            if (!fallback_notified && requested != "auto" && driver_selected_ != drv) {
                std::cerr << "[display] requested driver '" << requested
                          << "' not available, using '" << drv << "'" << std::endl;
                fallback_notified = true;
            }
            driver_selected_ = drv;
        }
        last_present_ = std::chrono::steady_clock::now();
        std::cout << "[display] SDL driver=" << driver_selected_ << " size="
                  << width_ << "x" << height_ << " (window created)" << std::endl;
        std::cout << "[display] vsync=" << (vsync_enabled_ ? "on" : "off") << std::endl;
        return true;
    }

    /**
     * @brief Present annotated frame to SDL texture and run watchdog bookkeeping.
     */
    bool present(const DisplayFrameInfo& frame) override {
        if (!renderer_ || !texture_) return true;
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            handleEvent(ev);
        }
        if (frame.metrics_valid && frame.metrics) {
            updateMetrics(*frame.metrics);
        }
        if (pending_quit_) return false;
        if (SDL_UpdateTexture(texture_, nullptr, frame.image.data, frame.image.step) != 0) {
            static bool warned = false;
            if (!warned) {
                std::cerr << "[display] SDL_UpdateTexture failed: " << SDL_GetError() << std::endl;
                warned = true;
            }
        }
        SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 255);
        SDL_RenderClear(renderer_);
        SDL_RenderCopy(renderer_, texture_, nullptr, nullptr);
        if (osd_enabled_) {
            renderOSD();
        }
        SDL_RenderPresent(renderer_);
        last_present_ = std::chrono::steady_clock::now();
        return !pending_quit_;
    }

    /**
     * @brief Store latest metrics snapshot for HUD rendering on present().
     */
    void updateMetrics(const PerfMetrics& metrics) override {
        std::lock_guard<std::mutex> lk(metrics_mutex_);
        metrics_ = metrics;
        has_metrics_ = true;
    }

    std::chrono::steady_clock::time_point lastPresentMono() const override {
        return last_present_;
    }

    std::string driverName() const override { return driver_selected_; }

    void close() override {
        if (texture_) {
            SDL_DestroyTexture(texture_);
            texture_ = nullptr;
        }
        if (renderer_) {
            SDL_DestroyRenderer(renderer_);
            renderer_ = nullptr;
        }
        if (window_) {
            SDL_DestroyWindow(window_);
            window_ = nullptr;
        }
        if (owns_sdl_) {
            SDL_QuitSubSystem(SDL_INIT_VIDEO);
            owns_sdl_ = false;
        }
    }

private:
    void handleEvent(const SDL_Event& e) {
        if (e.type == SDL_QUIT) {
            pending_quit_ = true;
            return;
        }
        if (e.type == SDL_KEYDOWN) {
            const SDL_Keycode key = e.key.keysym.sym;
            if (key == SDLK_ESCAPE || key == SDLK_q) {
                pending_quit_ = true;
            } else if (key == SDLK_o) {
                osd_enabled_ = !osd_enabled_;
                std::cout << "[display] osd " << (osd_enabled_ ? "on" : "off") << std::endl;
            } else if (key == SDLK_v) {
                toggleVsync();
            }
        }
    }

    void toggleVsync() {
#if SDL_VERSION_ATLEAST(2,0,18)
        bool request = !vsync_enabled_;
        if (SDL_RenderSetVSync(renderer_, request ? 1 : 0) == 0) {
            vsync_enabled_ = request;
            std::cout << "[display] vsync=" << (vsync_enabled_ ? "on" : "off") << std::endl;
        } else {
            std::cout << "[display] vsync toggle unsupported: " << SDL_GetError() << std::endl;
        }
#else
        vsync_enabled_ = !vsync_enabled_;
        std::cout << "[display] vsync toggle requested (not supported on this SDL build)" << std::endl;
#endif
    }

    void renderOSD() {
        PerfMetrics metrics_copy{};
        bool ready = false;
        {
            std::lock_guard<std::mutex> lk(metrics_mutex_);
            if (has_metrics_) {
                metrics_copy = metrics_;
                ready = true;
            }
        }
        if (!ready) return;

        auto qv = [&](const char* key) -> int {
            auto it = metrics_copy.queue_sizes.find(key);
            return (it == metrics_copy.queue_sizes.end()) ? 0 : it->second;
        };

        char line1[64];
        char line2[64];
        char line3[64];
        char line4[64];
        char line5[64];
        std::snprintf(line1, sizeof(line1), "IN %4.1f OUT %4.1f DROP %4.1f",
                      metrics_copy.input_fps, metrics_copy.output_fps, metrics_copy.drop_percentage);
        std::snprintf(line2, sizeof(line2), "CAP %4.1f PP %4.1f IN50 %4.1f",
                      metrics_copy.latency_ms.capture,
                      metrics_copy.latency_ms.preprocess,
                      metrics_copy.latency_ms.inference_p50);
        std::snprintf(line3, sizeof(line3), "IN95 %4.1f POST %4.1f OVL %4.1f",
                      metrics_copy.latency_ms.inference_p95,
                      metrics_copy.latency_ms.postprocess,
                      metrics_copy.latency_ms.overlay);
        std::snprintf(line4, sizeof(line4), "ENC %4.1f DISP %4.1f",
                      metrics_copy.latency_ms.encode,
                      metrics_copy.latency_ms.display);
        std::snprintf(line5, sizeof(line5), "Q %d %d %d %d %d",
                      qv("cap_pp"), qv("pp_sched"), qv("sched_inf"), qv("inf_post"), qv("post_reord"));

        std::vector<std::string> lines = {
            line1, line2, line3, line4, line5
        };
        int max_chars = 0;
        for (auto& line : lines) {
            std::transform(line.begin(), line.end(), line.begin(), [](unsigned char c){ return static_cast<char>(std::toupper(c)); });
            max_chars = std::max<int>(max_chars, static_cast<int>(line.size()));
        }
        const int padding = 4;
        SDL_Rect bg{padding, padding, max_chars * 8 + padding, static_cast<int>(lines.size()) * 8 + padding};
        SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 160);
        SDL_RenderFillRect(renderer_, &bg);
        SDL_SetRenderDrawColor(renderer_, 255, 255, 255, 255);
        int y = bg.y + 2;
        for (const auto& line : lines) {
            drawString(bg.x + 2, y, line);
            y += 8;
        }
    }

    void drawString(int x, int y, const std::string& text) {
        int cx = x;
        for (char ch : text) {
            drawChar(cx, y, ch);
            cx += 8;
        }
    }

    void drawChar(int x, int y, char ch) {
        const Glyph& g = glyphFor(ch);
        for (int row = 0; row < 8; ++row) {
            uint8_t bits = g[row];
            for (int col = 0; col < 8; ++col) {
                if (bits & (1u << col)) {
                    SDL_RenderDrawPoint(renderer_, x + col, y + row);
                }
            }
        }
    }

    std::string driver_hint_;
    std::string driver_selected_{};
    SDL_Window* window_{nullptr};
    SDL_Renderer* renderer_{nullptr};
    SDL_Texture* texture_{nullptr};
    int width_{0};
    int height_{0};
    bool osd_enabled_{true};
    bool pending_quit_{false};
    bool vsync_enabled_{false};
    bool owns_sdl_{false};
    mutable std::mutex metrics_mutex_;
    PerfMetrics metrics_{};
    bool has_metrics_{false};
    std::chrono::steady_clock::time_point last_present_ = std::chrono::steady_clock::now();
};

std::unique_ptr<IDisplay> createDisplay(const std::string& mode, const std::string& driver_hint) {
    if (mode != "sdl") {
        return nullptr;
    }
    return std::make_unique<DisplaySDL>(driver_hint);
}

} // namespace yolov5

#endif // HAVE_SDL2
