#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "common/vk_common.h"

#include <cstdint>
#include <string>

class Application;

class Window {
public:
    struct Extent {
        uint32_t    width;
        uint32_t    height;
    };
    static constexpr uint32_t MIN_WIDTH = 400;
    static constexpr uint32_t MIN_HEIGHT = 300;

    enum class Mode {
		Headless,
		Fullscreen,
		FullscreenBorderless,
		FullscreenStretch,
		Default
    };

    struct Properties {
        std::string title       {};
        Mode        mode        { Mode::Default };
        bool        resizable   { true };
        Extent      extent      { 800, 600 };
    };

    Window(Application* app, const Properties& properties);
    ~Window();

    vk::raii::SurfaceKHR createSurface(const vk::raii::Instance& instance) const;

    bool shouldClose() const;

    void processEvents();

    void waitEvents();

    void close() const;

    Extent resize(const Extent& extent);

    [[nodiscard]] Extent getFramebufferSize() const;
    [[nodiscard]] Extent getWindowSize() const;

private:
    Properties  properties  {};
    GLFWwindow* handle      { nullptr };
};