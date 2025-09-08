#include "window.h"

#include "engine/so_engine.h"
#include "input_events.h"

#include <unordered_map>

namespace
{
void windowCloseCallback(GLFWwindow* window) {
    glfwSetWindowShouldClose(window, GLFW_TRUE);
}

void windowSizeCallback(GLFWwindow* window, int width, int height) {
    if (auto* engine = reinterpret_cast<SoEngine*>(glfwGetWindowUserPointer(window))) {
        engine->resizeWindow(static_cast<uint32_t>(width), static_cast<uint32_t>(height));
    }
}

void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    if (auto* app = reinterpret_cast<SoEngine*>(glfwGetWindowUserPointer(window))) {
        app->resizeFramebuffer();
    }
}

KeyCode translateKeyCode(int key) {
    static const std::unordered_map<int, KeyCode> keyLookup = {
        {GLFW_KEY_SPACE, KeyCode::Space},
        {GLFW_KEY_APOSTROPHE, KeyCode::Apostrophe},
        {GLFW_KEY_COMMA, KeyCode::Comma},
        {GLFW_KEY_MINUS, KeyCode::Minus},
        {GLFW_KEY_PERIOD, KeyCode::Period},
        {GLFW_KEY_SLASH, KeyCode::Slash},
        {GLFW_KEY_0, KeyCode::Num0},
        {GLFW_KEY_1, KeyCode::Num1},
        {GLFW_KEY_2, KeyCode::Num2},
        {GLFW_KEY_3, KeyCode::Num3},
        {GLFW_KEY_4, KeyCode::Num4},
        {GLFW_KEY_5, KeyCode::Num5},
        {GLFW_KEY_6, KeyCode::Num6},
        {GLFW_KEY_7, KeyCode::Num7},
        {GLFW_KEY_8, KeyCode::Num8},
        {GLFW_KEY_9, KeyCode::Num9},
        {GLFW_KEY_SEMICOLON, KeyCode::Semicolon},
        {GLFW_KEY_EQUAL, KeyCode::Equal},
        {GLFW_KEY_A, KeyCode::A},
        {GLFW_KEY_B, KeyCode::B},
        {GLFW_KEY_C, KeyCode::C},
        {GLFW_KEY_D, KeyCode::D},
        {GLFW_KEY_E, KeyCode::E},
        {GLFW_KEY_F, KeyCode::F},
        {GLFW_KEY_G, KeyCode::G},
        {GLFW_KEY_H, KeyCode::H},
        {GLFW_KEY_I, KeyCode::I},
        {GLFW_KEY_J, KeyCode::J},
        {GLFW_KEY_K, KeyCode::K},
        {GLFW_KEY_L, KeyCode::L},
        {GLFW_KEY_M, KeyCode::M},
        {GLFW_KEY_N, KeyCode::N},
        {GLFW_KEY_O, KeyCode::O},
        {GLFW_KEY_P, KeyCode::P},
        {GLFW_KEY_Q, KeyCode::Q},
        {GLFW_KEY_R, KeyCode::R},
        {GLFW_KEY_S, KeyCode::S},
        {GLFW_KEY_T, KeyCode::T},
        {GLFW_KEY_U, KeyCode::U},
        {GLFW_KEY_V, KeyCode::V},
        {GLFW_KEY_W, KeyCode::W},
        {GLFW_KEY_X, KeyCode::X},
        {GLFW_KEY_Y, KeyCode::Y},
        {GLFW_KEY_Z, KeyCode::Z},
        {GLFW_KEY_LEFT_BRACKET, KeyCode::LeftBracket},
        {GLFW_KEY_RIGHT_BRACKET, KeyCode::RightBracket},
        {GLFW_KEY_BACKSLASH, KeyCode::Backslash},
        {GLFW_KEY_GRAVE_ACCENT, KeyCode::GraveAccent},
        {GLFW_KEY_ESCAPE, KeyCode::Escape},
        {GLFW_KEY_ENTER, KeyCode::Enter},
        {GLFW_KEY_TAB, KeyCode::Tab},
        {GLFW_KEY_BACKSPACE, KeyCode::Backspace},
        {GLFW_KEY_INSERT, KeyCode::Insert},
        {GLFW_KEY_DELETE, KeyCode::Delete},
        {GLFW_KEY_RIGHT, KeyCode::Right},
        {GLFW_KEY_LEFT, KeyCode::Left},
        {GLFW_KEY_DOWN, KeyCode::Down},
        {GLFW_KEY_UP, KeyCode::Up},
        {GLFW_KEY_PAGE_UP, KeyCode::PageUp},
        {GLFW_KEY_PAGE_DOWN, KeyCode::PageDown},
        {GLFW_KEY_HOME, KeyCode::Home},
        {GLFW_KEY_END, KeyCode::End},
        {GLFW_KEY_CAPS_LOCK, KeyCode::CapsLock},
        {GLFW_KEY_SCROLL_LOCK, KeyCode::ScrollLock},
        {GLFW_KEY_NUM_LOCK, KeyCode::NumLock},
        {GLFW_KEY_PRINT_SCREEN, KeyCode::PrintScreen},
        {GLFW_KEY_PAUSE, KeyCode::Pause},
        {GLFW_KEY_F1, KeyCode::F1},
        {GLFW_KEY_F2, KeyCode::F2},
        {GLFW_KEY_F3, KeyCode::F3},
        {GLFW_KEY_F4, KeyCode::F4},
        {GLFW_KEY_F5, KeyCode::F5},
        {GLFW_KEY_F6, KeyCode::F6},
        {GLFW_KEY_F7, KeyCode::F7},
        {GLFW_KEY_F8, KeyCode::F8},
        {GLFW_KEY_F9, KeyCode::F9},
        {GLFW_KEY_F10, KeyCode::F10},
        {GLFW_KEY_F11, KeyCode::F11},
        {GLFW_KEY_F12, KeyCode::F12},
        {GLFW_KEY_LEFT_SHIFT, KeyCode::LeftShift},
        {GLFW_KEY_LEFT_CONTROL, KeyCode::LeftControl},
        {GLFW_KEY_LEFT_ALT, KeyCode::LeftAlt},
        {GLFW_KEY_LEFT_SUPER, KeyCode::LeftSuper},
        {GLFW_KEY_RIGHT_SHIFT, KeyCode::RightShift},
        {GLFW_KEY_RIGHT_CONTROL, KeyCode::RightControl},
        {GLFW_KEY_RIGHT_ALT, KeyCode::RightAlt},
        {GLFW_KEY_RIGHT_SUPER, KeyCode::RightSuper},
        {GLFW_KEY_MENU, KeyCode::Menu}
    };

    if (auto it = keyLookup.find(key); it != keyLookup.end()) {
        return it->second;
    }

    return KeyCode::Unknown;
}

KeyAction translateKeyAction(int action) {
    switch (action) {
    case GLFW_PRESS: return KeyAction::Down;
    case GLFW_RELEASE: return KeyAction::Up;
    case GLFW_REPEAT: return KeyAction::Repeat;
    default: return KeyAction::Unknown;
    }
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    KeyCode keyCode = translateKeyCode(key);
    KeyAction keyAction = translateKeyAction(action);

    if (auto* engine = reinterpret_cast<SoEngine*>(glfwGetWindowUserPointer(window))) {
        engine->inputEvent(KeyInputEvent(keyCode, keyAction));
    }
}

MouseButton translateMouseButton(int button) {
    if (button < GLFW_MOUSE_BUTTON_6) {
        return static_cast<MouseButton>(button);
    }
    return MouseButton::Unknown;
}

MouseAction translateMouseAction(int action) {
    if (action == GLFW_PRESS) {
        return MouseAction::Down;
    } else if (action == GLFW_RELEASE) {
        return MouseAction::Up;
    }
    return MouseAction::Unknown;
}

void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
    if (auto* engine = reinterpret_cast<SoEngine*>(glfwGetWindowUserPointer(window))) {
        engine->inputEvent(MouseButtonInputEvent{
            MouseButton::Unknown,
            MouseAction::Move,
            static_cast<float>(xpos),
            static_cast<float>(ypos)});
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (auto* engine = reinterpret_cast<SoEngine*>(glfwGetWindowUserPointer(window))) {
        double xPos, yPos;
        glfwGetCursorPos(window, &xPos, &yPos);

        engine->inputEvent(MouseButtonInputEvent{
            translateMouseButton(button),
            translateMouseAction(action),
            static_cast<float>(xPos),
            static_cast<float>(yPos)});
    }
}

}   // namespace

Window::Window(SoEngine* engine, const Properties& properties):
    properties(properties)
{
    // Initialize GLFW
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    switch (properties.mode)
    {
    default:
        handle = glfwCreateWindow(
            static_cast<int>(properties.extent.width),
            static_cast<int>(properties.extent.height),
            properties.title.c_str(),
            nullptr,
            nullptr
        );
    }

    if (!handle) {
        throw std::runtime_error("Failed to create GLFW window");
    }

    resize(properties.extent);

    glfwSetWindowUserPointer(handle, engine);

    glfwSetWindowCloseCallback(handle, windowCloseCallback);
    glfwSetWindowSizeCallback(handle, windowSizeCallback);
    glfwSetFramebufferSizeCallback(handle, framebufferSizeCallback);
    glfwSetKeyCallback(handle, keyCallback);
    glfwSetCursorPosCallback(handle, cursorPositionCallback);
    glfwSetMouseButtonCallback(handle, mouseButtonCallback);
}

Window::~Window() {
    glfwDestroyWindow(handle);
    glfwTerminate();
}

vk::raii::SurfaceKHR Window::createSurface(const vk::raii::Instance& instance) const {
    VkSurfaceKHR surface;
    if (glfwCreateWindowSurface(*instance, handle, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create window surface");
    }

    return vk::raii::SurfaceKHR(instance, surface);
}

bool Window::shouldClose() const {
    return glfwWindowShouldClose(handle);
}

void Window::processEvents() {
    glfwPollEvents();
}

void Window::waitEvents() {
    glfwWaitEvents();
}

void Window::close() const {
    glfwSetWindowShouldClose(handle, GLFW_TRUE);
}

Window::Extent Window::resize(const Extent& newExtent) {
    if (!properties.resizable) {
        return properties.extent;
    }
    properties.extent = Window::Extent{
        std::max(newExtent.width, MIN_WIDTH),
        std::max(newExtent.height, MIN_HEIGHT)
    };

    return properties.extent;
}

Window::Extent Window::getFramebufferSize() const {
    int width = 0, height = 0;
    glfwGetFramebufferSize(handle, &width, &height);
    return {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
}

Window::Extent Window::getWindowSize() const {
    return properties.extent;
}