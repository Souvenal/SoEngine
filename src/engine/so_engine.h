#pragma once

#include "platform/window.h"
#include "platform/input_events.h"
#include "app/application.h"
#include "utils/timer.h"

#include <memory>
#include <filesystem>

enum class ExitCode {
    Success,
    Close,
    FatalError
};

class SoEngine {
public:
    SoEngine(const std::filesystem::path& appDir);
    ~SoEngine();

    /**
     * @brief Initializes the application
     */
    void initialize();

    /**
     * @brief Runs the main update and render loop
     */
    void mainLoop();

    /**
     * @brief Handles a single frame of the main loop
     */
    void mainLoopFrame();

    void update();

    /**
     * @brief Terminates the application
     * @param code Determines how the application should exit
     */
    void terminate(ExitCode code);

    /**
     * @brief Requests to close the application at the next available point
     */
    void close();

    /**
     * @brief Resizes the application window, triggered by resize window callback
     */
    void resizeWindow(uint32_t width, uint32_t height);

    /**
     * @brief Resizes the framebuffer, triggered by framebuffer resize callback
     */
    void resizeFramebuffer();

    /**
     * @brief Handles the input event triggered by the window
     * @param event The input event to handle
     */
    void inputEvent(const InputEvent& event);

    Window& getWindow();

    Application& getApp();

private:
    Window::Properties windowProperties;
    bool                fixedSimulationFPS { false };
    float               simulationFrameTime { 1.0f / 60.0f };

    std::filesystem::path appDir;

    Timer timer;
    std::unique_ptr<Window> window;
    std::unique_ptr<Application> app;
    ExitCode exitCode {ExitCode::Success};
};