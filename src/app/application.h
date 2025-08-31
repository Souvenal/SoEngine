#pragma once

#include "window/window.h"
#include "window/input_events.h"
#include "core/so_engine.h"
#include "utils/timer.h"

#include <memory>
#include <filesystem>

enum class ExitCode {
    Success,
    Close,
    FatalError
};

class Application {
public:
    Application(const std::filesystem::path& appDir);
    ~Application();

    /**
     * @brief Initializes the application
     * @return An exitCode representing the outcome of the initializaition
     */
    ExitCode initialize();

    /**
     * @brief Runs the main update and render loop
     * @return An exitCode representing the outcome of the main loop
     */
    ExitCode mainLoop();

    /**
     * @brief Handles a single frame of the main loop
     * @return An exitCode representing the outcome of the frame
     */
    ExitCode mainLoopFrame();

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

    SoEngine& getEngine();

private:
    Window::Properties windowProperties;
    bool                fixedSimulationFPS { false };
    float               simulationFrameTime { 1.0f / 60.0f };

    std::filesystem::path appDir;

    Timer timer;
    std::unique_ptr<Window> window;
    std::unique_ptr<SoEngine> engine;
    bool should_close{false};
};