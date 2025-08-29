#include "application.h"

Application::Application(const std::filesystem::path& appDir)
    : appDir(appDir)
{

}

Application::~Application() {

}

ExitCode Application::initialize() {
    window = std::make_unique<Window>(this, Window::Properties{
        .title = "LearnVulkan",
        .mode = Window::Mode::Default,
        .resizable = true,
        .extent = {800, 600}
    });
    if (!window) {
        std::println(stderr, "Failed to create window");
        return ExitCode::FatalError;
    }
    
    engine = std::make_unique<SoEngine>(appDir);
    if (!engine) {
        std::println(stderr, "Failed to create engine");
        return ExitCode::FatalError;
    }
    engine->prepare(window.get());

    return ExitCode::Success;
}

ExitCode Application::mainLoop() {
    ExitCode code = ExitCode::Success;
    while (code == ExitCode::Success) {
        code = mainLoopFrame();
    }
    return code;
}

ExitCode Application::mainLoopFrame() {
    try {
        update();

        if (engine->shouldTerminate()) {
            terminate(ExitCode::Close);
            return ExitCode::Close;
        }

        window->processEvents();
        if (window->shouldClose()) {
            terminate(ExitCode::Close);
            return ExitCode::Close;
        }
    } catch (const std::exception& err) {
        std::println(stderr, "Fatal error: {}", err.what());
        return ExitCode::FatalError;
    }

    return ExitCode::Success;
}

void Application::update() {
    auto deltaTime = timer.tick<Timer::Seconds>();

    if (fixedSimulationFPS) {
        deltaTime = simulationFrameTime;
    }

    engine->update(deltaTime);
}

void Application::terminate(ExitCode code) {
    engine->terminate();
}

void Application::close() {

}

void Application::resizeWindow(uint32_t width, uint32_t height) {
    auto actualExtent = window->resize({width, height});
    windowProperties.extent = actualExtent;

    resizeFramebuffer();
}

void Application::resizeFramebuffer() {
    auto framebufferExtent = window->getFramebufferSize();
    // Wait for the framebuffer to be valid
    while (framebufferExtent.width == 0 || framebufferExtent.height == 0) {
        window->waitEvents();
        framebufferExtent = window->getFramebufferSize();
    }
    engine->recreateSwapChain();
}

void Application::inputEvent(const InputEvent& event) {
    engine->inputEvent(event);
}