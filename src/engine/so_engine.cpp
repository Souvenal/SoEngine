#include "so_engine.h"

#include "utils/logging.h"
#include "app/model_app.h"
#include "app/compute_app.h"

SoEngine::SoEngine(const std::filesystem::path& appDir)
    : appDir(appDir) {
    InitOptions opts;
    opts.level = spdlog::level::debug;
    opts.enableFile = true;
    opts.logDirectory = appDir / "logs";
    logging::init(opts);
    LOG_INFO("Application initialized at {}", appDir.string());
    }

SoEngine::~SoEngine() {
    logging::shutdown();
}

void SoEngine::initialize() {
    window = std::make_unique<Window>(this, Window::Properties{
        .title = "LearnVulkan",
        .mode = Window::Mode::Default,
        .resizable = true,
        .extent = {800, 600}
    });
    if (!window) {
        LOG_ERROR("Failed to create window");
        exitCode = ExitCode::FatalError;
        return;
    }

    AppInfo appInfo{};
    // ComputeAppInfo computeAppInfo{ 800'000 };
    // app = std::make_unique<ComputeApp>(appDir, appInfo, computeAppInfo);
    ModelAppInfo modelAppInfo{};
    app = std::make_unique<ModelApp>(appDir, appInfo, modelAppInfo);
    if (!app) {
        LOG_ERROR("Failed to create application");
        exitCode = ExitCode::FatalError;
        return;
    }

    try {
        // app->prepare(window.get());
        app->onInit(window.get());
        exitCode = ExitCode::Success;
    } catch (const vk::SystemError& err) {
        LOG_CORE_ERROR("Vulkan error initializing application: {}", err.what());
        exitCode = ExitCode::FatalError;
    } catch (const std::exception& err) {
        LOG_CORE_ERROR("Standard error initializing application: {}", err.what());
        exitCode = ExitCode::FatalError;
    }
}

void SoEngine::mainLoop() {
    while (exitCode == ExitCode::Success) {
        mainLoopFrame();
    }

    terminate(exitCode);
}

void SoEngine::mainLoopFrame() {
    try {
        window->processEvents();
        if (window->shouldClose()) {
            close();
        }
        if (app->shouldTerminate()) {
            close();
            return;
        }

        update();

        render();
    } catch (const std::exception& err) {
        LOG_ERROR("Fatal error: {}", err.what());
        exitCode = ExitCode::FatalError;
    }
}

void SoEngine::update() {
    double deltaTime = timer.tick<Timer::Seconds>();

    if (fixedSimulationFPS) {
        deltaTime = simulationFrameTime;
    }

    app->onUpdate(deltaTime);
}

void SoEngine::render() {
    app->onRender();
}

void SoEngine::terminate(ExitCode code) {
    app->onShutdown();
    app.reset();
    window.reset();
}

void SoEngine::close() {
    exitCode = ExitCode::Close;
}

void SoEngine::resizeWindow(uint32_t width, uint32_t height) {
    auto actualExtent = window->resize({width, height});
    windowProperties.extent = actualExtent;

    resizeFramebuffer();
}

void SoEngine::resizeFramebuffer() {
    auto framebufferExtent = window->getFramebufferSize();
    // Wait for the framebuffer to be valid
    while (framebufferExtent.width == 0 || framebufferExtent.height == 0) {
        window->waitEvents();
        framebufferExtent = window->getFramebufferSize();
    }
    app->recreateSwapchain();
}

void SoEngine::inputEvent(const InputEvent& event) {
    app->onInputEvent(event);
}