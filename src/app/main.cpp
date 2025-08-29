#include "application.h"

int main(int argc, char* argv[]) {
    const std::filesystem::path binPath(argv[0]);
    // Location: <project_root>/bin
    std::filesystem::path appDir = binPath.parent_path().parent_path();

    Application app(appDir);

    app.initialize();
    ExitCode code = app.mainLoop();

    // SoEngine engine(appDir);

    // try {
    //     engine.run();
    // } catch (vk::SystemError& err) {
    //     std::println(stderr, "Vulkan error: {}", err.what());
    // } catch (std::exception& err) {
    //     std::println(stderr, "{}", err.what());
    //     return EXIT_FAILURE;
    // }

    return EXIT_SUCCESS;
}