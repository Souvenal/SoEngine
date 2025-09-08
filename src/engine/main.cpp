#include "so_engine.h"

int main(int argc, char* argv[]) {
    const std::filesystem::path binPath(argv[0]);
    // Location: <project_root>/bin
    std::filesystem::path appDir = binPath.parent_path().parent_path();

    SoEngine engine(appDir);

    engine.initialize();
    engine.mainLoop();

    return EXIT_SUCCESS;
}