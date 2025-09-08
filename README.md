# So Engine
This is a learning project, for grasping some basic concepts in Vulkan APIs and building a game engine.

## Dependencies
- spdlog        (included in vcpkg)
- glfw3         (included in vcpkg)
- glm           (included in vcpkg)
- ktxvulkan     (included in vcpkg)
- tinygltf      (included in vcpkg)
- Vulkan SDK    (note: need to be installed locally, so that tools like vkconfig can be utilized)

![](./examples/viking_room.gif)

## Configuration

Because this project uses vcpkg for dependency management, the user need to specify the environment variable `VCPKG_ROOT` to integrate vcpkg with cmake, and add a `CMakeUserPresets.json` locally, to specify the user's compiler and generator. It may look like this:
```
{
  "version": 2,
  "configurePresets": [
    {
      "name": "vcpkg-local-debug",
      "inherits": "vcpkg-debug",
      "generator": "Ninja",
      "cacheVariables": {
        "CMAKE_CXX_COMPILER": "/usr/local/bin/clang++"
      }
    },
    {
      "name": "vcpkg-local-release",
      "inherits": "vcpkg-release",
      "generator": "Ninja",
      "cacheVariables": {
        "CMAKE_CXX_COMPILER": "/usr/local/bin/clang++"
      }
    }
  ]
}
```

## Architecture Overview

This project adopts an object-oriented, modular architecture. The core idea is to implement different application features by inheriting from a common `Application` base class:

- **`Application`**: Implement some basic logic like Vulkan instance, physical device, logical device and swapchain initialization
- **`ComputeApp`**: Inherits from `Application` and implements compute-focused tasks (such as particle simulation) using Vulkan compute pipelines.
- **`ModelApp`**: Inherits from `Application` and implements model loading, rendering, and texture management.

Each derived class can have its own configuration struct (such as `ComputeAppInfo`, `ModelAppInfo`). This design makes it easy to extend the engine with new application types—simply inherit from `Application` and implement the required interfaces.

I have just decoupled window management procedure from the rendering engine (basically following the same architecture in [Vulkan-Samples](https://github.com/KhronosGroup/Vulkan-Samples)). Maybe the next improve is to construct a user interface using Dear ImGui?

## Functions Implemented

### Compute Application
The program runs a simulation of up to millions of paricle collision simulation(works on my laptop).

### Model Application
The program can load a gltf model from `models` directory, while loading ktx2 texures. However, it supports only one texure file for mapping.

## Issues
Because I'm working on macos, there might be some compatibility issues building on other platforms.

## To do
- Support cubemaps and texture arrays
- Deffered shading
- Ray tracing