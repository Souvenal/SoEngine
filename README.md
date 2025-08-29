# So Engine
This is a learning project, for grasping some basic concepts in Vulkan APIs and building a game engine.

## Dependencies
- glfw3         (included in vcpkg)
- glm           (included in vcpkg)
- ktxvulkan     (included in vcpkg)
- tinygltf      (included in vcpkg)
- Vulkan SDK    (note: need to be installed locally, so that tools like vkconfig can be utilized)

![](./examples/viking_room.gif)

## Usage

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

## Functions Implemented
The program can load a gltf model from `models` directory, while loading ktx2 texures. However, it supports only one texure file for mapping.

I have just decoupled window management procedure from the rendering engine (basically following the same architecture in [Vulkan-Samples](https://github.com/KhronosGroup/Vulkan-Samples)). Maybe the next improve is to construct a user interface using Dear ImGui?

## Issues
Because I'm working on macos, there might be some compatibility issues building on other platforms.

## To do
- Support cubemaps and texture arrays
- Deffered shading
- Ray tracing