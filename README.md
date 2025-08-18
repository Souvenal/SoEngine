# LearnVulkan
This is a learning project, for grasping some basic concepts in Vulkan APIs.

## Dependencies
- glfw3
- glm
- Vulkan SDK

## Functions Implemented
The program can load a wavefront obj model from `models` directory, while loading texures. However, it supports only one texure file for mapping.

## Issues
Because I'm working on macos, there might be some compatibility issues building on Windows/Linux.

Also, the main file is written using old vulkan APIs, rather than modern cpp module and vulkan namespace. That's because I had some difficulty working on vscode intellisense plugin, for it's hard to support modern cpp modules. (However it works fine with Clion.)