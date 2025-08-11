# LearnVulkan
This is a learning project, for grasping some basic concepts in Vulkan APIs.

## Dependencies
- glfw3
- glm
- Vulkan SDK

## Functions Implemented
The program successfully built should be able to render a rectangle, using index buffer and vertex buffer, with indices data hardcoded in the cpp file.

After implementing uniform buffer object, it can now render a rotating 3D square.

## Issues
Because I'm working on macos, there might be some compatibility issues building on Windows/Linux.

Also, the main file is written using old vulkan APIs, rather than modern cpp module and vulkan namespace. That's because I had some difficulty working on vscode intellisense plugin, for it's hard to support modern cpp modules. (However it works fine with Clion.)