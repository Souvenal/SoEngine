# LearnVulkan
This is a learning project, for grasping some basic concepts in Vulkan APIs.

## Dependencies
- glfw3
- glm
- Vulkan SDK
- stb-image
- tiny-obj-loader

![](./examples/viking_room.gif)

## Functions Implemented
The program can load a wavefront obj model from `models` directory, while loading texures. However, it supports only one texure file for mapping. It also enables MSAA for rendering.

The program uses cpp wrappers for Vulkan APIs. It uses dynamic rendering when supported, and uses traditional render approach when not, boosting backward compability.

## Issues
Because I'm working on macos, there might be some compatibility issues building on Windows/Linux.

When using traditonal render approach, if `MAX_FRAME_IN_FLIGHT` is set lower than `swapchainImgeCount`, there will be some validation layer warning, even though the program functions correctly.