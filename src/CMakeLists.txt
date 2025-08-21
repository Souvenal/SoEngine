file(GLOB_RECURSE ALL_INCLUDE CONFIGURE_DEPENDS "*.h")
file(GLOB_RECURSE ALL_SOURCES CONFIGURE_DEPENDS "*.cpp")

target_sources(${TARGET} PUBLIC ${ALL_INCLUDE} ${ALL_SOURCES})
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/ext)
target_link_libraries(${TARGET} PRIVATE stb)
target_link_libraries(${TARGET} PRIVATE tinyobjloader)

target_compile_features(${TARGET} PRIVATE cxx_std_23)

find_package(glfw3 REQUIRED)
if (glfw3_FOUND)
#     target_include_directories(${TARGET} PRIVATE ${GLFW_INCLUDE_DIRS})
    target_link_libraries(${TARGET} PRIVATE glfw)
else()
    message(FATAL_ERROR "GLFW not found")
endif()

find_package(glm REQUIRED)
if (glm_FOUND)
    # target_include_directories(${TARGET} PRIVATE ${GLM_INCLUDE_DIRS})
    target_link_libraries(${TARGET} PRIVATE glm::glm)
else()
    message(FATAL_ERROR "GLM not found")
endif()

find_package (Vulkan REQUIRED)
# set up Vulkan C++ module
add_library(VulkanCppModule)
add_library(Vulkan::cppm ALIAS VulkanCppModule)

target_compile_definitions(VulkanCppModule PUBLIC
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
)
target_include_directories(VulkanCppModule
    PUBLIC
    "${Vulkan_INCLUDE_DIR}"
)
target_include_directories(${TARGET} PRIVATE
    Vulkan::Headers
)
target_link_libraries(VulkanCppModule
    PUBLIC
    Vulkan::Vulkan
)

set_target_properties(VulkanCppModule PROPERTIES CXX_STANDARD 23)

target_sources(VulkanCppModule
    PUBLIC
    FILE_SET cxx_modules TYPE CXX_MODULES
    BASE_DIRS
    "${Vulkan_INCLUDE_DIR}"
    FILES
    "${Vulkan_INCLUDE_DIR}/vulkan/vulkan.cppm"
)

target_link_libraries(${TARGET} PUBLIC Vulkan::cppm)