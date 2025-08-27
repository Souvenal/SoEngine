file(GLOB_RECURSE ALL_INCLUDE CONFIGURE_DEPENDS "*.h" "*.hpp")
file(GLOB_RECURSE ALL_SOURCES CONFIGURE_DEPENDS "*.cpp")

target_sources(${TARGET} PUBLIC ${ALL_INCLUDE} ${ALL_SOURCES})

target_compile_features(${TARGET} PRIVATE cxx_std_23)
# target_compile_options(${TARGET} PRIVATE
#     -Wall
#     -fsanitize=address
# )
# target_link_options(${TARGET} PRIVATE
#     -fsanitize=address
# )

find_package(glfw3 CONFIG REQUIRED)
target_link_libraries(${TARGET} PRIVATE glfw)

find_package(glm CONFIG REQUIRED)
target_link_libraries(${TARGET} PRIVATE glm::glm)

find_package(Stb REQUIRED)
target_include_directories(${TARGET} PRIVATE ${Stb_INCLUDE_DIR})

find_path(TINYGLTF_INCLUDE_DIRS "tiny_gltf.h")
target_include_directories(${TARGET} PRIVATE ${TINYGLTF_INCLUDE_DIRS})

find_package(Ktx CONFIG REQUIRED)
target_link_libraries(${TARGET} PRIVATE KTX::ktx)

find_package (Vulkan REQUIRED)
# target_link_libraries(${TARGET} PRIVATE Vulkan::Vulkan)
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