set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# spdlog
set(SPDLOG_BUILD_EXAMPLE OFF)
set(SPDLOG_BUILD_TESTS OFF)
set(SPDLOG_INSTALL OFF)
set(SPDLOG_SANITIZE_ADDRESS ON)
set(SPDLOG_USE_STD_FORMAT ON)
set(SPDLOG_FMT_EXTERNAL OFF)
add_subdirectory(spdlog) # libspdlog
set_target_properties(spdlog PROPERTIES FOLDER ${THIRD_PARTY_DIR}/spdlog)

# glfw
set(GLFW_BUILD_DOCS OFF)
set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
set(GLFW_INSTALL OFF)
set(GLFW_LIBRARY_TYPE STATIC)
add_subdirectory(glfw) # libglfw
set_target_properties(glfw PROPERTIES FOLDER ${THIRD_PARTY_DIR}/glfw)

# glm
set(GLM_BUILD_INSTALL OFF)
set(GLM_BUILD_LIBRARY OFF)
set(GLM_BUILD_TESTS OFF)
set(GLM_ENABLE_CXX20 ON)
add_subdirectory(glm) # libglm

# tinygltf
set(TINYGLTF_BUILD_BUILDER_EXAMPLE OFF)
set(TINYGLTF_BUILD_GL_EXAMPLES OFF)
set(TINYGLTF_BUILD_LOADER_EXAMPLE OFF)
set(TINYGLTF_INSTALL OFF)
set(TINYGLTF_INSTALL_VENDOR OFF)
add_subdirectory(tinygltf) # libtinygltf

# fastgltf
set(FASTGLTF_COMPILE_AS_CPP20 ON)
set(FASTGLTF_USE_64BIT_FLOAT ON)
add_subdirectory(fastgltf) # libfastgltf

# ktx
set(KTX_FEATURE_KTX1        ON)    # Enable KTX1 support
set(KTX_FEATURE_KTX2        ON)    # Enable KTX2 support
set(KTX_FEATURE_VK_UPLOAD   ON)    # Enable Vulkan upload support
set(KTX_FEATURE_GL_UPLOAD   OFF)   # Disable OpenGL upload support
set(KTX_FEATURE_ETC_UNPACK  OFF)   # Disable ETC unpacking, only basic decoding needed
set(KTX_FEATURE_TOOLS       OFF)   # Do not build CLI tools
set(KTX_FEATURE_TESTS       OFF)   # Do not build tests
set(KTX_FEATURE_DOC         OFF)   # Do not build documentation
set(KTX_FEATURE_PY          OFF)   # Do not build Python bindings
set(KTX_FEATURE_JNI         OFF)   # Do not build Java bindings
set(KTX_FEATURE_LOADTEST_APPS OFF) # Do not build loadtest apps
set(KTX_FEATURE_TOOLS_CTS   OFF)   # Do not build CTS tools
set(KTX_WERROR              OFF)   # Do not treat warnings as errors
set(BUILD_SHARED_LIBS       OFF)   # Build static library (ON for shared library)
set(BUILD_TESTING           OFF)   # Do not build testing targets
add_subdirectory(ktx)

# ImGui
set(IMGUI_DIR ${THIRD_PARTY_DIR}/imgui)
file(GLOB IMGUI_SOURCES CONFIGURE_DEPENDS
    "${IMGUI_DIR}/*.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_glfw.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp")
file(GLOB IMGUI_HEADERS CONFIGURE_DEPENDS
    "${IMGUI_DIR}/*.h"
    "${IMGUI_DIR}/backends/imgui_impl_glfw.h"
    "${IMGUI_DIR}/backends/imgui_impl_vulkan.h")
add_library(imgui STATIC ${IMGUI_SOURCES} ${IMGUI_HEADERS})
target_include_directories(imgui PUBLIC
    $<BUILD_INTERFACE:${IMGUI_DIR}>
    $<BUILD_INTERFACE:${IMGUI_DIR}/backends>)
target_link_libraries(imgui PUBLIC glfw)
target_link_libraries(imgui PUBLIC Vulkan::Vulkan)