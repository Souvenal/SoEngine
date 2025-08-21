set(SLANGC_EXECUTABLE slangc)
set(SHADERS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SHADERS_BINARY_DIR ${TARGET_DIR}/shaders)

#function (add_slang_shader_target TARGET)
#    set(options)
#    set(oneValueArgs)
#    set(multiValueArgs SOURCES)
#    cmake_parse_arguments (SHADER "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
#
#    set (ENTRY_POINTS -entry vertMain -entry fragMain)
#    add_custom_command (
#            OUTPUT ${SHADERS_BINARY_DIR}
#            COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_BINARY_DIR}
#    )
#    add_custom_command (
#            OUTPUT  ${SHADERS_BINARY_DIR}/slang.spv
#            COMMAND ${SLANGC_EXECUTABLE} ${SHADER_SOURCES} -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name ${ENTRY_POINTS} -o slang.spv
#            WORKING_DIRECTORY ${SHADERS_SOURCE_DIR}
#            DEPENDS ${SHADER_SOURCES}
#            COMMENT "Compiling Slang Shaders"
#            VERBATIM
#    )
#    add_custom_target (${TARGET} DEPENDS ${SHADER_SOURCES})
#endfunction()

#add_slang_shader_target(slang_shader_compile SOURCES shader.slang)
add_custom_command(
        OUTPUT ${SHADERS_BINARY_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_BINARY_DIR}
        COMMENT "Creating shaders directory"
)
set (ENTRY_POINTS -entry vertMain -entry fragMain)
set(SHADER_SOURCES shader.slang)
add_custom_target(
        compile_shaders
        ALL
        COMMAND ${SLANGC_EXECUTABLE} ${SHADER_SOURCES}
            -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name
            ${ENTRY_POINTS}
            -o ${SHADERS_BINARY_DIR}/slang.spv
        WORKING_DIRECTORY ${SHADERS_SOURCE_DIR}
        DEPENDS ${SHADER_SOURCES} ${SHADERS_BINARY_DIR}
        VERBATIM
)
add_dependencies(${TARGET} compile_shaders)