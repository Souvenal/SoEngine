set(SLANGC_EXECUTABLE slangc)
set(SHADERS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SHADERS_BINARY_DIR ${TARGET_DIR}/shaders)

# 目录创建：生成一个哨兵文件用于依赖
add_custom_command(
    OUTPUT ${SHADERS_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_BINARY_DIR}
    COMMENT "Creating shaders output directory"
    VERBATIM
)
add_custom_target(shaders_dir DEPENDS ${SHADERS_BINARY_DIR})

# 聚合目标（不直接执行命令，只收集依赖）
add_custom_target(compile_shaders)

# Function: compile_slang(<source> <entry> [more entries...])
function(compile_slang SHADER_SOURCE)
    if (ARGC LESS 2)
        message(FATAL_ERROR "compile_slang requires at least 1 entry point")
    endif()

    # Collect entry points (all arguments after first)
    list(REMOVE_AT ARGV 0)
    set(ENTRY_POINTS ${ARGV})

    # Build -entry 参数列表
    set(ENTRY_ARGS)
    foreach(EP IN LISTS ENTRY_POINTS)
        list(APPEND ENTRY_ARGS -entry ${EP})
    endforeach()

    # Output file: keep original name, only change extension to .spv
    get_filename_component(SHADER_BASE "${SHADER_SOURCE}" NAME_WE)
    set(OUTPUT_FILE "${SHADERS_BINARY_DIR}/${SHADER_BASE}.spv")

    add_custom_command(
        OUTPUT  ${OUTPUT_FILE}
        COMMAND ${SLANGC_EXECUTABLE} ${SHADER_SOURCE}
                -target spirv
                -profile spirv_1_4
                -emit-spirv-directly
                -fvk-use-entrypoint-name
                ${ENTRY_ARGS}
                -o ${OUTPUT_FILE}
        WORKING_DIRECTORY ${SHADERS_SOURCE_DIR}
        DEPENDS ${SHADER_SOURCE} ${SHADERS_BINARY_DIR}
        COMMENT "Compiling ${SHADER_SOURCE} -> ${OUTPUT_FILE} (entries: ${ENTRY_POINTS})"
        VERBATIM
    )

    # Each shader gets its own target
    set(SHADER_TARGET "${SHADER_BASE}_${FIRST_ENTRY}_target")
    add_custom_target(${SHADER_TARGET} DEPENDS ${OUTPUT_FILE})

    # 目录 -> 单个shader -> 聚合
    add_dependencies(${SHADER_TARGET} shaders_dir)
    add_dependencies(compile_shaders ${SHADER_TARGET})
endfunction()

compile_slang(shader.slang vertMain fragMain)
compile_slang(compute.slang vertMain fragMain compMain)
compile_slang(gradient.slang compMain)
compile_slang(gradient_color.slang compMain)
compile_slang(sky.slang compMain)

add_dependencies(${TARGET} compile_shaders)