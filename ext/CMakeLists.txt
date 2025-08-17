add_library(stb SHARED stb.cpp)
target_include_directories(stb PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/stb)

add_library(tinyobjloader SHARED tiny_obj_loader.cpp)
target_include_directories(tinyobjloader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/tinyobjloader)