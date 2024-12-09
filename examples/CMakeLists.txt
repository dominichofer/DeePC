include_directories(${CMAKE_SOURCE_DIR}/src/deepcpp)

add_executable(
    cpp_1d
    cpp_1d.cpp
)

target_link_libraries(
    cpp_1d
    DeePCpp
)
