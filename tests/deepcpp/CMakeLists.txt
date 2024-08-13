include_directories(${CMAKE_SOURCE_DIR}/src/deepcpp)

add_executable(
    DeePCppTest
    test_algorithm.cpp
    test_matrix.cpp
    test_vector.cpp
)

target_link_libraries(
    DeePCppTest
    DeePCpp
    gtest_main
)

include(GoogleTest)
gtest_discover_tests(DeePCppTest)
