include_directories(${CMAKE_SOURCE_DIR}/src/deepcpp)

add_executable(
    DeePCppTest
    test_algorithm.cpp
    test_deepc.cpp
    test_helpers.cpp
    test_lti.cpp
)

target_link_libraries(
    DeePCppTest
    DeePCpp
    gtest_main
)

include(GoogleTest)
gtest_discover_tests(DeePCppTest)
