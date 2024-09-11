cmake_minimum_required(VERSION 3.18)
project(DeePCpp)
enable_testing()

set(CMAKE_CXX_STANDARD 20)
if (MSVC)
	set(CMAKE_CXX_FLAGS "/std:c++latest /EHsc /MP")
	set(CMAKE_CXX_FLAGS_DEBUG "/Od /MDd")
	set(CMAKE_CXX_FLAGS_RELEASE "/Ox /MD /GL")
else()
	set(CMAKE_CXX_FLAGS "-march=native -w")
	set(CMAKE_CXX_FLAGS_DEBUG "-g")
	set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

include(FetchContent)

find_package (Eigen3 3.4 NO_MODULE)
 
# Eigen
if (NOT TARGET Eigen3::Eigen)
        FetchContent_Declare(
                Eigen
                GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
                GIT_TAG 3.4.0
                GIT_SHALLOW TRUE
                GIT_PROGRESS TRUE
        )
        set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
        set(EIGEN_BUILD_TESTING OFF)
        set(EIGEN_BUILD_DOC OFF)
        set(EIGEN_BUILD_PKGCONFIG OFF)
        FetchContent_MakeAvailable(Eigen)
endif (NOT TARGET Eigen3::Eigen)

# Google Test
FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
)
FetchContent_MakeAvailable(googletest)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Google Benchmark
FetchContent_Declare(googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.5
)
FetchContent_MakeAvailable(googlebenchmark)

# OpenMP
find_package(OpenMP)
link_libraries(OpenMP::OpenMP_CXX)

add_subdirectory(src/deepcpp)
add_subdirectory(tests/deepcpp)
