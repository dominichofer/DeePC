add_library(
    DeePCpp STATIC
    algorithm.h
    algorithm.cpp
    lti.h
    lti.cpp
)

target_link_libraries (DeePCpp Eigen3::Eigen)