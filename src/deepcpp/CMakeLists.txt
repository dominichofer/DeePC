add_library(
    DeePCpp STATIC
    algorithm.h
    algorithm.cpp
    deepc.h
    lti.h
    lti.cpp
)

target_link_libraries (DeePCpp Eigen3::Eigen)