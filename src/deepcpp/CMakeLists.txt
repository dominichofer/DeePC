add_library(
    DeePCpp STATIC
    algorithm.h
    algorithm.cpp
    controller.h
    controller.cpp
    deepc.h
    deepc.cpp
    finite_queue.h
    finite_queue.cpp
    lti.h
    lti.cpp
)

target_link_libraries(DeePCpp Eigen3::Eigen)