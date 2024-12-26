add_library(
    DeePCpp SHARED
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


install(
    FILES
        algorithm.h
        controller.h
        deepc.h
        finite_queue.h
        lti.h
    DESTINATION include/DeePCpp
)

install(
    TARGETS DeePCpp
    EXPORT DeePCppTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(
    EXPORT DeePCppTargets
    FILE DeePCppTargets.cmake
    NAMESPACE DeePCpp::
    DESTINATION lib/cmake/DeePCpp
)