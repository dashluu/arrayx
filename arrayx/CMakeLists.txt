cmake_minimum_required(VERSION 3.30)
project(arrayx)

set(CMAKE_CXX_STANDARD 23)

file(GLOB HEADER_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/core/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/graph/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/utils.h"
)

file(GLOB SRC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/graph/*.cpp"
)

if(APPLE)
    add_subdirectory(runtime/metal/kernels)
    file(GLOB MTL_HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal/runner/*.h")
    file(GLOB MTL_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal/runner/*.cpp")
    # set(CMAKE_BUILD_TYPE Debug)
    add_executable(${PROJECT_NAME} main.cpp ${SRC_FILES} ${HEADER_FILES} ${MTL_SRC_FILES} ${MTL_HEADER_FILES})
    set(MTL_CPP "${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal/metal-cpp")
    target_include_directories(${PROJECT_NAME} PRIVATE ${MTL_CPP})
    target_link_libraries(${PROJECT_NAME} PRIVATE
        "-framework Metal"
        "-framework Foundation"
        "-framework QuartzCore"
    )
    # Ensure kernels are built before your executable
    add_dependencies(${PROJECT_NAME} mtl_kernels)
endif()
