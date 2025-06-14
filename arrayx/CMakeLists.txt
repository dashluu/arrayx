cmake_minimum_required(VERSION 3.30)
project(arrayx)

set(CMAKE_CXX_STANDARD 23)

# Set up nanobind
set(DEV_MODULE Development.Module)
find_package(Python 3.12 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

add_definitions(-DPROJECT_ROOT="${CMAKE_SOURCE_DIR}/..")

file(GLOB HEADER_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/core/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/device/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/graph/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/array/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/nn/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/python/*.h"
)

file(GLOB SRC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/device/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/graph/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/array/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/nn/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/python/*.cpp"
)

if(APPLE)
    add_subdirectory(runtime/metal/kernels)
    file(GLOB MTL_HEADER_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/device/metal/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/graph/metal/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal/*.h"
    )
    file(GLOB MTL_SRC_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/device/metal/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/graph/metal/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal/*.cpp"
    )
    # set(CMAKE_BUILD_TYPE Debug)
    # add_executable(${PROJECT_NAME} main.cpp ${SRC_FILES} ${HEADER_FILES} ${MTL_SRC_FILES} ${MTL_HEADER_FILES})
    nanobind_add_module(${PROJECT_NAME} ${SRC_FILES} ${HEADER_FILES} ${MTL_SRC_FILES} ${MTL_HEADER_FILES})
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

# Add stub files
function(add_nanobind_stub STUB_NAME MODULE_NAME)
    nanobind_add_stub(
        ${STUB_NAME}
        MODULE arrayx.${MODULE_NAME}
        OUTPUT arrayx/${MODULE_NAME}/__init__.pyi
        PYTHON_PATH $<TARGET_FILE_DIR:arrayx>
        DEPENDS arrayx
    )
endfunction()

add_nanobind_stub(core_stub core)
add_nanobind_stub(nn_stub nn)
add_nanobind_stub(optim_stub optim)

# Define output directory for Python module and move files after build
set(PYTHON_MODULE_DIR "${CMAKE_SOURCE_DIR}/../python")
message(STATUS "PYTHON_MODULE_DIR is set to: ${PYTHON_MODULE_DIR}")
get_filename_component(PYTHON_MODULE_DIR_ABS "${PYTHON_MODULE_DIR}" ABSOLUTE)
message(STATUS "Absolute path: ${PYTHON_MODULE_DIR_ABS}")
# Add custom command to move files after build
add_custom_target(move_files ALL
    # Remove the destination directory if it exists
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${PYTHON_MODULE_DIR}/${PROJECT_NAME}"
    COMMAND ${CMAKE_COMMAND} -E rename "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}" "${PYTHON_MODULE_DIR}/${PROJECT_NAME}"
    COMMAND ${CMAKE_COMMAND} -E touch "${PYTHON_MODULE_DIR}/${PROJECT_NAME}/__init__.pyi"
    COMMAND ${CMAKE_COMMAND} -E rename "$<TARGET_FILE:${PROJECT_NAME}>" "${PYTHON_MODULE_DIR}/$<TARGET_FILE_NAME:${PROJECT_NAME}>"
    DEPENDS core_stub nn_stub optim_stub
    COMMENT "Moving Python module and stub files"
)
