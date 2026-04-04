cmake_minimum_required(VERSION 3.14)
project(LatLibrary)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 3.11 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

file(GLOB SRC_FILES src/*.cpp)

add_library(lat_core SHARED ${SRC_FILES})
target_include_directories(lat_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

pybind11_add_module(lattice_env 
    pybind/pybind_wrapper.cpp
    pybind/PyLatticeEnv.cpp
    pybind/RL_ENUM_Wrapper.cpp
)

target_link_libraries(lattice_env PRIVATE 
    lat_core
    pybind11::module
    Python3::Module
)

target_include_directories(lattice_env PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/pybind
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test.cpp")
    add_executable(test_exec ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp)
    target_link_libraries(test_exec lat_core)
    enable_testing()
    add_test(NAME RunTestExec COMMAND test_exec)
endif()

install(TARGETS lat_core DESTINATION lib)
install(TARGETS lattice_env DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)