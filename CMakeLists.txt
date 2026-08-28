cmake_minimum_required(VERSION 3.15)
project(VeltraxxSparseEngine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  add_compile_options(/O2 /W4)
else()
  add_compile_options(-O3 -Wall -Wextra -march=native)
endif()

add_executable(sparse_engine
    main/main.cpp
    main/csr_loader.cpp
    main/sparse_engine.cpp
    main/benchmark.cpp
    main/mnist_loader.cpp
)

target_include_directories(sparse_engine PRIVATE main/)
