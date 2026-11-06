cmake_minimum_required(VERSION 3.16)
project(ann_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(ANN_NATIVE "Compile with -march=native" ON)
option(ANN_SANITIZE "Build with AddressSanitizer and UBSan" OFF)

add_compile_options(-Wall -Wextra -Wpedantic)
if(ANN_NATIVE)
  add_compile_options(-march=native)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

if(ANN_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

find_package(OpenMP)

# Core library (header + sources). More sources are added as components land.
add_library(ann INTERFACE)
target_include_directories(ann INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(OpenMP_CXX_FOUND)
  target_link_libraries(ann INTERFACE OpenMP::OpenMP_CXX)
endif()

# Tests (doctest is vendored in third_party/ so no network is needed at build time)
enable_testing()
add_executable(ann_tests
  tests/test_main.cpp
  tests/test_dataset.cpp
)
target_include_directories(ann_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_link_libraries(ann_tests PRIVATE ann)
add_test(NAME ann_tests COMMAND ann_tests)

# Small tool that loads SIFT1M and prints shapes; replaced by bench/ later.
add_executable(inspect_data tools/inspect_data.cpp)
target_link_libraries(inspect_data PRIVATE ann)
