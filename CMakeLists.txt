cmake_minimum_required(VERSION 3.20)
project(Gungnir VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release for noticeable speed
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(MSVC)
    add_compile_options(/W4 /permissive- /EHsc /Zc:preprocessor)
    add_compile_options($<$<CONFIG:Release>:/O2>)
    add_compile_options($<$<CONFIG:Release>:/arch:AVX2>)
    add_compile_options($<$<CONFIG:Release>:/GL>)
    add_link_options($<$<CONFIG:Release>:/LTCG>)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    add_compile_options($<$<CONFIG:Release>:-O3>)
    add_compile_options($<$<CONFIG:Release>:-mavx2>)
endif()

add_executable(gungnir
    src/main.cpp
    src/bitboard.cpp
    src/attacks.cpp
    src/position.cpp
    src/movegen.cpp
    src/perft.cpp
)

target_include_directories(gungnir PRIVATE src)
