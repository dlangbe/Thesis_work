project(Layers)

cmake_minimum_required(VERSION 3.8)

include_directories("./")

set(CMAKE_CXX_FLAGS "-std=c++14 -g -I -O3 -Wall -Wextra")

add_executable(Layers layers.hpp
                        layers.cpp
                        secure_float.hpp
                        CNN.cpp)


