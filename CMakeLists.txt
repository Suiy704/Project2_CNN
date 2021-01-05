cmake_minimum_required(VERSION 3.17)
project(CNN)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 14)

add_executable(CNN Matrix.cpp main.cpp)
target_link_libraries(CNN ${OpenCV_LIBS})
