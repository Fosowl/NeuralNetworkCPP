cmake_minimum_required(VERSION 3.10)

project(neuralnetwork VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set( PROJECT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR} )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14 -Wfatal-errors -Wl, -ldl")
file(GLOB SRC
     "src/*.cpp"
)

add_executable(${PROJECT_NAME} ${SRC})
include_directories( include )

target_include_directories(neuralnetwork PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           )