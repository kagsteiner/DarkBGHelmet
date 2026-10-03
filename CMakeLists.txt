cmake_minimum_required(VERSION 3.20)
project(DarkBGHelmet CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Wno-sign-compare)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

option(BG_NATIVE "Optimise for the build machine's CPU" ON)
if(BG_NATIVE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    add_compile_options(-mcpu=native)
  else()
    add_compile_options(-march=native)
  endif()
endif()

find_package(Threads REQUIRED)

file(GLOB_RECURSE BG_LIB_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_SOURCE_DIR}/engine/core/*.cpp
  ${CMAKE_SOURCE_DIR}/engine/nn/*.cpp
  ${CMAKE_SOURCE_DIR}/engine/eval/*.cpp
  ${CMAKE_SOURCE_DIR}/engine/search/*.cpp
  ${CMAKE_SOURCE_DIR}/engine/api/*.cpp)
add_library(bgengine STATIC ${BG_LIB_SOURCES})
target_include_directories(bgengine PUBLIC ${CMAKE_SOURCE_DIR}/engine ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(bgengine PUBLIC Threads::Threads)

add_executable(bgcli engine/cli/main.cpp)
target_link_libraries(bgcli bgengine)

add_executable(bgei engine/cli/bgei.cpp)
target_link_libraries(bgei bgengine)

add_executable(bgplay engine/cli/play.cpp)
target_link_libraries(bgplay bgengine)

add_executable(bgselfplay engine/cli/selfplay.cpp)
target_link_libraries(bgselfplay bgengine)

if(EXISTS ${CMAKE_SOURCE_DIR}/engine/server/main.cpp)
  add_executable(bgserver engine/server/main.cpp)
  target_link_libraries(bgserver bgengine)
endif()

file(GLOB BG_TRAIN_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/training/*.cpp)
if(BG_TRAIN_SOURCES)
  add_executable(bgtrain ${BG_TRAIN_SOURCES})
  target_link_libraries(bgtrain bgengine)
endif()

file(GLOB BG_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/*.cpp)
add_executable(bgtests ${BG_TEST_SOURCES})
target_link_libraries(bgtests bgengine)
target_compile_definitions(bgtests PRIVATE BG_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
