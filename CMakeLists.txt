cmake_minimum_required(VERSION 3.12.0)
project(lab02 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED "ON")

# Set default build type to Release.
if(NOT CMAKE_BUILD_TYPE OR "${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
message(STATUS)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS)
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  add_definitions(-DBUILD_TYPE_DEBUG)
endif()

# Locate MPI compiler.
find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILER "${MPI_CXX_COMPILER}")

# Locate Boost.
find_package(Boost 1.72.0 REQUIRED
  COMPONENTS filesystem iostreams serialization
  HINTS ${BOOST_DIR} $ENV{BOOST_DIR} $ENV{mkBoostPrefix})
message(STATUS "Using the Boost-${Boost_VERSION} configuration found at ${Boost_DIR}")
message(STATUS)
include_directories(${Boost_INCLUDE_DIRS})

# Locate deal.II and initialize its variables.
find_package(deal.II 9.3.1 REQUIRED
  HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR} $ENV{mkDealiiPrefix})
deal_ii_initialize_cached_variables()

# Add useful compiler flags.
#file(GLOB_RECURSE SOURCE_FILES "src/*.cpp" "src/*.hpp")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfloat-conversion -Wmissing-braces -Wnon-virtual-dtor -Wfatal-errors")

add_library(SystemAssembler STATIC src/system_assembler.cpp)
deal_ii_setup_target(SystemAssembler)


add_library(Poisson STATIC src/holzapfel/Poisson.cpp)
deal_ii_setup_target(Poisson)

add_library(LVBase STATIC src/lv_base.cpp)
deal_ii_setup_target(LVBase)

set(MODELS guccione holzapfel)

foreach(MODEL ${MODELS})
    add_library(LeftVentricle_${MODEL} STATIC src/${MODEL}/left_ventricle.cpp)
    deal_ii_setup_target(LeftVentricle_${MODEL})

    add_library(TensorUtils_${MODEL} STATIC src/${MODEL}/tensor_utils.cpp)
    deal_ii_setup_target(TensorUtils_${MODEL})

    add_executable(cardiac_${MODEL} src/${MODEL}/main.cpp)
    target_link_libraries(cardiac_${MODEL} LeftVentricle_${MODEL} LVBase SystemAssembler TensorUtils_${MODEL} Poisson)
    deal_ii_setup_target(cardiac_${MODEL})
endforeach()


add_executable(cardiac src/main.cpp)
target_link_libraries(cardiac
    LeftVentricle_holzapfel LeftVentricle_guccione LVBase
    SystemAssembler TensorUtils_holzapfel TensorUtils_guccione Poisson)
deal_ii_setup_target(cardiac)
