# cmake needs this line
cmake_minimum_required(VERSION 2.8)

# Define project name
project(HOG_project)

#add_definitions( -fopenmp -O2)
SET(GCC_COVERAGE_COMPILE_FLAGS "-std=c++14 -O2")
#SET(GCC_COVERAGE_LINK_FLAGS    "-fopenmp")
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )
#SET( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}" )

FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")



# Add OpenCV headers location to your include paths
include_directories(${OpenCV_INCLUDE_DIRS} ..)

# Declare the executable target built from your sources
add_executable(test_functional test_functional.cpp ../HOG.cpp)

# Link your application with OpenCV libraries
target_link_libraries(test_functional ${OpenCV_LIBS})
