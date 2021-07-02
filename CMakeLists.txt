# Copyright (c) 2021, SBEL GPU Development Team
# Copyright (c) 2021, University of Wisconsin - Madison
# All rights reserved.

# ---------------------------------------------------------------------------- #
# CMake Project Settings
# ---------------------------------------------------------------------------- #

cmake_minimum_required(VERSION 3.10)

# Version information
set(VERSION_MAJOR   0)
set(VERSION_MINOR   0)
set(VERSION_PATCH   0)

project(
	SBEL-GPU-Physics 
	VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}
	LANGUAGES CXX CUDA
)

# ---------------------------------------------------------------------------- #
# Global Configuration
# ---------------------------------------------------------------------------- #

# The compiler will build against this C++ standard, if available.
set(TargetCXXStandard "STD_AUTODETECT" CACHE STRING "The C++ standard used by the compiler")
set_property(
	CACHE TargetCXXStandard
	PROPERTY
	STRINGS STD_AUTODETECT STD_CXX11 STD_CXX14 STD_CXX17 STD_CXX20 STD_CXX2b
)

set(ProjectIncludeRoot "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_BINARY_DIR}/src")


# Find CUB library (this might need to be done in source-level config)
find_package(
	CUB REQUIRED
	PATHS ${CMAKE_CURRENT_LIST_DIR}/thirdparty/cub/cub
)

find_path(
	NVIDIAJitifyPath
	NAMES jitify.hpp
	PATHS "${CMAKE_CURRENT_LIST_DIR}/thirdparty/jitify"

)

# ---------------------------------------------------------------------------- #
# Source-level configuration
# ---------------------------------------------------------------------------- #

add_subdirectory(src/core)
add_subdirectory(src/granular)

# ---------------------------------------------------------------------------- #
# Final Link Step
# ---------------------------------------------------------------------------- #

add_library(
	sbel_gpu_physics 
	SHARED
	$<TARGET_OBJECTS:core>
	$<TARGET_OBJECTS:granular>
	)
	
add_subdirectory(src/demo)