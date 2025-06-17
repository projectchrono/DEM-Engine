# Copyright (c) 2021, SBEL GPU Development Team
# Copyright (c) 2021, University of Wisconsin - Madison
#
#	SPDX-License-Identifier: BSD-3-Clause

# ---------------------------------------------------------------------------- #
# CMake Project Settings
# ---------------------------------------------------------------------------- #

cmake_minimum_required(VERSION 3.18)

# Version information
set(DEME_VERSION_MAJOR   2)
set(DEME_VERSION_MINOR   1)
set(DEME_VERSION_PATCH   0)

project(
	Chrono-DEM-Engine
	VERSION ${DEME_VERSION_MAJOR}.${DEME_VERSION_MINOR}.${DEME_VERSION_PATCH}
	LANGUAGES CXX CUDA
)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting CMAKE_BUILD_TYPE to 'Release' (default)")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

add_compile_options(-fno-lto)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

# Mark compile time (not runtime), affects how some kernel files include
add_compile_definitions(DEME_BEING_CMAKE_COMPILED)

# ---------------------------------------------------------------------------- #
# Additional Packages
# ---------------------------------------------------------------------------- #

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(FetchContent)


find_package(CUDAToolkit REQUIRED)
find_package(Python COMPONENTS Interpreter Development)

add_subdirectory(thirdparty/pybind11)

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

set(CMAKE_INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib/python3.10/site-packages/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Find CUB library (this might need to be done in source-level config)
find_package(
	CUB REQUIRED
	HINTS ${CUDAToolkit_ROOT}/lib64/cmake/cub
)

# Find NVIDIA's Jitify library
find_path(
	NVIDIAJitifyPath
	NAMES jitify.hpp
	PATHS "${CMAKE_CURRENT_LIST_DIR}/thirdparty/jitify"
)

set(NVIDIAJitifyPath "${CMAKE_CURRENT_LIST_DIR}/thirdparty/jitify")

# Include pyBind11 library
include_directories(${PYTHON_INCLUDE_DIRS})

# Let the user decide if they want to use ChPF
option(USE_CHPF "Toggle the use of ChPF for outputting" OFF)

if(USE_CHPF)
	# Find ChPF, else fetch it 
    find_package(ChPF 3.0 QUIET)
    if (NOT ChPF_FOUND)         
        FetchContent_Declare(
            ChPF
            GIT_REPOSITORY  https://gitlab.com/uwsbel/chpf.git
            GIT_TAG         0699f01e720cecf58719a1651ab797c6a62a1897
        )
        FetchContent_MakeAvailable(ChPF)
    endif()

	# Based on whether ChPF was found or fetched, set the target name CMake can use to find it 
    if (ChPF_FOUND) 
        set(ChPF_IMPORTED_NAME "ChPF::ChPF")
    elseif (chpf_POPULATED)
        set(ChPF_IMPORTED_NAME "ChPF")
    endif()
endif()

# Let the user decide if they want to use managed arrays, rather than default cudaMalloc and cudaMallocHost memory.
# Note that turning this on gives no performance benefits, and it's considered legacy.
set(USE_MANAGED_ARRAYS_DESC
	"Use managed CUDA arrays (instead of manual CUDA host and device memory) to store data"
	"Switching it on makes the project operable on a Linux platform (excluding WSL) only."
)
option(USE_MANAGED_ARRAYS "${USE_MANAGED_ARRAYS_DESC}" OFF)

# ---------------------------------------------------------------------------- #
# Global Configuration
# ---------------------------------------------------------------------------- #

include(cmake/CxxStdAutodetect.cmake)

# The compiler will build against this C++ standard, if available.
set(TargetCXXStandard "STD_AUTODETECT" CACHE STRING "The C++ standard used by the compiler")
set_property(
	CACHE TargetCXXStandard
	PROPERTY
	STRINGS STD_AUTODETECT STD_CXX11 STD_CXX14 STD_CXX17 STD_CXX20
)

# Defining option to build PyDEME wrapper

option(PYTHON_BUILD "Sets the build process for Python wrapper around DEME" OFF)
mark_as_advanced(PYTHON_BUILD)

# Convert the standard into something CMake will understand
if(TargetCXXStandard STREQUAL STD_CXX11)
	set(CXXSTD_MAX 11)
elseif(TargetCXXStandard STREQUAL STD_CXX14)
	set(CXXSTD_MAX 14)
elseif(TargetCXXStandard STREQUAL STD_CXX17)
	set(CXXSTD_MAX 17)
elseif(TargetCXXStandard STREQUAL STD_CXX20)
	set(CXXSTD_MAX 20)
else()
	set(CXXSTD_MAX 17)
endif()
cxx_std_autodetect()

# Allow the use of #include <> for project headers (allowing cleaner better relative pathing)
set(ProjectIncludeSource "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(ProjectIncludeGenerated "${CMAKE_BINARY_DIR}/src")

# Global fix for CUDA language bug
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})


#------------------------------------------------------------
# Install destinations for data and demo programs
#------------------------------------------------------------

set(DEME_INSTALL_DEMO "bin")

# ---------------------------------------------------------------------------- #
# Source-level configuration
# ---------------------------------------------------------------------------- #

add_subdirectory(src/core)
add_subdirectory(src/DEM)
add_subdirectory(src/algorithms)

# ---------------------------------------------------------------------------- #
# Final Library Generation
# ---------------------------------------------------------------------------- #

add_library(
	simulator_multi_gpu
	STATIC
	$<TARGET_OBJECTS:core>
	$<TARGET_OBJECTS:DEM>
	$<TARGET_OBJECTS:algorithms>
)

# Extract include directories from the object bundles
get_target_property(CORE_INTERFACE core INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(DEM_INTERFACE DEM INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(ALGORITHMS_INTERFACE algorithms INTERFACE_INCLUDE_DIRECTORIES)

# All targets use the same include directories, so we can cheat and just export one of them
target_include_directories(simulator_multi_gpu
	PUBLIC ${CORE_INTERFACE}
)

if(NOT PYTHON_BUILD)
	# If use ChPF, inform the source
	if(USE_CHPF)
		target_compile_definitions(simulator_multi_gpu PUBLIC DEME_USE_CHPF)
		set(USE_CHPF_STR "ON")

		target_link_libraries(simulator_multi_gpu
			PUBLIC CUDA::cudart
			PUBLIC CUDA::nvrtc
			PUBLIC CUDA::cuda_driver
			PUBLIC ${ChPF_IMPORTED_NAME}
			PUBLIC DEMERuntimeDataHelper
		)
	else()
		set(USE_CHPF_STR "OFF")

		target_link_libraries(simulator_multi_gpu
			PUBLIC CUDA::cudart
			PUBLIC CUDA::nvrtc
			PUBLIC CUDA::cuda_driver
			PUBLIC DEMERuntimeDataHelper
		)
	endif()
else()
	# py version does not use ChPF
	set(USE_CHPF_STR "OFF")
	
	target_link_libraries(simulator_multi_gpu
		PUBLIC CUDA::cudart
		PUBLIC CUDA::nvrtc
		PUBLIC CUDA::cuda_driver
		PUBLIC DEMERuntimeDataHelper_python
		PUBLIC ${PYTHON_LIBRARIES}
	)
endif()

# If use managed arrays, define a macro
if(USE_MANAGED_ARRAYS)
	target_compile_definitions(simulator_multi_gpu PUBLIC DEME_USE_MANAGED_ARRAYS)
endif()

# Specific to Windows...
if(WIN32)
	target_link_libraries(simulator_multi_gpu 
		PUBLIC dbghelp)
endif()

# Attach include directories to the top-level library target
set_target_properties(simulator_multi_gpu
	PROPERTIES
	LINKER_LANGUAGE CUDA
)

# ---------------------------------------------------------------------------- #
# Export and Install The Generated Targets
# ---------------------------------------------------------------------------- #

set(config_directory ${CMAKE_INSTALL_LIBDIR}/cmake/DEME)


# Install the library, creating an export target for it as well
if(NOT PYTHON_BUILD)
	install(
		TARGETS simulator_multi_gpu DEMERuntimeDataHelper
		EXPORT DEMETargets
		LIBRARY
			DESTINATION ${CMAKE_INSTALL_LIBDIR}
	)
else()
	install(
		TARGETS simulator_multi_gpu DEMERuntimeDataHelper_python
		EXPORT DEMETargets
		LIBRARY
			DESTINATION ${CMAKE_INSTALL_LIBDIR}
	)
endif()

install(CODE "message(\"Removing build-tree libraries from installation target...\")")
install(CODE "file(REMOVE \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/libDEMERuntimeDataHelper.so\")")

if(NOT PYTHON_BUILD)
	install(
		TARGETS DEMERuntimeDataHelper_install
		EXPORT DEMETargets
		LIBRARY
			DESTINATION ${CMAKE_INSTALL_LIBDIR}
	)
endif()

# Export the generated library target
export(EXPORT DEMETargets
	FILE "${CMAKE_CURRENT_BINARY_DIR}/DEME/DEMETargets.cmake"
	NAMESPACE DEME::
)

# Set the install location for the exported target file
install(
	EXPORT DEMETargets
	DESTINATION
		${config_directory}
	NAMESPACE DEME::
)


# These macros aid in the generation of the installable config files


# Generate the **Build Tree** version of DEMEConfig.cmake from the base file
set(CONF_INCLUDE_DIRS "${ProjectIncludeSource}" "${ProjectIncludeGenerated}")

configure_file(cmake/DEMEConfig.cmake.in
	"${CMAKE_CURRENT_BINARY_DIR}/DEME/DEMEConfig.cmake"
	@ONLY
)


# Generate the _Install Tree_ version of DEMEConfig.cmake from the base file
file(RELATIVE_PATH REL_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/${config_directory} ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})
set(CONF_INCLUDE_DIRS "\${DEMECMakeDir}/${REL_INCLUDE_DIR}")
set(CONF_SHARE_DATA_DIRS "\${DEMECMakeDir}/${REL_INCLUDE_DIR}/../share/DEME/data")

# NOTE: This is placed in CMAKE_FILES_DIRECTORY to keep it out of the build tree
configure_file(cmake/DEMEConfig.cmake.in
	"${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/DEMEConfig.cmake"
	@ONLY
)


# Automatically generate a DEMEConfigVersion.cmake file

write_basic_package_version_file(
	${CMAKE_CURRENT_BINARY_DIR}/DEME/DEMEConfigVersion.cmake
	VERSION ${DEME_VERSION_MAJOR}.${DEME_VERSION_MINOR}.${DEME_VERSION_PATCH}
	COMPATIBILITY AnyNewerVersion
)

# Install the generated config files
install(
	FILES
		"${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/DEMEConfig.cmake"
		"${CMAKE_CURRENT_BINARY_DIR}/DEME/DEMEConfigVersion.cmake"
	DESTINATION
		${config_directory}
)

# ---------------------------------------------------------------------------- #
# Copy data directory over
# ---------------------------------------------------------------------------- #
file(COPY ${CMAKE_CURRENT_LIST_DIR}/data/ DESTINATION ${CMAKE_BINARY_DIR}/data/)
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/data/ DESTINATION ${CMAKE_INSTALL_DATADIR}/DEME/data/)

# ---------------------------------------------------------------------------- #
# Copy kernels over
# ---------------------------------------------------------------------------- #
file(COPY ${CMAKE_CURRENT_LIST_DIR}/src/kernel/ DESTINATION ${CMAKE_BINARY_DIR}/kernel/)
file(COPY ${CMAKE_CURRENT_LIST_DIR}/src/DEM/Defines.h DESTINATION ${CMAKE_BINARY_DIR}/DEM)
file(COPY ${CMAKE_CURRENT_LIST_DIR}/src/DEM/VariableTypes.h DESTINATION ${CMAKE_BINARY_DIR}/DEM)
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/src/kernel/ DESTINATION ${CMAKE_INSTALL_DATADIR}/DEME/kernel/)

# ---------------------------------------------------------------------------- #
# Build embedded demos
# ---------------------------------------------------------------------------- #
add_subdirectory(src/demo)

