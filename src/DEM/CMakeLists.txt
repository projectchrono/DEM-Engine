# Copyright (c) 2021, SBEL GPU Development Team
# Copyright (c) 2021, University of Wisconsin - Madison
# 
#	SPDX-License-Identifier: BSD-3-Clause

message(STATUS "==== Building DEM Components ====")

add_library(DEM OBJECT)

target_include_directories(
	DEM
	PUBLIC ${ProjectIncludeSource}
	PUBLIC ${ProjectIncludeGenerated}
)

if(USE_CHPF)
	target_link_libraries(
		DEM
		PUBLIC CUB::CUB
		PUBLIC ${ChPF_IMPORTED_NAME}
	)
else()
	target_link_libraries(
		DEM
		PUBLIC CUB::CUB
	)
endif()

# if(WIN32)
# target_compile_options(DEM PRIVATE /GR)
# endif()

set(DEM_headers
	${CMAKE_CURRENT_SOURCE_DIR}/kT.h
	${CMAKE_CURRENT_SOURCE_DIR}/dT.h
	${CMAKE_CURRENT_SOURCE_DIR}/API.h
	${CMAKE_CURRENT_SOURCE_DIR}/Defines.h
	${CMAKE_CURRENT_SOURCE_DIR}/Structs.h
	${CMAKE_CURRENT_SOURCE_DIR}/Models.h
	${CMAKE_CURRENT_SOURCE_DIR}/VariableTypes.h
	${CMAKE_CURRENT_SOURCE_DIR}/BdrsAndObjs.h
	${CMAKE_CURRENT_SOURCE_DIR}/utils/HostSideHelpers.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/Samplers.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/AuxClasses.h
)

set(DEM_sources
	${CMAKE_CURRENT_SOURCE_DIR}/kT.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/dT.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/APIPublic.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/APIPrivate.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/MeshUtils.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/AuxClasses.cpp
)

target_sources(
	DEM
	PUBLIC ${DEM_headers}
	PRIVATE ${DEM_sources}
)

set_target_properties(
	DEM PROPERTIES
	POSITION_INDEPENDENT_CODE True
	CXX_STANDARD ${CXXSTD_SUPPORTED}
	PUBLIC_HEADER "${DEM_headers}"
)

#foreach(header ${DEM_headers})
#	set_target_properties(DEM PROPERTIES PUBLIC_HEADER ${header})
#endforeach()

install(
	TARGETS DEM
	PUBLIC_HEADER
		DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/DEM 
)

install(
	DIRECTORY utils
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/DEM
	FILES_MATCHING
		PATTERN "utils/*.h"
		PATTERN "utils/*.hpp"
)

# Build Python module if requested
if(DEME_BUILD_PYTHON)
	message(STATUS "==== Building Python Bindings ====")
	
	pybind11_add_module(DEME 
		${DEM_headers}
		${DEM_sources}
		${CMAKE_CURRENT_SOURCE_DIR}/python/bindings.cpp
	)
	
	set_target_properties(DEME PROPERTIES 
		LINKER_LANGUAGE CXX
		CXX_STANDARD ${CXXSTD_SUPPORTED}
		POSITION_INDEPENDENT_CODE True
	)
	
	target_include_directories(DEME
		PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
		PUBLIC ${ProjectIncludeSource}
		PUBLIC ${ProjectIncludeGenerated}
	)
	
	target_link_libraries(DEME
		PUBLIC CUDA::cudart
		PUBLIC CUDA::nvrtc
		PUBLIC CUDA::cuda_driver
		PUBLIC core
		PUBLIC DEMERuntimeDataHelper_python
		PUBLIC algorithms
		PUBLIC pybind11::headers
	)
	
	# Install the Python module
	install(TARGETS DEME
		LIBRARY DESTINATION .
		COMPONENT python
	)
	
	message(STATUS "==== Completed Building Python Bindings ====")
endif()
