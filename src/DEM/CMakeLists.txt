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

if(DEME_ENABLE_NVTX)
	target_compile_definitions(DEM PUBLIC DEME_ENABLE_NVTX)
endif()

target_link_libraries(
	DEM
	PUBLIC CUB::CUB
)

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
	${CMAKE_CURRENT_SOURCE_DIR}/utils/ClumpMassProperties.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/AnalyticalOutput.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/CombinedOwnerUtils.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/HostSideHelpers.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/MeshUtils.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/Samplers.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/AuxClasses.h
)

set(DEM_sources
	${CMAKE_CURRENT_SOURCE_DIR}/kT.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/dT.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/APIPublic.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/APIPrivate.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/ClumpMassProperties.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/AnalyticalOutput.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/CombinedOwnerUtils.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/MeshUtils.cpp
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
	
	# Keep the compiled extension private so the public ``deme`` package can
	# provide a stable Python namespace and compatibility aliases.
	pybind11_add_module(_deme
		${DEM_headers}
		${DEM_sources}
		${CMAKE_CURRENT_SOURCE_DIR}/python/bindings.cpp
	)
	if(DEME_BUILD_VISUALIZER)
		target_sources(_deme PRIVATE $<TARGET_OBJECTS:visualization> $<TARGET_OBJECTS:visualization_bridge>)
	endif()
	
	set_target_properties(_deme PROPERTIES
		LINKER_LANGUAGE CXX
		CXX_STANDARD ${CXXSTD_SUPPORTED}
		POSITION_INDEPENDENT_CODE True
	)
	
	target_include_directories(_deme
		PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
		PUBLIC ${ProjectIncludeSource}
		PUBLIC ${ProjectIncludeGenerated}
	)
	
	target_link_libraries(_deme
		PUBLIC CUDA::cudart
		PUBLIC CUDA::nvrtc
		PUBLIC CUDA::cuda_driver
		PUBLIC core_python
		PUBLIC DEMERuntimeDataHelper_python
		PUBLIC algorithms
		PUBLIC pybind11::headers
	)
	if(DEME_BUILD_VISUALIZER)
		target_link_libraries(_deme PUBLIC raylib)
		target_compile_definitions(_deme PRIVATE DEME_HAS_VISUALIZER=1)
	endif()
	
	# Install the Python module
	install(TARGETS _deme
		LIBRARY DESTINATION deme
		COMPONENT python
	)

	# ``deme`` is the canonical package. DEME.py preserves the historical
	# capitalized import without duplicating the compiled extension.
	install(FILES
		${CMAKE_CURRENT_SOURCE_DIR}/python/deme/__init__.py
		${CMAKE_CURRENT_SOURCE_DIR}/python/deme/_cuda.py
		DESTINATION deme
		COMPONENT python
	)
	install(FILES
		${CMAKE_CURRENT_SOURCE_DIR}/python/DEME.py
		DESTINATION .
		COMPONENT python
	)
	
	message(STATUS "==== Completed Building Python Bindings ====")
endif()
