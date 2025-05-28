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
	PUBLIC ${PYTHON_INCLUDE_DIRS}
)

if(NOT PYTHON_BUILD)
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
else()
	# Python version does not use ChPF
	target_link_libraries(
		DEM
		PUBLIC CUB::CUB
		PUBLIC ${PYTHON_LIBRARIES}
		PUBLIC ${CMAKE_INSTALL_INCLUDEDIR}/DEM
		PUBLIC core
		PUBLIC DEMERuntimeDataHelper_python
		PUBLIC algorithms
		pybind11::headers
		-static-libgcc -static-libstdc++
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
	${CMAKE_CURRENT_SOURCE_DIR}/HostSideHelpers.hpp
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

set(PyDEME_sources
	${CMAKE_CURRENT_SOURCE_DIR}/PyDEME.cpp
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
	INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
)

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

if(PYTHON_BUILD)
	message(STATUS "==== Building PyDEME Components ====")

	pybind11_add_module(DEME ${DEM_headers} ${DEM_sources} ${PyDEME_sources})

	set_target_properties(DEME PROPERTIES LINKER_LANGUAGE CXX)

	target_include_directories(
		DEME
		PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
		PUBLIC ${ProjectIncludeSource}
		PUBLIC ${ProjectIncludeGenerated}
		PUBLIC ${CMAKE_INSTALL_INCLUDEDIR}/DEM
		PUBLIC ${CMAKE_INSTALL_INCLUDEDIR}/core
		PUBLIC ${CMAKE_INSTALL_INCLUDEDIR}/algorithms
	)

	target_link_libraries(
		DEME
		PUBLIC CUDA::cudart
		PUBLIC CUDA::nvrtc
		PUBLIC CUDA::cuda_driver	
		PUBLIC ${ChPF_IMPORTED_NAME}
		PUBLIC ${PYTHON_LIBRARIES}
		PUBLIC core
		PUBLIC DEMERuntimeDataHelper_python
		PUBLIC algorithms
		PUBLIC pybind11::headers
		PRIVATE pybind11::embed
		-static-libgcc -static-libstdc++
	)

	set_target_properties(
		DEME PROPERTIES
		POSITION_INDEPENDENT_CODE True
		CXX_STANDARD ${CXXSTD_SUPPORTED}
		PUBLIC_HEADER "${DEM_headers}"
		INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib/python3.10/site-packages/lib"
	)

	add_compile_options(-fno-lto)

	message(STATUS "==== Completed Building PyDEME Components ====")
endif()
