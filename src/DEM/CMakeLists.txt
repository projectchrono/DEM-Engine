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
	PUBLIC ${NVIDIAMathDir}
)

target_link_libraries(
	DEM
	PUBLIC CUB::CUB
	PUBLIC ${ChPF_IMPORTED_NAME}
)


set(DEM_headers
	${CMAKE_CURRENT_SOURCE_DIR}/kT.h
	${CMAKE_CURRENT_SOURCE_DIR}/dT.h
	${CMAKE_CURRENT_SOURCE_DIR}/API.h
	${CMAKE_CURRENT_SOURCE_DIR}/DEMDefines.h
	${CMAKE_CURRENT_SOURCE_DIR}/DEMStructs.h
	${CMAKE_CURRENT_SOURCE_DIR}/DEMModels.h
	${CMAKE_CURRENT_SOURCE_DIR}/VariableTypes.h
	${CMAKE_CURRENT_SOURCE_DIR}/DEMBdrsAndObjs.h
	${CMAKE_CURRENT_SOURCE_DIR}/HostSideHelpers.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/Samplers.hpp
)

set(DEM_sources
	${CMAKE_CURRENT_SOURCE_DIR}/kT.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/dT.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/APIPublic.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/APIPrivate.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/MeshUtils.cpp
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

