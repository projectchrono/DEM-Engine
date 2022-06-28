# Copyright (c) 2021, SBEL GPU Development Team
# Copyright (c) 2021, University of Wisconsin - Madison
# All rights reserved.

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
)


set(DEM_headers
	${CMAKE_CURRENT_SOURCE_DIR}/kT.h
	${CMAKE_CURRENT_SOURCE_DIR}/dT.h
	${CMAKE_CURRENT_SOURCE_DIR}/ApiSystem.h
	${CMAKE_CURRENT_SOURCE_DIR}/DEMDefines.h
	${CMAKE_CURRENT_SOURCE_DIR}/DEMStructs.h
	${CMAKE_CURRENT_SOURCE_DIR}/DEMModels.h
	${CMAKE_CURRENT_SOURCE_DIR}/VariableTypes.h
	${CMAKE_CURRENT_SOURCE_DIR}/DEMBdrsAndObjs.h
	${CMAKE_CURRENT_SOURCE_DIR}/HostSideHelpers.hpp
)

set(DEM_sources
	${CMAKE_CURRENT_SOURCE_DIR}/kT.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/dT.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/ApiSystem.cpp
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



