# Copyright (c) 2021, SBEL GPU Development Team
# Copyright (c) 2021, University of Wisconsin - Madison
# All rights reserved.

message(STATUS "==== Configuring Core Components ====")

set (core_message "[core]")
message(STATUS "${core_message} Generating API version header...")

configure_file(
	ApiVersion.h.in ApiVersion.h
	USE_SOURCE_PERMISSIONS
	@ONLY
)

if(TargetCXXStandard STREQUAL STD_CXX11)
	set(core_cxxver 11)
elseif(TargetCXXStandard STREQUAL STD_CXX14)
	set(core_cxxver 14)
elseif(TargetCXXStandard STREQUAL STD_CXX17)
	set(core_cxxver 17)
elseif(TargetCXXStandard STREQUAL STD_CXX20)
	set(core_cxxver 20)
else()
	# TODO: Perform C++ version autodetection based on the compiler ID
endif()

add_library(core OBJECT)

target_include_directories(
	core
	PUBLIC ${ProjectIncludeRoot}
	PUBLIC ${NVIDIAJitifyPath}
	PUBLIC ${NVIDIAMathDir}
)

target_link_libraries(
	core
	PUBLIC CUB::CUB
)

set(core_headers
	${CMAKE_CURRENT_SOURCE_DIR}/ApiVersion.h
	${CMAKE_CURRENT_SOURCE_DIR}/utils/ManagedAllocator.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/JitHelper.h
	${CMAKE_CURRENT_SOURCE_DIR}/utils/ThreadManager.h
	${CMAKE_CURRENT_SOURCE_DIR}/utils/GpuError.h
	${CMAKE_CURRENT_SOURCE_DIR}/utils/GpuManager.h
	${CMAKE_CURRENT_SOURCE_DIR}/utils/Macros.h
	${CMAKE_CURRENT_SOURCE_DIR}/utils/chpf/particle_writer.hpp
	${NVIDIAMathDir}/helper_math.h
)

set(core_sources
	${CMAKE_CURRENT_SOURCE_DIR}/DebugInfo.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/GpuManager.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/JitHelper.cpp
)

target_sources(
	core
	INTERFACE ${core_headers}
	PRIVATE ${core_sources}
)

set_target_properties(
	core PROPERTIES 
	POSITION_INDEPENDENT_CODE True
	CXX_STANDARD ${core_cxxver}
)
