# Copyright (c) 2021, SBEL GPU Development Team
# Copyright (c) 2021, University of Wisconsin - Madison
# All rights reserved.

message(STATUS "==== Building CUDA Kernels ====")

add_library(kernel OBJECT)

target_include_directories(
	kernel
	PUBLIC ${ProjectIncludeRoot}
	PUBLIC ${NVIDIAMathDir}
)

target_link_libraries(
	kernel
	PUBLIC CUB::CUB
)

# .h, .hpp
set(kernel_headers
	${CMAKE_CURRENT_SOURCE_DIR}/cudaKernels.cuh
)

# .c, .cpp, .cu
set(kernel_sources
	${CMAKE_CURRENT_SOURCE_DIR}/cudaKernels.cu
)

target_sources(
	kernel
	INTERFACE ${kernel_headers}
	PRIVATE ${kernel_sources}
)

set_target_properties(
	kernel PROPERTIES
	POSITION_INDEPENDENT_CODE True
)

