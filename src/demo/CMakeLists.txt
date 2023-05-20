# ------------------------------------------------------------------------------
# Additional include paths and libraries
# ------------------------------------------------------------------------------

# INCLUDE_DIRECTORIES(${ProjectIncludeRoot})

SET(LIBRARIES
	simulator_multi_gpu
)

# ------------------------------------------------------------------------------
# List of all executables
# ------------------------------------------------------------------------------

SET(DEMOS
		DEMdemo_Sieve
		DEMdemo_SingleSphereCollide
		DEMdemo_TestPack
		DEMdemo_RotatingDrum
		DEMdemo_Centrifuge
		DEMdemo_GameOfLife
		DEMdemo_BallDrop
		DEMdemo_Mixer
		DEMdemo_ConePenetration
		DEMdemo_Shake
		DEMdemo_WheelDP
		DEMdemo_Meshed_WheelDP_SlopeSlip_Moon
		DEMdemo_Meshed_WheelDP_SlopeSlip_Earth
		DEMdemo_Meshed_WheelDP_SlopeSlip_Earth_AllSph
		DEMdemo_Meshed_WheelDP_SlopeSlip_Moon_AllSph
		DEMdemo_Indentation
		DEMdemo_Repose_GRC
		DEMdemo_GRCPrep_Part1
		DEMdemo_GRCPrep_Part2
		DEMdemo_GRCPrep_Part3
		DEMdemo_GRCPrep_Part1_Large
		DEMdemo_GRCPrep_Part2_Large
		DEMdemo_GRCPrep_Part3_Large
		DEMdemo_GRCPrep_Part1_Small
		DEMdemo_GRCPrep_Part2_Small
		DEMdemo_GRCPrep_Part3_Small
)

# ------------------------------------------------------------------------------
# Add all executables
# ------------------------------------------------------------------------------

message(STATUS "Demo programs for DEM solver...")

FOREACH(PROGRAM ${DEMOS})
		
		message(STATUS "...add ${PROGRAM}")

		add_executable(${PROGRAM}  "${PROGRAM}.cpp")
		
		source_group("" FILES "${PROGRAM}.cpp")
		
		target_link_libraries(${PROGRAM} 
			PUBLIC ${LIBRARIES}
			PUBLIC ${EXTERNAL_LIBRARIES}
		)
		
		add_dependencies(${PROGRAM} ${LIBRARIES})

		set_target_properties(
			${PROGRAM} PROPERTIES
			CXX_STANDARD ${CXXSTD_SUPPORTED}
		)

ENDFOREACH(PROGRAM)

