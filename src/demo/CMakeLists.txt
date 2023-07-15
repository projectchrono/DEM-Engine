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
		DEMdemo_Repose
		DEMdemo_Sieve
		DEMdemo_SingleSphereCollide
		DEMdemo_TestPack
		DEMdemo_RotatingDrum
		DEMdemo_Centrifuge
		DEMdemo_GameOfLife
		DEMdemo_BallDrop
		DEMdemo_Mixer
		DEMdemo_GRCPrep_Part1
		DEMdemo_GRCPrep_Part2
		DEMdemo_ConePenetration
		DEMdemo_Shake
		DEMdemo_WheelDP
		DEMdemo_WheelDPSimplified
		DEMdemo_WheelSlopeSlip
		DEMdemo_Indentation
		DEMdemo_Granular_Repose
		DEMdemo_Granular_Flow
		DEMdemo_Granular_FlowSpheresDrum
		DEMdemo_Granular_NonSpherical
		DEMdemo_Granular_WoodenSpheresHopper
		DEMdemo_Granular_PlasticSpheresDrum_sensDistr
		DEMdemo_Granular_PlasticSpheresDrum_sensInertia
		DEMdemo_Granular_PlasticSpheresHopper
		DEMdemo_Granular_PlasticCylindersHopper
		DEMdemo_Granular_WoodenSpheresDrum
		DEMdemo_Granular_WoodenSpheresDrum_sensDistr
		DEMdemo_Granular_WoodenCylinderDrum
		DEMdemo_Granular_WoodenCubeDrum
		DEMdemo_Granular_WoodenCubeHopper
		DEMdemo_Granular_FreeFalling
		DEMdemo_Granular_WoodenCylindersHopper
		DEMdemo_SolarSystem
		DEMdemo_Repose_Cylinder
		DEMdemo_Electrostatic
		DEMdemo_FlexibleMesh
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

