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
		DEMdemo_Plow
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
		DEMdemo_Granular_WoodenSphereHopper
		DEMdemo_Granular_PlasticSphereDrum
		DEMdemo_Granular_PlasticSphereHopper
		DEMdemo_Granular_PlasticCylinderDrum
		DEMdemo_Granular_PlasticCylinderHopper
		DEMdemo_Granular_WoodenSphereDrum
		DEMdemo_Granular_WoodenCylinderDrum
		DEMdemo_Granular_WoodenCubeDrum
		DEMdemo_Granular_WoodenCubeHopper
		DEMdemo_Granular_FreeFalling
		DEMdemo_Granular_WoodenCylinderHopper
		DEMdemo_Granular_WoodenCylinder_SphereHopper
		DEMdemo_PlasticSphere_CylinderHopper
		DEMdemo_PlasticCylinder_SphereHopper
		DEMdemo_SolarSystem
		DEMdemo_Repose_Cylinder
		DEMdemo_Electrostatic
		DEMdemo_FlexibleMesh
		DEMdemo_Fracture
		DEMdemo_FractureBox
		DEMdemo_FractureBeam
		DEMdemo_FractureCylinderSteel
		test_SPH_boundary
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

