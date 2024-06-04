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
		DEMdemo_Repose2D
		DEMdemo_Plow
		DEMdemo_Sieve
		DEMdemo_SingleSphereCollide
		DEMdemo_TestPack
		DEMdemo_RotatingDrum
		DEMdemo_Centrifuge
		DEMdemo_GameOfLife
		DEMdemo_BallDrop
		DEMdemo_BallDrop2D
		DEMdemo_Mixer
		DEMdemo_GRCPrep_Part1
		DEMdemo_GRCPrep_Part2
		DEMdemo_GRCPrep_Part3
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
		DEMdemo_Hopper_Sphere_Cylinder
		DEMdemo_Fracture_Box
		DEMdemo_FractureBeam
		DEMdemo_Indentation_2
		DEMdemo_Force2D
		DEMdemo_Force3D
		DEMdemo_FractureCylinderSteel
		DEMdemo_FillingGeo
		DEMdemo_Force3DSingle
		DEMdemo_ContactChain
		gael
		test
		)

# ------------------------------------------------------------------------------
# Add all executables
# ------------------------------------------------------------------------------

message(STATUS "Demo programs for DEM solver...")

FOREACH(PROGRAM ${DEMOS})
		
		message(STATUS "...add ${PROGRAM}")

		add_executable(${PROGRAM}  "${PROGRAM}.cpp")

		set_target_properties(
			${PROGRAM} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${DEME_INSTALL_DEMO}"
		)

		if(WIN32)
			# set_property(TARGET ${PROGRAM} 
			# 	PROPERTY VS_DEBUGGER_ENVIRONMENT "PATH=$<TARGET_FILE_DIR:DEMERuntimeDataHelper_install>;%PATH%"
			# )
			# set_property(TARGET ${PROGRAM} 
			# 	PROPERTY VS_DEBUGGER_ENVIRONMENT "PATH=$<TARGET_FILE_DIR:DEMERuntimeDataHelper>;%PATH%"
			# )
			add_custom_command(TARGET ${PROGRAM} POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       "$<TARGET_FILE_DIR:DEMERuntimeDataHelper_install>/DEMERuntimeDataHelper_install.dll"
					   "$<TARGET_FILE_DIR:DEMERuntimeDataHelper_install>/DEMERuntimeDataHelper.dll"
                       "$<TARGET_FILE_DIR:${PROGRAM}>")
		endif()
		
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

		# install(TARGETS ${PROGRAM} DESTINATION ${DEME_INSTALL_DEMO})

ENDFOREACH(PROGRAM)

