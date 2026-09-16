# Documentation is intentionally a standalone target: it parses headers and
# must not import the Python extension or compile CUDA code.
find_package(Doxygen REQUIRED)
find_program(SPHINX_BUILD_EXECUTABLE NAMES sphinx-build REQUIRED)

add_custom_target(
	docs
	COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/_build/doxygen
	COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
	COMMAND ${SPHINX_BUILD_EXECUTABLE} -W --keep-going -b html
	        ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/_build/html
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	COMMENT "Building DEM-Engine Doxygen and Sphinx documentation"
	VERBATIM
)
