#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Defines.h"
#include <limits>
#include <stdint.h>
#include <algorithm>
#include <cmath>

#include <DEM/VariableTypes.h>

namespace py = pybind11;

// Defining PyBind11 inclusion code

py::enum_<VERBOSITY>(obj, "VERBOSITY")
    .value("QUIET", VERBOSITY::QUIET)
    .value("ERROR", VERBOSITY::ERROR)
    .value("WARNING", VERBOSITY::WARNING)
    .value("INFO", VERBOSITY::INFO)
    .value("STEP_ANOMALY", VERBOSITY::STEP_ANOMALY)
    .value("STEP_METRIC", VERBOSITY::STEP_METRIC)
    .value("DEBUG", VERBOSITY::DEBUG)
    .value("STEP_DEBUG", VERBOSITY::STEP_DEBUG)
    .export_values();

py::enum_<TIME_INTEGRATOR>(obj, "TIME_INTEGRATOR")
    .value("FORWARD_EULER", TIME_INTEGRATOR::FORWARD_EULER)
    .value("CENTERED_DIFFERENCE", TIME_INTEGRATOR::CENTERED_DIFFERENCE)
    .value("EXTENDED_TAYLOR", TIME_INTEGRATOR::EXTENDED_TAYLOR)
    .value("CHUNG", TIME_INTEGRATOR::CHUNG)
    .export_values();

py::enum_<OWNER_TYPE>(obj, "OWNER_TYPE")
    .value("CLUMP", OWNER_TYPE::CLUMP)
    .value("ANALYTICAL", OWNER_TYPE::ANALYTICAL)
    .value("MESH", OWNER_TYPE::MESH)
    .export_values();

py::enum_<FORCE_MODEL>(obj, "FORCE_MODEL")
    .value("HERTZIAN", FORCE_MODEL::HERTZIAN)
    .value("HERTZIAN_FRICTIONLES", FORCE_MODEL::HERTZIAN_FRICTIONLES)
    .value("CUSTOM", FORCE_MODEL::CUSTOM)
    .export_values();

py::enum_<OUTPUT_CONTENT>(obj, "OUTPUT_CONTENT")
    .value("XYZ", OUTPUT_CONTENT::XYZ)
    .value("QUAT", OUTPUT_CONTENT::QUAT)
    .value("ABSV", OUTPUT_CONTENT::ABSV)
    .value("VEL", OUTPUT_CONTENT::VEL)
    .value("ANG_VEL", OUTPUT_CONTENT::ANG_VEL)
    .value("ABS_ACC", OUTPUT_CONTENT::ABS_ACC)
    .value("ACC", OUTPUT_CONTENT::ACC)
    .value("ANG_ACC", OUTPUT_CONTENT::ANG_ACC)
    .value("FAMILY", OUTPUT_CONTENT::FAMILY)
    .value("MAT", OUTPUT_CONTENT::MAT)
    .value("OWNER_WILDCARD", OUTPUT_CONTENT::OWNER_WILDCARD)
    .value("GEO_WILDCARD", OUTPUT_CONTENT::GEO_WILDCARD)
    .value("EXP_FACTOR", OUTPUT_CONTENT::EXP_FACTOR)
    .export_values();

py::enum_<SPATIAL_DIR>(obj, "SPATIAL_DIR")
    .value("X", OUTPUT_CONTENT::X)
    .value("Y", OUTPUT_CONTENT::Y)
    .value("Z", OUTPUT_CONTENT::Z)
    .value("NONE", OUTPUT_CONTENT::NONE)

py::enum_<CNT_OUTPUT_CONTENT>(obj, "CNT_OUTPUT_CONTENT")
    .value("CNT_TYPE", CNT_OUTPUT_CONTENT::CNT_TYPE)
    .value("FORCE", CNT_OUTPUT_CONTENT::FORCE)
    .value("POINT", CNT_OUTPUT_CONTENT::POINT)
    .value("COMPONENT", CNT_OUTPUT_CONTENT::COMPONENT)
    .value("NORMAL", CNT_OUTPUT_CONTENT::NORMAL)
    .value("TORQUE", CNT_OUTPUT_CONTENT::TORQUE)
    .value("CNT_WILDCARD", CNT_OUTPUT_CONTENT::CNT_WILDCARD)
    .value("OWNER", CNT_OUTPUT_CONTENT::OWNER)
    .value("GEO_ID", CNT_OUTPUT_CONTENT::GEO_ID)
    .value("NICKNAME", CNT_OUTPUT_CONTENT::NICKNAME)
    .export_values();
