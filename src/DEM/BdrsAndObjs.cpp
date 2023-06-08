#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "BdrsAndObjs.h"
#include <vector>
#include <string>
#include <limits>
#include <iostream>
#include <sstream>
#include <array>
#include <cmath>

#include <nvmath/helper_math.cuh>
#include <DEM/Defines.h>
#include <DEM/Structs.h>
#include <core/utils/ManagedAllocator.hpp>
#include <DEM/HostSideHelpers.hpp>

namespace py = pybind11;

// Defining PyBind11 inclusion code

PYBIND11_MODULE(BdrsAndObjs, obj) {
    py::class_<deme::DEMExternObj>(obj, "DEMExternObj")
        .def(py::init<>())
        .def("SetFamily", &deme::DEMExternObj::SetFamily, "Defines an object contact family number")
        .def("SetMass", &deme::DEMExternObj::SetMass, "Sets the mass of this object")
        .def("SetMOI", &deme::DEMExternObj::SetMOI, "Sets the MOI (in the principal frame)")
        .def("SetInitQuat", &deme::DEMExternObj::SetInitQuat,
             "Set the initial quaternion for this object (before simulation initializes).")
        .def("SetInitPos", &deme::DEMExternObj::SetInitPos,
             "Set the initial position for this object (before simulation initializes).")
        .def("AddPlane", &deme::DEMExternObj::AddPlane, "Add a plane with infinite size.")
        .def("AddPlate", &deme::DEMExternObj::AddPlate, "Add a plate with finite size.")
        .def("AddZCylinder", &deme::DEMExternObj::AddZCylinder, "Add a z-axis-aligned cylinder of infinite length")
        .def("AddCylinder", &deme::DEMExternObj::AddCylinder,
             "Add a cylinder of infinite length, which is along a user-specific axis")
        .def_readwrite("types", &deme::DEMExternObj::types)
        .def_readwrite("materials", &deme::DEMExternObj::materials)
        .def_readwrite("family_code", &deme::DEMExternObj::family_code)
        .def_readwrite("init_pos", &deme::DEMExternObj::init_pos)
        .def_readwrite("init_oriQ", &deme::DEMExternObj::init_oriQ)
        .def_readwrite("mass", &deme::DEMExternObj::mass)
        .def_readwrite("MOI", &deme::DEMExternObj::MOI)
        .def_readwrite("load_order", &deme::DEMExternObj::load_order)
        .def_readwrite("entity_params", &deme::DEMExternObj::entity_params);
}