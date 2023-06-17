#include <DEM/Defines.h>
#include <core/utils/ManagedAllocator.hpp>
#include <core/utils/ManagedMemory.hpp>
#include <core/utils/csv.hpp>
#include <core/utils/GpuError.h>
#include <core/utils/Timer.hpp>
#include <core/utils/RuntimeData.h>
#include "Struts.h"

#include <sstream>
#include <exception>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <unordered_map>
#include <nvmath/helper_math.cuh>
#include <DEM/HostSideHelpers.hpp>
#include <filesystem>
#include <cstring>
#include <cassert>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <memory>

namespace py = pybind11;

PYBIND11_MODULE(DEME, obj) {
    py::class_<deme::DEMMaterial>(obj, "DEMMaterial")
        .def(py::init<const std::unordered_map<std::string, float>&>())
        .def("__del__", &deme::DEMMaterial::~DEMMaterial,
             "Links the python __del__ function to DEMMaterial's destructor")
        .def_readwrite("mat_prop", &deme::DEMMaterial::mat_prop);

    py::class_<deme::DEMClumpTemplate>(obj, "DEMClumpTemplate")
        .def(py::init<>())
        .def("SetVolume", &deme::DEMClumpTemplate::SetVolume)
        .def("ReadComponentFromFile", &deme::DEMClumpTemplate::ReadComponentFromFile)
        .def("InformCentroidPrincipal", &deme::DEMClumpTemplate::InformCentroidPrincipal)
        .def("Move", &deme::DEMClumpTemplate::Move)
        .def("Scale", &deme::DEMClumpTemplate::Scale)
        .def("AssignName", &deme::DEMClumpTemplate::AssignName);

    py::class_<deme::DEMClumpBatch>(obj, "DEMClumpBatch")
        .def(py::init<size_t&>())
        .def("__del__", &deme::DEMClumpBatch::~DEMClumpBatch,
             "Links the python __del__ function to DEMClumpBatch's destructor")
        .def("SetTypes", &deme::DEMClumpBatch::SetTypes)
        .def("SetType", &deme::DEMClumpBatch::SetType)
        .def("SetPos", &deme::DEMClumpBatch::SetPos)
        .def("SetVel", &deme::DEMClumpBatch::SetVel)
        .def("SetAngVel", &deme::DEMClumpBatch::SetAngVel)
        .def("SetOriQ", &deme::DEMClumpBatch::SetOriQ)
        .def("SetFamilies", &deme::DEMClumpBatch::SetFamilies)
        .def("SetFamily", &deme::DEMClumpBatch::SetFamily)
        .def("SetExistingContacts", &deme::DEMClumpBatch::SetExistingContacts)
        .def("SetExistingContactWildcards", &deme::DEMClumpBatch::SetExistingContactWildcards)
        .def("AddExistingContactWildcard", &deme::DEMClumpBatch::AddExistingContactWildcard)
        .def("SetOwnerWildcards", &deme::DEMClumpBatch::SetOwnerWildcards)
        .def("AddOwnerWildcard", &deme::DEMClumpBatch::AddOwnerWildcard)
        .def("SetGeometryWildcards", &deme::DEMClumpBatch::SetGeometryWildcards)
        .def("AddGeometryWildcard", &deme::DEMClumpBatch::AddGeometryWildcard)
        .def("GetNumContacts", &deme::DEMClumpBatch::GetNumContacts);
}