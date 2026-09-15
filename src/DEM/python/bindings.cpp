#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <vector>
#include <string>
#include <limits>
#include <iostream>
#include <sstream>
#include <array>
#include <cstdint>
#include <cmath>
#include <memory>

#include "core/utils/RuntimeData.h"
#include "PythonCudaIncludes.hpp"
#include "core/utils/DataMigrationHelper.hpp"
#include "core/utils/DEMEPaths.h"
#include "core/ApiVersion.h"
#include "DEM/utils/HostSideHelpers.hpp"
#include "DEM/utils/Samplers.hpp"
#include "DEM/Defines.h"
#include "DEM/Structs.h"
#include "DEM/API.h"
#include "DEM/AuxClasses.h"
#include "DEM/VariableTypes.h"
#ifdef DEME_HAS_VISUALIZER
    #include "DEM/utils/DEMVisualizer.h"
#endif

namespace py = pybind11;

// Custom type caster for float3 and float4
namespace pybind11 {
namespace detail {
template <>
struct type_caster<float3> {
  public:
    PYBIND11_TYPE_CASTER(float3, _("float3"));

    // Python -> C++
    bool load(handle src, bool) {
        if (!isinstance<py::sequence>(src))
            return false;
        auto seq = reinterpret_borrow<py::sequence>(src);
        if (seq.size() != 3)
            return false;
        value.x = seq[0].cast<float>();
        value.y = seq[1].cast<float>();
        value.z = seq[2].cast<float>();
        return true;
    }

    // C++ -> Python
    static handle cast(const float3& src, return_value_policy, handle) {
        return py::make_tuple(src.x, src.y, src.z).release();
    }
};

template <>
struct type_caster<float4> {
  public:
    PYBIND11_TYPE_CASTER(float4, _("float4"));

    // Python -> C++
    bool load(handle src, bool) {
        if (!isinstance<py::sequence>(src))
            return false;
        auto seq = reinterpret_borrow<py::sequence>(src);
        if (seq.size() != 4)
            return false;
        value.x = seq[0].cast<float>();
        value.y = seq[1].cast<float>();
        value.z = seq[2].cast<float>();
        value.w = seq[3].cast<float>();
        return true;
    }

    // C++ -> Python
    static handle cast(const float4& src, return_value_policy, handle) {
        return py::make_tuple(src.x, src.y, src.z, src.w).release();
    }
};
}  // namespace detail
}  // namespace pybind11

PYBIND11_MODULE(_deme, obj) {
    // Called once by the package bootstrap, before user code can create a solver. Keep this state
    // private to the extension so importing Python cannot change a separately launched C++ program.
    obj.def("_set_cuda_include_paths", [](const std::vector<std::string>& paths) {
        auto& configured = deme::python::CudaIncludePaths();
        static bool initialized = false;
        const std::vector<std::filesystem::path> requested(paths.begin(), paths.end());
        if (initialized && configured != requested) {
            throw py::value_error("Python CUDA include paths have already been configured");
        }
        if (!initialized) {
            configured = requested;
            initialized = true;
        }
    });

    // Report the same version as the CMake project so Python and C++ callers
    // cannot observe conflicting release versions.
    obj.attr("__version__") = std::to_string(DEME_VERSION_MAJOR) + "." + std::to_string(DEME_VERSION_MINOR) + "." +
                              std::to_string(DEME_VERSION_PATCH);
    // Obtaining the location of python's site-packages dynamically and setting it
    // as path prefix
    try {
        py::module _site = py::module_::import("site");
        std::string loc;

        // Try getsitepackages() first (works in most environments)
        if (py::hasattr(_site, "getsitepackages")) {
            py::object site_packages = _site.attr("getsitepackages")();
            if (py::isinstance<py::list>(site_packages) && py::len(site_packages) > 0) {
                loc = site_packages.cast<py::list>()[0].cast<std::string>();
            }
        }

        // Fallback to USER_SITE if getsitepackages fails
        if (loc.empty() && py::hasattr(_site, "USER_SITE")) {
            loc = _site.attr("USER_SITE").cast<std::string>();
        }

        // Last resort: use sys.prefix
        if (loc.empty()) {
            py::module sys = py::module_::import("sys");
            std::string prefix = sys.attr("prefix").cast<std::string>();
            loc = prefix + "/lib/python" + sys.attr("version_info").attr("major").cast<std::string>() + "." +
                  sys.attr("version_info").attr("minor").cast<std::string>() + "/site-packages";
        }

        // Setting path prefix
        std::filesystem::path path = loc;
        DEMERuntimeDataHelper::SetPathPrefix(path);
    } catch (const py::error_already_set& e) {
        // If all else fails, don't set the prefix and let the system use default paths
        py::print(
            "Warning: Could not determine site-packages location for data files, using default paths. "
            "Data file loading may fail. Set DEME_DATA_PATH environment variable if you encounter issues.");
    }

    deme::SetDEMEDataPath();
    deme::SetDEMEKernelPath();
    deme::SetDEMEIncludePath();

    // To define methods independent of a class, use obj.def() syntax to wrap them!
    obj.def("FrameTransformGlobalToLocal", &deme::FrameTransformGlobalToLocal,
            "Translating the inverse of the provided vec then applying a local inverse rotation of the provided rot_Q, "
            "then return the result.");
    obj.def("FrameTransformLocalToGlobal", &deme::FrameTransformLocalToGlobal,
            "Apply a local rotation then a translation, then return the result.");
    obj.def("GetDEMEDataFile", &deme::GetDEMEDataFile, "Resolve a path relative to DEME's installed data directory.",
            py::arg("relative_path"));
    obj.def(
        "DEMBoxGridSampler",
        static_cast<std::vector<float3> (*)(const std::vector<float>&, const std::vector<float>&, float, float, float)>(
            &deme::DEMBoxGridSampler),
        py::arg("BoxCenter"), py::arg("HalfDims"), py::arg("GridSizeX"), py::arg("GridSizeY") = -1.0,
        py::arg("GridSizeZ") = -1.0,
        "Return regularly spaced points inside a box. Negative Y or Z spacing reuses the X spacing.");
    obj.def("DEMBoxHCPSampler",
            static_cast<std::vector<float3> (*)(const std::vector<float>&, const std::vector<float>&, float)>(
                &deme::DEMBoxHCPSampler),
            "Return hexagonally close-packed points inside a box.", py::arg("BoxCenter"), py::arg("HalfDims"),
            py::arg("separation"));
    obj.def("DEMCylSurfSampler",
            static_cast<std::vector<std::vector<float>> (*)(const std::vector<float>&, const std::vector<float>&, float,
                                                            float, float, float)>(&deme::DEMCylSurfSampler),
            py::arg("CylCenter"), py::arg("CylAxis"), py::arg("CylRad"), py::arg("CylHeight"), py::arg("ParticleRad"),
            py::arg("spacing") = 1.2,
            "Return points on a cylindrical surface. ``spacing`` scales the nominal particle-diameter separation.");

    obj.attr("PI") = py::float_(M_PI);
    // Export both the precise name and its legacy mesh-oriented spelling. They deliberately carry the same value.
    obj.attr("SPHERE_TRIANGLE_CONTACT") = py::int_(static_cast<unsigned int>(deme::SPHERE_TRIANGLE_CONTACT));
    obj.attr("SPHERE_MESH_CONTACT") = py::int_(static_cast<unsigned int>(deme::SPHERE_MESH_CONTACT));

    py::class_<DEMERuntimeDataHelper>(obj, "DEMERuntimeDataHelper")
        .def(py::init<>())
        .def_static("SetPathPrefix", &DEMERuntimeDataHelper::SetPathPrefix);

    py::class_<deme::PDSampler>(obj, "PDSampler",
                                "Poisson-disk volumetric sampler enforcing a minimum point separation.")
        .def(py::init<float>(), py::arg("separation"), "Create a sampler with the requested minimum separation.")
        .def("SetSeparation", &deme::PDSampler::SetSeparation, "Set the minimum separation used by subsequent samples.",
             py::arg("separation"))
        .def("SetRandomEngineSeed", &deme::PDSampler::SetRandomEngineSeed,
             "Seed the random engine so Poisson-disk samples can be reproduced.", py::arg("seed"))
        .def("SampleBox",
             static_cast<std::vector<std::vector<float>> (deme::PDSampler::*)(
                 const std::vector<float>& center, const std::vector<float>& halfDim)>(&deme::PDSampler::SampleBox),
             "Return points sampled from the specified box volume.", py::arg("center"), py::arg("halfDim"))

        .def("SampleSphere",
             static_cast<std::vector<std::vector<float>> (deme::PDSampler::*)(const std::vector<float>&, float)>(
                 &deme::PDSampler::SampleSphere),
             "Return points sampled from the specified spherical volume.", py::arg("center"), py::arg("radius"))

        .def("SampleCylinderX",
             static_cast<std::vector<std::vector<float>> (deme::PDSampler::*)(const std::vector<float>&, float, float)>(
                 &deme::PDSampler::SampleCylinderX),
             "Return points sampled from an X-aligned cylindrical volume.", py::arg("center"), py::arg("radius"),
             py::arg("halfHeight"))

        .def("SampleCylinderY",
             static_cast<std::vector<std::vector<float>> (deme::PDSampler::*)(const std::vector<float>&, float, float)>(
                 &deme::PDSampler::SampleCylinderY),
             "Return points sampled from a Y-aligned cylindrical volume.", py::arg("center"), py::arg("radius"),
             py::arg("halfHeight"))

        .def("SampleCylinderZ",
             static_cast<std::vector<std::vector<float>> (deme::PDSampler::*)(const std::vector<float>&, float, float)>(
                 &deme::PDSampler::SampleCylinderZ),
             "Return points sampled from a Z-aligned cylindrical volume.", py::arg("center"), py::arg("radius"),
             py::arg("halfHeight"))

        .def("GetSeparation", &deme::Sampler::GetSeparation, "Return the current minimum separation.");

    py::class_<deme::GridSampler>(obj, "GridSampler", "Regular-grid volumetric point sampler.")
        .def(py::init<float>(), py::arg("separation"), "Create a regular-grid sampler with uniform point spacing.")
        .def("SetSeparation", &deme::GridSampler::SetSeparation, "Set the point spacing used by subsequent samples.",
             py::arg("separation"))
        .def("SampleBox",
             static_cast<std::vector<std::vector<float>> (deme::GridSampler::*)(
                 const std::vector<float>& center, const std::vector<float>& halfDim)>(&deme::GridSampler::SampleBox),
             "Return points sampled from the specified box volume.", py::arg("center"), py::arg("halfDim"))

        .def("SampleSphere",
             static_cast<std::vector<std::vector<float>> (deme::GridSampler::*)(const std::vector<float>&, float)>(
                 &deme::GridSampler::SampleSphere),
             "Return points sampled from the specified spherical volume.", py::arg("center"), py::arg("radius"))

        .def("SampleCylinderX",
             static_cast<std::vector<std::vector<float>> (deme::GridSampler::*)(
                 const std::vector<float>&, float, float)>(&deme::GridSampler::SampleCylinderX),
             "Return points sampled from an X-aligned cylindrical volume.", py::arg("center"), py::arg("radius"),
             py::arg("halfHeight"))

        .def("SampleCylinderY",
             static_cast<std::vector<std::vector<float>> (deme::GridSampler::*)(
                 const std::vector<float>&, float, float)>(&deme::GridSampler::SampleCylinderY),
             "Return points sampled from a Y-aligned cylindrical volume.", py::arg("center"), py::arg("radius"),
             py::arg("halfHeight"))

        .def("SampleCylinderZ",
             static_cast<std::vector<std::vector<float>> (deme::GridSampler::*)(
                 const std::vector<float>&, float, float)>(&deme::GridSampler::SampleCylinderZ),
             "Return points sampled from a Z-aligned cylindrical volume.", py::arg("center"), py::arg("radius"),
             py::arg("halfHeight"))

        .def("GetSeparation", &deme::Sampler::GetSeparation, "Return the current grid spacing.");

    py::class_<deme::HCPSampler>(obj, "HCPSampler", "Hexagonally close-packed volumetric point sampler.")
        .def(py::init<float>(), py::arg("separation"), "Create an HCP sampler with the requested point separation.")
        .def("SampleBox",
             static_cast<std::vector<std::vector<float>> (deme::HCPSampler::*)(
                 const std::vector<float>& center, const std::vector<float>& halfDim)>(&deme::HCPSampler::SampleBox),
             "Return points sampled from the specified box volume.", py::arg("center"), py::arg("halfDim"))

        .def("SetSeparation", &deme::HCPSampler::SetSeparation, "Set the point separation used by subsequent samples.",
             py::arg("separation"))
        .def("SampleSphere",
             static_cast<std::vector<std::vector<float>> (deme::HCPSampler::*)(const std::vector<float>&, float)>(
                 &deme::HCPSampler::SampleSphere),
             "Return points sampled from the specified spherical volume.", py::arg("center"), py::arg("radius"))

        .def(
            "SampleCylinderX",
            static_cast<std::vector<std::vector<float>> (deme::HCPSampler::*)(const std::vector<float>&, float, float)>(
                &deme::HCPSampler::SampleCylinderX),
            "Return points sampled from an X-aligned cylindrical volume.", py::arg("center"), py::arg("radius"),
            py::arg("halfHeight"))

        .def(
            "SampleCylinderY",
            static_cast<std::vector<std::vector<float>> (deme::HCPSampler::*)(const std::vector<float>&, float, float)>(
                &deme::HCPSampler::SampleCylinderY),
            "Return points sampled from a Y-aligned cylindrical volume.", py::arg("center"), py::arg("radius"),
            py::arg("halfHeight"))

        .def(
            "SampleCylinderZ",
            static_cast<std::vector<std::vector<float>> (deme::HCPSampler::*)(const std::vector<float>&, float, float)>(
                &deme::HCPSampler::SampleCylinderZ),
            "Return points sampled from a Z-aligned cylindrical volume.", py::arg("center"), py::arg("radius"),
            py::arg("halfHeight"))

        .def("GetSeparation", &deme::HCPSampler::GetSeparation, "Return the current point separation.");

    // Have to forward declare this templated method so pybind can see it
    std::vector<float>& (deme::DataContainer::*get_float_dc)(const std::string&) = &deme::DataContainer::Get<float>;
    py::class_<deme::DataContainer, std::shared_ptr<deme::DataContainer>>(
        obj, "DataContainer", "String-keyed container used to return named arrays from DEME.")
        .def(py::init<>(), "Create an empty data container.")
        .def("Get", get_float_dc, py::return_value_policy::reference_internal,
             "Return the stored float array named ``key`` by reference. This is primarily used for wildcard output.",
             py::arg("key"));

    py::class_<deme::ContactInfoContainer, deme::DataContainer, std::shared_ptr<deme::ContactInfoContainer>>(
        obj, "ContactInfoContainer",
        "Named arrays describing contacts selected by ``DEMSolver.SetContactOutputContent``.")
        .def(py::init<unsigned int&, const std::vector<std::pair<std::string, std::string>>&>())
        .def("GetContactType", &deme::ContactInfoContainer::GetContactType,
             "Get contact type as strings from the container.")
        .def("GetPoint", &deme::ContactInfoContainer::GetPoint, "Get contact points as vectors from the container.")
        .def("GetAOwner", &deme::ContactInfoContainer::GetAOwner, "Get AOwner number from the container.")
        .def("GetBOwner", &deme::ContactInfoContainer::GetBOwner, "Get BOwner number from the container.")
        .def("GetAGeo", &deme::ContactInfoContainer::GetAGeo, "Get AGeo number from the container.")
        .def("GetBGeo", &deme::ContactInfoContainer::GetBGeo, "Get BGeo number from the container.")
        .def("GetAOwnerFamily", &deme::ContactInfoContainer::GetAOwnerFamily,
             "Get AOwnerFamily number from the container.")
        .def("GetBOwnerFamily", &deme::ContactInfoContainer::GetBOwnerFamily,
             "Get BOwnerFamily number from the container.")
        .def("GetForce", &deme::ContactInfoContainer::GetForce, "Get force as vectors from the container.")
        .def("GetTorque", &deme::ContactInfoContainer::GetTorque, "Get torque as vectors from the container.")
        .def("GetNormal", &deme::ContactInfoContainer::GetNormal, "Get contact normal as vectors from the container.");

    py::class_<deme::DEMInspector, std::shared_ptr<deme::DEMInspector>>(
        obj, "DEMInspector", "Evaluate a configured aggregate or spatial quantity from initialized simulation state.")
        .def(py::init<deme::DEMSolver*, deme::DEMDynamicThread*, const std::string&>())
        .def("GetValue", &deme::DEMInspector::GetValue,
             "Synchronize as needed and return the inspector's current scalar value.");

    py::class_<deme::DEMInitializer, std::shared_ptr<deme::DEMInitializer>>(obj, "DEMInitializer").def(py::init<>());

    py::class_<deme::DEMTrackedObj, deme::DEMInitializer, std::shared_ptr<deme::DEMTrackedObj>>(obj, "DEMTrackedObj")
        .def(py::init<deme::DEMTrackedObj>());

    py::class_<deme::DEMTracker, std::shared_ptr<deme::DEMTracker>>(obj, "Tracker",
                                                                    R"doc(
Access and modify owners associated with a tracked batch or object.

Create trackers with ``DEMSolver.Track`` during setup. Most state queries and
updates require the solver to be initialized and synchronized. ``offset``
selects an owner within the tracked collection, starting at zero. Keep the
parent solver alive while using a tracker.

Fixed-size CUDA exchange methods validate ranges, capacities, and pointer metadata by default. Pass
``validate=False`` only when those preconditions are guaranteed and avoiding validation overhead matters. Required
device routing and documented transformations, including quaternion normalization, still apply.
)doc")
        .def(py::init<deme::DEMSolver*>(), py::arg("solver"),
             "Create a tracker associated with ``solver``. Prefer ``DEMSolver.Track`` so the tracked owner range is "
             "configured correctly.")
        .def("GetOwnerID", &deme::DEMTracker::GetOwnerID,
             "Return the simulation-wide owner ID at ``offset`` within this tracker.", py::arg("offset") = 0)
        .def("GetOwnerIDs", &deme::DEMTracker::GetOwnerIDs,
             "Return simulation-wide owner IDs for every owner covered by this tracker.")

        .def("Pos", &deme::DEMTracker::GetPos,
             "Return the global-frame center-of-mass position of the tracked owner at ``offset``.",
             py::arg("offset") = 0)
        .def("Positions", &deme::DEMTracker::GetPositions,
             "Return global-frame center-of-mass positions for all owners covered by this tracker.")

        .def("AngVelLocal", &deme::DEMTracker::GetAngVelLocal,
             "Return the angular velocity of the owner at ``offset`` in its local principal-axis frame. Rotate it by "
             "``OriQ(offset)`` to obtain the global-frame value.",
             py::arg("offset") = 0)
        .def("AngularVelocitiesLocal", &deme::DEMTracker::GetAngularVelocitiesLocal,
             "Return local principal-axis-frame angular velocities for all tracked owners.")

        .def("AngVelGlobal", &deme::DEMTracker::GetAngVelGlobal,
             "Return the global-frame angular velocity of the owner at ``offset``.", py::arg("offset") = 0)
        .def("AngularVelocitiesGlobal", &deme::DEMTracker::GetAngularVelocitiesGlobal,
             "Return global-frame angular velocities for all tracked owners.")

        .def("Vel", &deme::DEMTracker::GetVel, "Return the global-frame linear velocity of the owner at ``offset``.",
             py::arg("offset") = 0)
        .def("Velocities", &deme::DEMTracker::GetVelocities,
             "Return global-frame linear velocities for all tracked owners.")

        .def("OriQ", &deme::DEMTracker::GetOriQ,
             "Return the quaternion rotating the owner-local frame to the global frame. Python ordering is "
             "``(x, y, z, w)`` (Chrono ``(e1, e2, e3, e0)``).",
             py::arg("offset") = 0)
        .def("OrientationQuaternions", &deme::DEMTracker::GetOrientationQuaternions,
             "Return local-to-global orientation quaternions for all tracked owners. Each quaternion uses Python "
             "ordering ``(x, y, z, w)``.")

        .def("GetFamily", &deme::DEMTracker::GetFamily, "Get the family number of the tracked object.",
             py::arg("offset") = 0)
        .def("GetFamilies", &deme::DEMTracker::GetFamilies, "Get the family numbers of all the tracked object.")

        .def("MOI", &deme::DEMTracker::GetMOI,
             "Get the moment of inertia (in principal axis frame) of the tracked object.", py::arg("offset") = 0)
        .def("MOIs", &deme::DEMTracker::GetMOIs,
             "Get the moment of inertia (in principal axis frame) of all the tracked objects.")

        .def("Mass", &deme::DEMTracker::Mass, "Get the mass of the tracked object.", py::arg("offset") = 0)
        .def("Masses", &deme::DEMTracker::Masses, "Get the masses of all the tracked objects.")

        .def("GetContactClumps", &deme::DEMTracker::GetContactClumps,
             "Get the clumps that are in contact with this tracked owner as a vector.", py::arg("offset") = 0)

        .def("ContactAcc", &deme::DEMTracker::GetContactAcc,
             "Get the a portion of the acceleration of this tracked object, that is the result of its contact with "
             "other simulation entities. In most cases, this means excluding the gravitational acceleration. The "
             "acceleration is in global frame.",
             py::arg("offset") = 0)
        .def("ContactAccelerations", &deme::DEMTracker::GetContactAccelerations,
             "Get the acceleration experienced by all objects tracked by this tracker, that is the result of their "
             "contact with other simulation entities. The acceleration is in global frame. In most cases, this means "
             "excluding the gravitational acceleration. The acceleration is in global frame.")

        .def("ContactAngAccLocal", &deme::DEMTracker::GetContactAngAccLocal,
             "Get the a portion of the angular acceleration of this tracked object, that is the result of its contact "
             "with other simulation entities. The acceleration is in this object's local frame.",
             py::arg("offset") = 0)
        .def("ContactAngularAccelerationsLocal", &deme::DEMTracker::GetContactAngularAccelerationsLocal,
             "Get the angular acceleration experienced by all objects tracked by this tracker, that is the result of "
             "their contact with other simulation entities. The acceleration is in this object's local frame.")

        .def("ContactAngAccGlobal", &deme::DEMTracker::GetContactAngAccGlobal,
             "Get the a portion of the angular acceleration of this tracked object, that is the result of its contact "
             "with other simulation entities. The acceleration is in this object's global frame.",
             py::arg("offset") = 0)
        .def("ContactAngularAccelerationsGlobal", &deme::DEMTracker::GetContactAngularAccelerationsGlobal,
             "Get the angular acceleration experienced by all objects tracked by this tracker, that is the result of "
             "their contact with other simulation entities. The acceleration is in this object's global frame.")

        .def("GetOwnerWildcardValue",
             static_cast<float (deme::DEMTracker::*)(const std::string&, size_t)>(
                 &deme::DEMTracker::GetOwnerWildcardValue),
             "Get the owner's wildcard value.", py::arg("name"), py::arg("offset") = 0)
        .def("GetOwnerWildcardValues",
             static_cast<std::vector<float> (deme::DEMTracker::*)(const std::string&)>(
                 &deme::DEMTracker::GetOwnerWildcardValues),
             "Get the owner wildcard values for all the owners entities tracked by this tracker.", py::arg("name"))
        .def(
            "PositionsToDevice",
            [](deme::DEMTracker& tracker, std::uintptr_t pointer, size_t capacity, int device, bool validate) {
                tracker.PositionsToDevice(reinterpret_cast<float3*>(pointer), capacity, device, validate);
            },
            R"doc(
Synchronously copy all tracked global positions to caller-owned CUDA memory.

``pointer`` is an integer address to writable, CUDA-accessible ``float3``
storage on logical CUDA ``device``. ``capacity`` is measured in ``float3``
elements and must cover every owner in this tracker. The caller owns the
allocation and must keep it alive through the call.
)doc",
            py::arg("pointer"), py::arg("capacity"), py::arg("device"), py::arg("validate") = true)
        .def(
            "SetPositionsFromDevice",
            [](deme::DEMTracker& tracker, std::uintptr_t pointer, int device, bool validate) {
                tracker.SetPositionsFromDevice(reinterpret_cast<const float3*>(pointer), device, validate);
            },
            "Synchronously set every tracked global position from CUDA ``float3`` storage. The buffer must contain "
            "one element per tracked owner; remote input is copied to the dynamic-worker device before unpacking.",
            py::arg("pointer"), py::arg("device"), py::arg("validate") = true)
        .def(
            "VelocitiesToDevice",
            [](deme::DEMTracker& tracker, std::uintptr_t pointer, size_t capacity, int device, bool validate) {
                tracker.VelocitiesToDevice(reinterpret_cast<float3*>(pointer), capacity, device, validate);
            },
            "Synchronously copy all tracked global linear velocities to caller-owned CUDA ``float3`` storage. "
            "``capacity`` is in elements; ``device`` is the allocation's logical CUDA device.",
            py::arg("pointer"), py::arg("capacity"), py::arg("device"), py::arg("validate") = true)
        .def(
            "SetVelocitiesFromDevice",
            [](deme::DEMTracker& tracker, std::uintptr_t pointer, int device, bool validate) {
                tracker.SetVelocitiesFromDevice(reinterpret_cast<const float3*>(pointer), device, validate);
            },
            "Synchronously set every tracked global linear velocity from CUDA ``float3`` storage. The buffer must "
            "contain one element per tracked owner; remote input is copied to the dynamic-worker device before "
            "unpacking.",
            py::arg("pointer"), py::arg("device"), py::arg("validate") = true)
        .def(
            "AngularVelocitiesLocalToDevice",
            [](deme::DEMTracker& tracker, std::uintptr_t pointer, size_t capacity, int device, bool validate) {
                tracker.AngularVelocitiesLocalToDevice(reinterpret_cast<float3*>(pointer), capacity, device, validate);
            },
            "Synchronously copy all tracked local-frame angular velocities to caller-owned CUDA ``float3`` storage. "
            "``capacity`` is in elements; ``device`` is the allocation's logical CUDA device.",
            py::arg("pointer"), py::arg("capacity"), py::arg("device"), py::arg("validate") = true)
        .def(
            "SetAngularVelocitiesFromDevice",
            [](deme::DEMTracker& tracker, std::uintptr_t pointer, int device, bool validate) {
                tracker.SetAngularVelocitiesFromDevice(reinterpret_cast<const float3*>(pointer), device, validate);
            },
            "Synchronously set every tracked local-frame angular velocity from CUDA ``float3`` storage. The buffer "
            "must contain one element per tracked owner; remote input is copied to the dynamic-worker device before "
            "unpacking.",
            py::arg("pointer"), py::arg("device"), py::arg("validate") = true)
        .def(
            "AngularVelocitiesGlobalToDevice",
            [](deme::DEMTracker& tracker, std::uintptr_t pointer, size_t capacity, int device, bool validate) {
                tracker.AngularVelocitiesGlobalToDevice(reinterpret_cast<float3*>(pointer), capacity, device, validate);
            },
            "Synchronously copy all tracked global-frame angular velocities to caller-owned CUDA ``float3`` storage. "
            "``capacity`` is in elements; ``device`` is the allocation's logical CUDA device.",
            py::arg("pointer"), py::arg("capacity"), py::arg("device"), py::arg("validate") = true)
        .def(
            "SetAngularVelocitiesGlobalFromDevice",
            [](deme::DEMTracker& tracker, std::uintptr_t pointer, int device, bool validate) {
                tracker.SetAngularVelocitiesGlobalFromDevice(reinterpret_cast<const float3*>(pointer), device,
                                                             validate);
            },
            "Synchronously set every tracked global angular velocity from CUDA ``float3`` storage. Each value is "
            "converted to the owner's local principal-axis frame using its orientation at call time. The buffer must "
            "contain one element per tracked owner; remote input is copied to the dynamic-worker device before "
            "unpacking.",
            py::arg("pointer"), py::arg("device"), py::arg("validate") = true)
        .def(
            "OrientationQuaternionsToDevice",
            [](deme::DEMTracker& tracker, std::uintptr_t pointer, size_t capacity, int device, bool validate) {
                tracker.OrientationQuaternionsToDevice(reinterpret_cast<float4*>(pointer), capacity, device, validate);
            },
            "Synchronously copy all tracked local-to-global quaternions to caller-owned CUDA ``float4`` storage in "
            "``(x, y, z, w)`` order. ``capacity`` is in elements.",
            py::arg("pointer"), py::arg("capacity"), py::arg("device"), py::arg("validate") = true)
        .def(
            "SetOrientationQuaternionsFromDevice",
            [](deme::DEMTracker& tracker, std::uintptr_t pointer, int device, bool validate) {
                tracker.SetOrientationQuaternionsFromDevice(reinterpret_cast<const float4*>(pointer), device, validate);
            },
            "Synchronously set every tracked local-to-global quaternion from CUDA ``float4`` storage in ``(x, y, z, "
            "w)`` order. With validation enabled, non-finite and zero-length inputs are rejected; finite, nonzero "
            "inputs are always normalized. The buffer must contain one element per tracked owner; remote input is "
            "copied to the dynamic-worker device before validation and unpacking.",
            py::arg("pointer"), py::arg("device"), py::arg("validate") = true)
        .def(
            "FamiliesToDevice",
            [](deme::DEMTracker& tracker, std::uintptr_t pointer, size_t capacity, int device, bool validate) {
                tracker.FamiliesToDevice(reinterpret_cast<unsigned int*>(pointer), capacity, device, validate);
            },
            "Synchronously copy all tracked family numbers to caller-owned CUDA ``uint32`` storage. ``capacity`` is "
            "in elements; ``device`` is the allocation's logical CUDA device.",
            py::arg("pointer"), py::arg("capacity"), py::arg("device"), py::arg("validate") = true)
        .def(
            "MassesToDevice",
            [](deme::DEMTracker& tracker, std::uintptr_t pointer, size_t capacity, int device, bool validate) {
                tracker.MassesToDevice(reinterpret_cast<float*>(pointer), capacity, device, validate);
            },
            "Synchronously copy all tracked masses to caller-owned CUDA ``float32`` storage. ``capacity`` is in "
            "elements; ``device`` is the allocation's logical CUDA device.",
            py::arg("pointer"), py::arg("capacity"), py::arg("device"), py::arg("validate") = true)
        .def(
            "MOIsToDevice",
            [](deme::DEMTracker& tracker, std::uintptr_t pointer, size_t capacity, int device, bool validate) {
                tracker.MOIsToDevice(reinterpret_cast<float3*>(pointer), capacity, device, validate);
            },
            "Synchronously copy all tracked principal moments of inertia to caller-owned CUDA ``float3`` storage. "
            "``capacity`` is in elements; ``device`` is the allocation's logical CUDA device.",
            py::arg("pointer"), py::arg("capacity"), py::arg("device"), py::arg("validate") = true)
        .def(
            "ContactAccelerationsToDevice",
            [](deme::DEMTracker& tracker, std::uintptr_t pointer, size_t capacity, int device, bool validate) {
                tracker.ContactAccelerationsToDevice(reinterpret_cast<float3*>(pointer), capacity, device, validate);
            },
            "Synchronously copy all tracked global-frame contact accelerations to caller-owned CUDA ``float3`` "
            "storage. Gravity and other non-contact acceleration are excluded.",
            py::arg("pointer"), py::arg("capacity"), py::arg("device"), py::arg("validate") = true)
        .def(
            "ContactAngularAccelerationsLocalToDevice",
            [](deme::DEMTracker& tracker, std::uintptr_t pointer, size_t capacity, int device, bool validate) {
                tracker.ContactAngularAccelerationsLocalToDevice(reinterpret_cast<float3*>(pointer), capacity, device,
                                                                 validate);
            },
            "Synchronously copy all tracked local-frame contact angular accelerations to caller-owned CUDA "
            "``float3`` storage.",
            py::arg("pointer"), py::arg("capacity"), py::arg("device"), py::arg("validate") = true)
        .def(
            "ContactAngularAccelerationsGlobalToDevice",
            [](deme::DEMTracker& tracker, std::uintptr_t pointer, size_t capacity, int device, bool validate) {
                tracker.ContactAngularAccelerationsGlobalToDevice(reinterpret_cast<float3*>(pointer), capacity, device,
                                                                  validate);
            },
            "Synchronously copy all tracked global-frame contact angular accelerations to caller-owned CUDA "
            "``float3`` storage.",
            py::arg("pointer"), py::arg("capacity"), py::arg("device"), py::arg("validate") = true)
        .def(
            "ContactWrenches",
            [](deme::DEMTracker& tracker) {
                std::vector<float3> forces;
                std::vector<float3> torques;
                tracker.ContactWrenches(forces, torques);
                return py::make_tuple(forces, torques);
            },
            R"doc(Return ``(forces, torques)`` with one reduced contact wrench per tracked owner.

Both tuple elements preserve tracker owner order and contain ``float3`` values in the global frame. Torque is measured
about each owner's current DEME position and includes both force-generated moments and force-model-only torque. Owners
without recorded contact receive zero. This reads current dT force records without triggering contact detection or
force evaluation, so callers must advance/synchronize the simulation as needed. Contact recording must remain enabled.)doc")
        .def(
            "ContactWrenchesToDevice",
            [](deme::DEMTracker& tracker, std::uintptr_t forces, std::uintptr_t torques, size_t capacity, int device) {
                tracker.ContactWrenchesToDevice(reinterpret_cast<float3*>(forces), reinterpret_cast<float3*>(torques),
                                                capacity, device);
            },
            R"doc(Synchronously write one reduced contact wrench per tracked owner to CUDA ``float3`` buffers.

Outputs preserve tracker owner order. Force and torque are global-frame resultants; torque is measured about each
owner's current DEME position and includes force-generated moments and force-model-only torque. Owners without contact
receive zero. ``force_destination`` and ``torque_destination`` are integer addresses of writable CUDA ``float3``
storage. ``capacity`` is the element capacity of each buffer and must cover every tracked owner. Both buffers must
belong to logical CUDA ``device``; CUDA selects the available inter-device transfer route for remote output. The call
is synchronous, so both buffers may be consumed when it returns. This reads current recorded dT forces and does not
trigger contact detection or force evaluation. Contact recording must remain enabled.)doc",
            py::arg("force_destination"), py::arg("torque_destination"), py::arg("capacity"), py::arg("device"))
        .def(
            "OwnerWildcardValuesToDevice",
            [](deme::DEMTracker& tracker, const std::string& name, std::uintptr_t pointer, size_t capacity, int device,
               bool validate) {
                tracker.OwnerWildcardValuesToDevice(name, reinterpret_cast<float*>(pointer), capacity, device,
                                                    validate);
            },
            "Synchronously copy the named ``float32`` owner wildcard for every tracked owner to caller-owned CUDA "
            "storage. ``capacity`` is in owner elements.",
            py::arg("name"), py::arg("pointer"), py::arg("capacity"), py::arg("device"), py::arg("validate") = true)

        .def("SetPos", static_cast<void (deme::DEMTracker::*)(float3, size_t)>(&deme::DEMTracker::SetPos),
             "Set the position of this tracked object.", py::arg("pos"), py::arg("offset") = 0)
        .def("SetPos", static_cast<void (deme::DEMTracker::*)(const std::vector<float3>&)>(&deme::DEMTracker::SetPos),
             "Set the positions of consecutive tracked objects.", py::arg("pos"))

        .def("SetAngVel", static_cast<void (deme::DEMTracker::*)(float3, size_t)>(&deme::DEMTracker::SetAngVel),
             "Set the angular velocity of this tracked object in its own local coordinate system.", py::arg("angVel"),
             py::arg("offset") = 0)
        .def("SetAngVel",
             static_cast<void (deme::DEMTracker::*)(const std::vector<float3>&)>(&deme::DEMTracker::SetAngVel),
             "Set the angular velocity of consecutive tracked objects in their own local coordinate systems.",
             py::arg("angVel"))

        .def("SetVel", static_cast<void (deme::DEMTracker::*)(float3, size_t)>(&deme::DEMTracker::SetVel),
             "Set the velocity of this tracked object in global frame.", py::arg("vel"), py::arg("offset") = 0)
        .def("SetVel", static_cast<void (deme::DEMTracker::*)(const std::vector<float3>&)>(&deme::DEMTracker::SetVel),
             "Set the velocity of consecutive tracked objects in global frame.", py::arg("vel"))

        .def("SetOriQ", static_cast<void (deme::DEMTracker::*)(float4, size_t)>(&deme::DEMTracker::SetOriQ),
             "Set the quaternion which represents the orientation of this tracked object's coordinate system.",
             py::arg("oriQ"), py::arg("offset") = 0)
        .def("SetOriQ", static_cast<void (deme::DEMTracker::*)(const std::vector<float4>&)>(&deme::DEMTracker::SetOriQ),
             "Set the quaternion which represents the orientation of consecutive tracked objects' coordinate systems.",
             py::arg("oriQ"))

        .def("AddAcc", static_cast<void (deme::DEMTracker::*)(float3, size_t)>(&deme::DEMTracker::AddAcc),
             "Add an extra acc to the tracked body, for the next time step. Note if the user intends to add a "
             "persistent external force, then using family prescription is the better method.",
             py::arg("acc"), py::arg("offset") = 0)
        .def("AddAcc", static_cast<void (deme::DEMTracker::*)(const std::vector<float3>&)>(&deme::DEMTracker::AddAcc),
             "Add an extra acc to consecutive tracked objects, (only) for the next time step. Note if the user intends "
             "to add a persistent external force, then using family prescription is the better method.",
             py::arg("acc"))
        .def(
            "AddAccFromDevice",
            [](deme::DEMTracker& tracker, std::uintptr_t source, int device, bool validate) {
                tracker.AddAccFromDevice(reinterpret_cast<const float3*>(source), device, validate);
            },
            R"doc(Synchronously queue one global-frame linear acceleration per tracked owner from CUDA memory.

``source`` is an integer address containing one CUDA ``float3`` per tracked owner in tracker order. Remote input is
copied from ``source_device`` to the dynamic-worker device before unpacking. Values replace any previously queued
next-step linear-acceleration contribution. Contact acceleration is accumulated on top during the next
force/integration step, gravity is applied separately, and the queued contribution is consumed after one step. Pass
``validate=False`` only when the range and pointer metadata are known to be valid. The source buffer may be reused when
this call returns.)doc",
            py::arg("source"), py::arg("source_device"), py::arg("validate") = true)

        .def("AddAngAcc", static_cast<void (deme::DEMTracker::*)(float3, size_t)>(&deme::DEMTracker::AddAngAcc),
             "Add an extra angular acceleration to the tracked body, for the next time step. Note if the user intends "
             "to add a persistent external torque, then using family prescription is the better method.",
             py::arg("angAcc"), py::arg("offset") = 0)
        .def("AddAngAcc",
             static_cast<void (deme::DEMTracker::*)(const std::vector<float3>&)>(&deme::DEMTracker::AddAngAcc),
             "Add an extra angular acceleration to consecutive tracked objects, (only) for the next time step. Note if "
             "the user intends to add a persistent external torque, then using family prescription is the better "
             "method.",
             py::arg("angAcc"))
        .def(
            "AddAngAccFromDevice",
            [](deme::DEMTracker& tracker, std::uintptr_t source, int device, bool validate) {
                tracker.AddAngAccFromDevice(reinterpret_cast<const float3*>(source), device, validate);
            },
            R"doc(Synchronously queue one local-frame angular acceleration per tracked owner from CUDA memory.

``source`` is an integer address containing one CUDA ``float3`` per tracked owner in tracker order. Remote input is
copied from ``source_device`` to the dynamic-worker device before unpacking. Values use each owner's local principal-
axis frame and replace any previously queued next-step angular-acceleration contribution. Contact angular acceleration
is accumulated on top during the next force/integration step, and the queued contribution is consumed after one step.
Pass ``validate=False`` only when the range and pointer metadata are known to be valid. The source buffer may be reused
when this call returns.)doc",
            py::arg("source"), py::arg("source_device"), py::arg("validate") = true)

        .def("SetFamily", static_cast<void (deme::DEMTracker::*)(unsigned int)>(&deme::DEMTracker::SetFamily),
             "Change the family numbers of all the entities tracked by this tracker.", py::arg("fam_num"))
        .def("SetFamily", static_cast<void (deme::DEMTracker::*)(unsigned int, size_t)>(&deme::DEMTracker::SetFamily),
             "Change the family number of one entities tracked by this tracker.", py::arg("fam_num"), py::arg("offset"))

        .def("UpdateMesh",
             static_cast<void (deme::DEMTracker::*)(const std::vector<float3>&)>(&deme::DEMTracker::UpdateMesh),
             "Apply the new mesh node positions such that the tracked mesh is replaced by the new_nodes.",
             py::arg("new_nodes"))
        .def("UpdateMeshByIncrement",
             static_cast<void (deme::DEMTracker::*)(const std::vector<float3>&)>(
                 &deme::DEMTracker::UpdateMeshByIncrement),
             "Change the coordinates of each mesh node by the given amount.", py::arg("deformation"))

        .def("GetMeshNodesGlobal",
             static_cast<std::vector<float3> (deme::DEMTracker::*)()>(&deme::DEMTracker::GetMeshNodesGlobal),
             "Get the current locations of all the nodes in the mesh being tracked.")

        .def("GetMesh", &deme::DEMTracker::GetMesh, "Get a handle for the mesh this tracker is tracking.")

        .def("SetOwnerWildcardValue", &deme::DEMTracker::SetOwnerWildcardValue,
             "Set a wildcard value of the owner this tracker is tracking.", py::arg("name"), py::arg("wc"),
             py::arg("offset") = 0)
        .def("SetOwnerWildcardValues", &deme::DEMTracker::SetOwnerWildcardValues,
             "Set a wildcard value of the owner this tracker is tracking.", py::arg("name"), py::arg("wc"))

        .def(
            "GetContactForcesAndLocalTorque",
            static_cast<size_t (deme::DEMTracker::*)(std::vector<float3>&, std::vector<float3>&, std::vector<float3>&,
                                                     size_t)>(&deme::DEMTracker::GetContactForcesAndLocalTorque),
            "Get all contact forces and local torques that concern this tracked object. Returns number of force pairs.",
            py::arg("points"), py::arg("forces"), py::arg("torques"), py::arg("offset") = 0)
        .def(
            "GetContactForcesAndLocalTorqueForAll",
            static_cast<size_t (deme::DEMTracker::*)(std::vector<float3>&, std::vector<float3>&, std::vector<float3>&)>(
                &deme::DEMTracker::GetContactForcesAndLocalTorqueForAll),
            "Get all contact forces and local torques that concern all objects tracked by this tracker. Returns number "
            "of force pairs.",
            py::arg("points"), py::arg("forces"), py::arg("torques"))

        .def("GetContactForcesAndGlobalTorque",
             static_cast<size_t (deme::DEMTracker::*)(std::vector<float3>&, std::vector<float3>&, std::vector<float3>&,
                                                      size_t)>(&deme::DEMTracker::GetContactForcesAndGlobalTorque),
             "Get all contact forces and global torques that concern this tracked object. Returns number of force "
             "pairs.",
             py::arg("points"), py::arg("forces"), py::arg("torques"), py::arg("offset") = 0)
        .def(
            "GetContactForcesAndGlobalTorqueForAll",
            static_cast<size_t (deme::DEMTracker::*)(std::vector<float3>&, std::vector<float3>&, std::vector<float3>&)>(
                &deme::DEMTracker::GetContactForcesAndGlobalTorqueForAll),
            "Get all contact forces and global torques that concern all objects tracked by this tracker. Returns "
            "number of force pairs.",
            py::arg("points"), py::arg("forces"), py::arg("torques"))

        .def("GetContactForces",
             static_cast<size_t (deme::DEMTracker::*)(std::vector<float3>&, std::vector<float3>&, size_t)>(
                 &deme::DEMTracker::GetContactForces),
             "Get all contact forces that concern this tracked object. Returns number of force pairs.",
             py::arg("points"), py::arg("forces"), py::arg("offset") = 0)
        .def("GetContactForcesForAll",
             static_cast<size_t (deme::DEMTracker::*)(std::vector<float3>&, std::vector<float3>&)>(
                 &deme::DEMTracker::GetContactForcesForAll),
             "Get all contact forces that concern all objects tracked by this tracker. Returns number of force pairs.",
             py::arg("points"), py::arg("forces"))
        .def(
            "GetContactForcesForAllToDevice",
            [](deme::DEMTracker& tracker, std::uintptr_t points, std::uintptr_t forces, size_t capacity, int device) {
                return tracker.GetContactForcesForAllToDevice(reinterpret_cast<float3*>(points),
                                                              reinterpret_cast<float3*>(forces), capacity, device);
            },
            R"doc(
Synchronously compact contact points and forces for all tracked owners into CUDA ``float3`` arrays.

Each pointer must address ``capacity`` elements on logical CUDA ``device``.
Capacity must cover the simulation's total recorded-contact count. The return
value is the number of valid compacted rows. Contact force recording must be
enabled.
)doc",
            py::arg("points"), py::arg("forces"), py::arg("capacity"), py::arg("device"))
        .def(
            "GetContactForcesAndLocalTorqueForAllToDevice",
            [](deme::DEMTracker& tracker, std::uintptr_t points, std::uintptr_t forces, std::uintptr_t torques,
               size_t capacity, int device) {
                return tracker.GetContactForcesAndLocalTorqueForAllToDevice(
                    reinterpret_cast<float3*>(points), reinterpret_cast<float3*>(forces),
                    reinterpret_cast<float3*>(torques), capacity, device);
            },
            "Synchronously compact contact points, forces, and owner-local extra torques into caller-owned CUDA "
            "``float3`` arrays. Capacity must cover the total recorded-contact count; the return value is the number "
            "of valid rows.",
            py::arg("points"), py::arg("forces"), py::arg("torques"), py::arg("capacity"), py::arg("device"))
        .def(
            "GetContactForcesAndGlobalTorqueForAllToDevice",
            [](deme::DEMTracker& tracker, std::uintptr_t points, std::uintptr_t forces, std::uintptr_t torques,
               size_t capacity, int device) {
                return tracker.GetContactForcesAndGlobalTorqueForAllToDevice(
                    reinterpret_cast<float3*>(points), reinterpret_cast<float3*>(forces),
                    reinterpret_cast<float3*>(torques), capacity, device);
            },
            "Synchronously compact contact points, forces, and global-frame extra torques into caller-owned CUDA "
            "``float3`` arrays. Capacity must cover the total recorded-contact count; the return value is the number "
            "of valid rows.",
            py::arg("points"), py::arg("forces"), py::arg("torques"), py::arg("capacity"), py::arg("device"));

    py::class_<deme::DEMForceModel, std::shared_ptr<deme::DEMForceModel>>(obj, "DEMForceModel")
        .def(py::init<deme::FORCE_MODEL>(), py::arg("model"),
             "Create a force-model definition of the selected built-in type.")
        .def("SetForceModelType", &deme::DEMForceModel::SetForceModelType, "Set the contact force model type")

        .def("DefineCustomModel", &deme::DEMForceModel::DefineCustomModel,
             "Define user-custom force model with a string which is your force calculation code.")
        .def("ReadCustomModelFile", &deme::DEMForceModel::ReadCustomModelFile,
             "Read user-custom force model from a file (which by default should reside in kernel/DEMUserScripts), "
             "which contains your force calculation code. Returns 0 if read successfully, otherwise 1.")
        .def("DefineCustomModelPrerequisites", &deme::DEMForceModel::DefineCustomModelPrerequisites,
             "Define user-custom force model's utility __device__ functions with a string.")
        .def("ReadCustomModelPrerequisitesFile", &deme::DEMForceModel::ReadCustomModelPrerequisitesFile,
             "Read user-custom force model's utility __device__ functions from a file (which by default should reside "
             "in kernel/DEMUserScripts). Returns 0 if read successfully, otherwise 1.")

        .def("SetMustHaveMatProp", &deme::DEMForceModel::SetMustHaveMatProp,
             "Specifiy the material properties that this force model will use")
        .def("SetMustPairwiseMatProp", &deme::DEMForceModel::SetMustPairwiseMatProp,
             "Specifiy the material properties that are pair-wise (instead of being associated with each individual "
             "material).")
        .def("SetPerContactWildcards", &deme::DEMForceModel::SetPerContactWildcards,
             "Set the names for the extra quantities that will be associated with each contact pair. For example, "
             "history-based models should have 3 float arrays to store contact history. Only float is supported. Note "
             "the initial value of all contact wildcard arrays is automatically 0")
        .def("SetPerOwnerWildcards", &deme::DEMForceModel::SetPerOwnerWildcards,
             " Set the names for the extra quantities that will be associated with each owner. For example, you can "
             "use this to associate a cohesion parameter to each particle. Only float is supported.");

    // The scene/frame interface also supports external renderers without the optional native viewer library.
    py::class_<deme::DEMVisualizationScene::Sphere>(obj, "VisualizationSphere")
        .def_readonly("offset", &deme::DEMVisualizationScene::Sphere::offset)
        .def_readonly("radius", &deme::DEMVisualizationScene::Sphere::radius)
        .def_readonly("owner", &deme::DEMVisualizationScene::Sphere::owner);
    py::class_<deme::DEMVisualizationScene::Triangle>(obj, "VisualizationTriangle")
        .def_readonly("a", &deme::DEMVisualizationScene::Triangle::a)
        .def_readonly("b", &deme::DEMVisualizationScene::Triangle::b)
        .def_readonly("c", &deme::DEMVisualizationScene::Triangle::c)
        .def_readonly("owner", &deme::DEMVisualizationScene::Triangle::owner);
    py::class_<deme::DEMVisualizationScene>(obj, "VisualizationScene")
        .def_readonly("revision", &deme::DEMVisualizationScene::revision)
        .def_readonly("spheres", &deme::DEMVisualizationScene::spheres)
        .def_readonly("triangles", &deme::DEMVisualizationScene::triangles);
    py::class_<deme::DEMVisualizationFrame>(obj, "VisualizationFrame")
        .def(py::init<>())
        .def_readonly("simulation_time", &deme::DEMVisualizationFrame::simulation_time)
        .def_readonly("revision", &deme::DEMVisualizationFrame::revision)
        .def_readonly("positions", &deme::DEMVisualizationFrame::positions)
        .def_readonly("orientations", &deme::DEMVisualizationFrame::orientations)
        .def_readonly("families", &deme::DEMVisualizationFrame::families)
        .def_readonly("velocities", &deme::DEMVisualizationFrame::velocities);

    py::class_<deme::DEMSolver>(
        obj, "DEMSolver",
        R"doc(
Own a DEM simulation and its dynamic and kinematic CUDA workers.

Configure materials, geometry, particles, solver options, and trackers before
calling ``Initialize``. Device placement is fixed at construction. Keep this
object alive while using any solver-owned material, template, tracker, or
inspector handle.

Fixed-size owner CUDA exchange methods validate ranges, capacities, and pointer metadata by default. Pass
``validate=False`` only when those preconditions are guaranteed and avoiding validation overhead matters. Required
device routing and documented transformations still apply.
)doc")
        .def("GetVisualizationScene", &deme::DEMSolver::GetVisualizationScene)
        .def("GetVisualizationFrame", &deme::DEMSolver::GetVisualizationFrame,
             py::arg("frame"), py::arg("include_velocities") = false)
        .def(py::init<unsigned int>(), py::arg("nGPUs") = 2,
             R"doc(
Construct a solver using one or two logical CUDA devices.

With one GPU, both workers use device 0. With two requested GPUs and at least
two visible devices, the dynamic worker uses device 0 and the kinematic worker
uses device 1. If only one device is visible, both workers use device 0 and a
warning is emitted. Only 1 and 2 are valid values.
)doc")
        .def(py::init<const std::vector<int>&>(), py::arg("device_ids"),
             R"doc(
Construct a solver on explicit logical CUDA device IDs.

One ID places both workers on that device. For two IDs, the first selects the
dynamic worker and the second selects the kinematic worker. Repeated IDs are
valid. IDs use the numbering visible to this process, including the effects of
``CUDA_VISIBLE_DEVICES``.
)doc")
        .def("GetGPUDeviceIDs", &deme::DEMSolver::GetGPUDeviceIDs,
             "Return ``[dynamic_worker_device, kinematic_worker_device]`` using process-visible logical CUDA IDs.")
        .def("UpdateStepSize", &deme::DEMSolver::UpdateStepSize,
             "Legacy name for SetTimeStepSize, kept for backward compatibility.", py::arg("ts") = -1.0)
        .def("SetNoForceRecord", &deme::DEMSolver::SetNoForceRecord,
             "Instruct the solver that there is no need to record the contact force (and contact point location etc.) "
             "in an array. After initialization, call UpdateSimParams() for this change to take effect in dT.",
             py::arg("flag") = true)
        .def("LoadSphereType",
             static_cast<std::shared_ptr<deme::DEMClumpTemplate> (deme::DEMSolver::*)(
                 float, float, const std::shared_ptr<deme::DEMMaterial>&)>(&deme::DEMSolver::LoadSphereType),
             "Load a reusable one-sphere clump template, deriving its spherical moment of inertia.", py::arg("mass"),
             py::arg("radius"), py::arg("material"))
        .def("LoadSphereType",
             static_cast<std::shared_ptr<deme::DEMClumpTemplate> (deme::DEMSolver::*)(
                 float, float, float, const std::shared_ptr<deme::DEMMaterial>&)>(&deme::DEMSolver::LoadSphereType),
             "Load a reusable one-sphere clump template with an explicitly supplied scalar moment of inertia.",
             py::arg("mass"), py::arg("moi"), py::arg("radius"), py::arg("material"))
        .def("LoadCombinedClumpType",
             static_cast<std::shared_ptr<deme::DEMCombinedTemplate> (deme::DEMSolver::*)(
                 const std::vector<std::shared_ptr<deme::DEMClumpTemplate>>&, const std::vector<float3>&,
                 const std::vector<float4>&, size_t)>(&deme::DEMSolver::LoadCombinedClumpType),
             "Load a rigid combined-clump template with fixed member-relative transforms.",
             py::arg("component_templates"), py::arg("component_rel_pos"),
             py::arg("component_rel_oriQ") = std::vector<float4>(), py::arg("master_component") = 0)
        .def("LoadCombinedMeshType",
             static_cast<std::shared_ptr<deme::DEMCombinedTemplate> (deme::DEMSolver::*)(
                 const std::vector<std::shared_ptr<deme::DEMMesh>>&, const std::vector<float3>&,
                 const std::vector<float4>&, size_t)>(&deme::DEMSolver::LoadCombinedMeshType),
             "Load a rigid combined-mesh template with fixed member-relative transforms.",
             py::arg("component_templates"), py::arg("component_rel_pos"),
             py::arg("component_rel_oriQ") = std::vector<float4>(), py::arg("master_component") = 0)
        .def("AddCombinedFromTemplate",
             static_cast<std::shared_ptr<deme::DEMCombinedInstances> (deme::DEMSolver::*)(
                 const std::shared_ptr<deme::DEMCombinedTemplate>&, const std::vector<float3>&,
                 const std::vector<float4>&)>(&deme::DEMSolver::AddCombinedFromTemplate),
             "Instantiate a combined template at one or more user-specified global poses (batch).",
             py::arg("combined_template"), py::arg("init_pos"), py::arg("init_oriQ") = std::vector<float4>())
        .def("AddCombinedFromTemplate",
             static_cast<std::shared_ptr<deme::DEMCombinedInstances> (deme::DEMSolver::*)(
                 const std::shared_ptr<deme::DEMCombinedTemplate>&, const float3&, const float4&)>(
                 &deme::DEMSolver::AddCombinedFromTemplate),
             "Instantiate a combined template at a single user-specified global pose.", py::arg("combined_template"),
             py::arg("init_pos") = make_float3(0), py::arg("init_oriQ") = make_float4(0, 0, 0, 1))
        .def("GetNumCombinedInstances", &deme::DEMSolver::GetNumCombinedInstances,
             "Return the number of combined instances currently cached.")
        .def(
            "GetCombinedInstanceInfo",
            [](deme::DEMSolver& self, size_t combined_instance_id) {
                deme::bodyID_t master_owner_id = deme::NULL_BODYID;
                std::vector<deme::bodyID_t> member_owner_ids;
                std::vector<float3> member_rel_pos;
                std::vector<float4> member_rel_oriQ;
                const bool ok = self.GetCombinedInstanceInfo(combined_instance_id, master_owner_id, member_owner_ids,
                                                             member_rel_pos, member_rel_oriQ);
                return py::make_tuple(ok, master_owner_id, member_owner_ids, member_rel_pos, member_rel_oriQ);
            },
            "Query combined-instance metadata; returns (ok, master_owner_id, member_owner_ids, member_rel_pos, "
            "member_rel_oriQ).",
            py::arg("combined_instance_id"))
        .def("EnsureKernelErrMsgLineNum", &deme::DEMSolver::EnsureKernelErrMsgLineNum,
             "If true, each jitification string substitution will do a one-liner to one-liner replacement, so that if "
             "the kernel compilation fails, the error meessage line number will reflex the actual spot where that "
             "happens (instead of some random number).",
             py::arg("flag") = true)
        .def("SetCollectAccRightAfterForceCalc", &deme::DEMSolver::SetCollectAccRightAfterForceCalc,
             "Reduce contact forces to accelerations right after calculating them, in the same kernel. This may give "
             "some performance boost if you have only polydisperse spheres, no clumps. After initialization, call "
             "UpdateSimParams() for this change to take effect in dT.",
             py::arg("flag") = true)

        .def("SetInitBinSize", &deme::DEMSolver::SetInitBinSize,
             " Explicitly instruct the bin size (for contact detection) that the solver should use.")
        .def("SetOutputFormat",
             static_cast<void (deme::DEMSolver::*)(const std::string&)>(&deme::DEMSolver::SetOutputFormat),
             "Choose sphere and clump output file format by name.", py::arg("format"))
        .def("SetOutputFormat",
             static_cast<void (deme::DEMSolver::*)(deme::OUTPUT_FORMAT)>(&deme::DEMSolver::SetOutputFormat),
             "Choose sphere and clump output file format. VTK is supported by WriteSphereFile.", py::arg("format"))
        .def("GetNumContacts", &deme::DEMSolver::GetNumContacts,
             "Get the number of kT-reported potential contact pairs.")
        .def(
            "SetOwnerPositionFromDevice",
            [](deme::DEMSolver& self, deme::bodyID_t owner_id, std::uintptr_t source, int device, size_t count,
               bool validate) {
                self.SetOwnerPositionFromDevice(owner_id, reinterpret_cast<const float3*>(source), device, count,
                                                validate);
            },
            R"doc(Synchronously set consecutive owner positions from ``count`` CUDA ``float3`` elements.

Positions are global-frame values. ``source_device`` identifies the CUDA-accessible source allocation; remote input is
copied to the dynamic-worker device before unpacking. When this call returns the source buffer may be reused.)doc",
            py::arg("owner_id"), py::arg("source"), py::arg("source_device"), py::arg("count") = 1,
            py::arg("validate") = true)
        .def(
            "SetOwnerOriQFromDevice",
            [](deme::DEMSolver& self, deme::bodyID_t owner_id, std::uintptr_t source, int device, size_t count,
               bool validate) {
                self.SetOwnerOriQFromDevice(owner_id, reinterpret_cast<const float4*>(source), device, count, validate);
            },
            R"doc(Synchronously set consecutive owner orientations from ``count`` CUDA ``float4`` elements.

Quaternion elements use public DEME ordering ``(x, y, z, w)``. ``source_device`` identifies the CUDA-accessible source
allocation; remote input is copied to the dynamic-worker device before validation and unpacking. With
``validate=True``, non-finite and zero-length quaternions are rejected. Every finite, nonzero quaternion is normalized
before storage even when validation is disabled. When this call returns the source buffer may be reused.)doc",
            py::arg("owner_id"), py::arg("source"), py::arg("source_device"), py::arg("count") = 1,
            py::arg("validate") = true)
        .def(
            "SetOwnerVelocityFromDevice",
            [](deme::DEMSolver& self, deme::bodyID_t owner_id, std::uintptr_t source, int device, size_t count,
               bool validate) {
                self.SetOwnerVelocityFromDevice(owner_id, reinterpret_cast<const float3*>(source), device, count,
                                                validate);
            },
            R"doc(Synchronously set consecutive global linear velocities from ``count`` CUDA ``float3`` elements.

``source_device`` identifies the CUDA-accessible source allocation; remote input is copied to the dynamic-worker
device before unpacking. When this call returns the source buffer may be reused.)doc",
            py::arg("owner_id"), py::arg("source"), py::arg("source_device"), py::arg("count") = 1,
            py::arg("validate") = true)
        .def(
            "SetOwnerAngVelFromDevice",
            [](deme::DEMSolver& self, deme::bodyID_t owner_id, std::uintptr_t source, int device, size_t count,
               bool validate) {
                self.SetOwnerAngVelFromDevice(owner_id, reinterpret_cast<const float3*>(source), device, count,
                                              validate);
            },
            R"doc(Synchronously set consecutive local-frame angular velocities from CUDA ``float3`` elements.

This has the same frame semantics as ``SetOwnerAngVel``. ``source_device`` identifies the CUDA-accessible source
allocation; remote input is copied to the dynamic-worker device before unpacking. When this call returns the source
buffer may be reused.)doc",
            py::arg("owner_id"), py::arg("source"), py::arg("source_device"), py::arg("count") = 1,
            py::arg("validate") = true)
        .def(
            "SetOwnerAngVelGlobalFromDevice",
            [](deme::DEMSolver& self, deme::bodyID_t owner_id, std::uintptr_t source, int device, size_t count,
               bool validate) {
                self.SetOwnerAngVelGlobalFromDevice(owner_id, reinterpret_cast<const float3*>(source), device, count,
                                                    validate);
            },
            R"doc(Synchronously set consecutive global angular velocities from ``count`` CUDA ``float3`` elements.

Each value is converted to the owner's local principal-axis frame using that owner's orientation at call time.
``source_device`` identifies the CUDA-accessible source allocation; remote input is copied to the dynamic-worker
device before unpacking. When this call returns the source buffer may be reused.)doc",
            py::arg("owner_id"), py::arg("source"), py::arg("source_device"), py::arg("count") = 1,
            py::arg("validate") = true)
        .def(
            "AddOwnerNextStepAccFromDevice",
            [](deme::DEMSolver& self, deme::bodyID_t owner_id, std::uintptr_t source, int device, size_t count,
               bool validate) {
                self.AddOwnerNextStepAccFromDevice(owner_id, reinterpret_cast<const float3*>(source), device, count,
                                                   validate);
            },
            R"doc(Synchronously queue global-frame linear accelerations for consecutive owners from CUDA memory.

``source`` is an integer address containing ``count`` CUDA ``float3`` values. Remote input is copied from
``source_device`` to the dynamic-worker device before unpacking. Values replace any previously queued next-step
linear-acceleration contribution. Contact acceleration is accumulated on top during the next force/integration step,
gravity is applied separately, and the queued contribution is consumed after one step. Pass ``validate=False`` only
when the owner range and pointer metadata are known to be valid. A zero count is a no-op; otherwise the source buffer
may be reused when this call returns.)doc",
            py::arg("owner_id"), py::arg("source"), py::arg("source_device"), py::arg("count") = 1,
            py::arg("validate") = true)
        .def(
            "AddOwnerNextStepAngAccFromDevice",
            [](deme::DEMSolver& self, deme::bodyID_t owner_id, std::uintptr_t source, int device, size_t count,
               bool validate) {
                self.AddOwnerNextStepAngAccFromDevice(owner_id, reinterpret_cast<const float3*>(source), device, count,
                                                      validate);
            },
            R"doc(Synchronously queue local-frame angular accelerations for consecutive owners from CUDA memory.

``source`` is an integer address containing ``count`` CUDA ``float3`` values. Remote input is copied from
``source_device`` to the dynamic-worker device before unpacking. Values use each owner's local principal-axis frame
and replace any previously queued next-step angular-acceleration contribution. Contact angular acceleration is
accumulated on top during the next force/integration step, and the queued contribution is consumed after one step. Pass
``validate=False`` only when the owner range and pointer metadata are known to be valid. A zero count is a no-op;
otherwise the source buffer may be reused when this call returns.)doc",
            py::arg("owner_id"), py::arg("source"), py::arg("source_device"), py::arg("count") = 1,
            py::arg("validate") = true)
        .def(
            "GetOwnerPositionToDevice",
            [](const deme::DEMSolver& self, std::uintptr_t destination, size_t capacity, int device,
               deme::bodyID_t owner_id, deme::bodyID_t count, bool validate) {
                self.GetOwnerPositionToDevice(reinterpret_cast<float3*>(destination), capacity, device, owner_id,
                                              count, validate);
            },
            "Synchronously write consecutive global owner positions as CUDA float3 elements.", py::arg("destination"),
            py::arg("capacity"), py::arg("destination_device"), py::arg("first_owner_id"), py::arg("count") = 1,
            py::arg("validate") = true)
        .def(
            "GetOwnerOriQToDevice",
            [](const deme::DEMSolver& self, std::uintptr_t destination, size_t capacity, int device,
               deme::bodyID_t owner_id, deme::bodyID_t count, bool validate) {
                self.GetOwnerOriQToDevice(reinterpret_cast<float4*>(destination), capacity, device, owner_id, count,
                                          validate);
            },
            "Synchronously write consecutive owner quaternions as CUDA float4 elements in (x, y, z, w) order.",
            py::arg("destination"), py::arg("capacity"), py::arg("destination_device"), py::arg("first_owner_id"),
            py::arg("count") = 1, py::arg("validate") = true)
        .def(
            "GetOwnerAngVelLocalToDevice",
            [](const deme::DEMSolver& self, std::uintptr_t destination, size_t capacity, int device,
               deme::bodyID_t owner_id, deme::bodyID_t count, bool validate) {
                self.GetOwnerAngVelLocalToDevice(reinterpret_cast<float3*>(destination), capacity, device, owner_id,
                                                 count, validate);
            },
            "Synchronously write consecutive local principal-axis-frame owner angular velocities as CUDA float3 "
            "elements.",
            py::arg("destination"), py::arg("capacity"), py::arg("destination_device"), py::arg("first_owner_id"),
            py::arg("count") = 1, py::arg("validate") = true)
        .def(
            "GetOwnerVelocityToDevice",
            [](const deme::DEMSolver& self, std::uintptr_t destination, size_t capacity, int device,
               deme::bodyID_t owner_id, deme::bodyID_t count, bool validate) {
                self.GetOwnerVelocityToDevice(reinterpret_cast<float3*>(destination), capacity, device, owner_id,
                                              count, validate);
            },
            "Synchronously write consecutive global owner velocities as CUDA float3 elements.", py::arg("destination"),
            py::arg("capacity"), py::arg("destination_device"), py::arg("first_owner_id"), py::arg("count") = 1,
            py::arg("validate") = true)
        .def(
            "GetOwnerAngVelGlobalToDevice",
            [](const deme::DEMSolver& self, std::uintptr_t destination, size_t capacity, int device,
               deme::bodyID_t owner_id, deme::bodyID_t count, bool validate) {
                self.GetOwnerAngVelGlobalToDevice(reinterpret_cast<float3*>(destination), capacity, device, owner_id,
                                                  count, validate);
            },
            "Synchronously write consecutive global owner angular velocities as CUDA float3 elements.",
            py::arg("destination"), py::arg("capacity"), py::arg("destination_device"), py::arg("first_owner_id"),
            py::arg("count") = 1, py::arg("validate") = true)
        .def(
            "GetOwnerAccToDevice",
            [](const deme::DEMSolver& self, std::uintptr_t destination, size_t capacity, int device,
               deme::bodyID_t owner_id, deme::bodyID_t count, bool validate) {
                self.GetOwnerAccToDevice(reinterpret_cast<float3*>(destination), capacity, device, owner_id, count,
                                         validate);
            },
            "Synchronously write consecutive global contact accelerations as CUDA float3 elements.",
            py::arg("destination"), py::arg("capacity"), py::arg("destination_device"), py::arg("first_owner_id"),
            py::arg("count") = 1, py::arg("validate") = true)
        .def(
            "GetOwnerAngAccGlobalToDevice",
            [](const deme::DEMSolver& self, std::uintptr_t destination, size_t capacity, int device,
               deme::bodyID_t owner_id, deme::bodyID_t count, bool validate) {
                self.GetOwnerAngAccGlobalToDevice(reinterpret_cast<float3*>(destination), capacity, device, owner_id,
                                                  count, validate);
            },
            "Synchronously write consecutive global contact angular accelerations as CUDA float3 elements.",
            py::arg("destination"), py::arg("capacity"), py::arg("destination_device"), py::arg("first_owner_id"),
            py::arg("count") = 1, py::arg("validate") = true)
        .def(
            "GetOwnerAngAccLocalToDevice",
            [](const deme::DEMSolver& self, std::uintptr_t destination, size_t capacity, int device,
               deme::bodyID_t owner_id, deme::bodyID_t count, bool validate) {
                self.GetOwnerAngAccLocalToDevice(reinterpret_cast<float3*>(destination), capacity, device, owner_id,
                                                 count, validate);
            },
            "Synchronously write consecutive local principal-axis-frame contact angular accelerations as CUDA "
            "float3 elements.",
            py::arg("destination"), py::arg("capacity"), py::arg("destination_device"), py::arg("first_owner_id"),
            py::arg("count") = 1, py::arg("validate") = true)
        .def(
            "GetOwnerFamilyToDevice",
            [](const deme::DEMSolver& self, std::uintptr_t destination, size_t capacity, int device,
               deme::bodyID_t owner_id, deme::bodyID_t count, bool validate) {
                self.GetOwnerFamilyToDevice(reinterpret_cast<unsigned int*>(destination), capacity, device, owner_id,
                                            count, validate);
            },
            "Synchronously write consecutive owner family numbers as CUDA uint32 elements.", py::arg("destination"),
            py::arg("capacity"), py::arg("destination_device"), py::arg("first_owner_id"), py::arg("count") = 1,
            py::arg("validate") = true)
        .def(
            "GetOwnerMassToDevice",
            [](const deme::DEMSolver& self, std::uintptr_t destination, size_t capacity, int device,
               deme::bodyID_t owner_id, deme::bodyID_t count, bool validate) {
                self.GetOwnerMassToDevice(reinterpret_cast<float*>(destination), capacity, device, owner_id, count,
                                          validate);
            },
            "Synchronously write consecutive owner masses as CUDA float elements.", py::arg("destination"),
            py::arg("capacity"), py::arg("destination_device"), py::arg("first_owner_id"), py::arg("count") = 1,
            py::arg("validate") = true)
        .def(
            "GetOwnerMOIToDevice",
            [](const deme::DEMSolver& self, std::uintptr_t destination, size_t capacity, int device,
               deme::bodyID_t owner_id, deme::bodyID_t count, bool validate) {
                self.GetOwnerMOIToDevice(reinterpret_cast<float3*>(destination), capacity, device, owner_id, count,
                                         validate);
            },
            "Synchronously write consecutive owner principal moments of inertia as CUDA float3 elements.",
            py::arg("destination"), py::arg("capacity"), py::arg("destination_device"), py::arg("first_owner_id"),
            py::arg("count") = 1, py::arg("validate") = true)
        .def(
            "GetOwnerWildcardValueToDevice",
            [](deme::DEMSolver& self, std::uintptr_t destination, size_t capacity, int device,
               deme::bodyID_t owner_id, const std::string& name, deme::bodyID_t count, bool validate) {
                self.GetOwnerWildcardValueToDevice(reinterpret_cast<float*>(destination), capacity, device, owner_id,
                                                   name, count, validate);
            },
            "Synchronously write one named owner wildcard for consecutive owners as CUDA float elements.",
            py::arg("destination"), py::arg("capacity"), py::arg("destination_device"), py::arg("first_owner_id"),
            py::arg("name"), py::arg("count") = 1, py::arg("validate") = true)
        .def(
            "GetOwnerContactWrench",
            [](const deme::DEMSolver& self, deme::bodyID_t owner_id, deme::bodyID_t count) {
                std::vector<float3> forces;
                std::vector<float3> torques;
                self.GetOwnerContactWrench(forces, torques, owner_id, count);
                return py::make_tuple(forces, torques);
            },
            R"doc(Return one reduced contact wrench per consecutive owner.

The returned tuple is ``(forces, torques)``; each vector is resized to ``count`` and follows consecutive owner order.
Force and torque are global-frame resultants, and torque is measured about DEME's current owner position. Torque
includes both force-generated moments and force-model-only torque such as rolling resistance. Owners without recorded
contact receive zero force and torque. This reads current dT force records and does not trigger contact detection or
force evaluation. ``first_owner_id`` selects the first owner in the range. Contact recording must remain enabled. A
zero count returns two empty vectors.)doc",
            py::arg("first_owner_id"), py::arg("count") = 1)
        .def(
            "GetOwnerContactWrenchToDevice",
            [](const deme::DEMSolver& self, std::uintptr_t forces, std::uintptr_t torques, size_t capacity, int device,
               deme::bodyID_t owner_id, deme::bodyID_t count) {
                self.GetOwnerContactWrenchToDevice(reinterpret_cast<float3*>(forces),
                                                   reinterpret_cast<float3*>(torques), capacity, device, owner_id,
                                                   count);
            },
            R"doc(Synchronously write one reduced contact wrench per consecutive owner to CUDA ``float3`` buffers.

Force and torque are global-frame resultants, and torque is measured about DEME's current owner position. Torque
includes both force-generated moments and force-model-only torque such as rolling resistance. ``force_destination``
and ``torque_destination`` are integer addresses of writable CUDA ``float3`` storage. ``capacity`` and ``count`` are
measured in owners, and ``first_owner_id`` selects the start of the range; owners without contact receive zero force
and torque. This reads current dT force records and does not trigger contact detection or force evaluation. Both
buffers belong to logical CUDA ``destination_device``; CUDA selects the available inter-device transfer route for
remote output. The call is synchronous, so both buffers may be consumed when it returns. Contact recording must remain
enabled. A zero count is a no-op.)doc",
            py::arg("force_destination"), py::arg("torque_destination"), py::arg("capacity"),
            py::arg("destination_device"), py::arg("first_owner_id"), py::arg("count") = 1)
        .def("GetTimeStepSize", &deme::DEMSolver::GetTimeStepSize, "Get the current time step size in simulation.")
        .def("SetCDUpdateFreq", &deme::DEMSolver::SetCDUpdateFreq,
             "Set the number of dT steps before it waits for a contact-pair update. After initialization, call "
             "UpdateSimParams() for this change to take effect in the workers.")
        .def("GetSimTime", &deme::DEMSolver::GetSimTime,
             "Get the simulation time passed since the start of simulation.")
        .def("SetSimTime", &deme::DEMSolver::SetSimTime,
             "Get the simulation time passed since the start of simulation.")
        .def("UpdateClumps", &deme::DEMSolver::UpdateClumps,
             "Transfer newly loaded clumps to the GPU-side in mid-simulation.")
        .def("SetAdaptiveTimeStepType", &deme::DEMSolver::SetAdaptiveTimeStepType,
             "Set the strategy for auto-adapting time step size. Currently, hertz_const computes one fixed setup-time "
             "timestep; max_vel and int_diff are accepted but not implemented.")
        .def("UseHertzConstTimeStep", &deme::DEMSolver::UseHertzConstTimeStep,
             "Use the setup-time Hertzian constant timestep estimate.")
        .def("SetIntegrator",
             static_cast<void (deme::DEMSolver::*)(const std::string&)>(&deme::DEMSolver::SetIntegrator),
             "Set the time integrator. After initialization, call UpdateSimParams() for this change to take effect.")
        .def("SetIntegrator",
             static_cast<void (deme::DEMSolver::*)(deme::TIME_INTEGRATOR)>(&deme::DEMSolver::SetIntegrator),
             "Set the time integrator. After initialization, call UpdateSimParams() for this change to take effect.")
        .def("GetInitStatus", &deme::DEMSolver::GetInitStatus, "Return whether this simulation system is initialized.")
        .def("GetJitStringSubs", &deme::DEMSolver::GetJitStringSubs,
             "Get the jitification string substitution laundary list. It is needed by some of this simulation system's "
             "friend classes.")
        .def("GetJitifyOptions", &deme::DEMSolver::GetJitifyOptions,
             "Get current jitification options. It is needed by some of this simulation system's friend classes.")
        .def("SetJitifyOptions", &deme::DEMSolver::SetJitifyOptions,
             "Set the jitification options. It is only needed by advanced users.")
        .def("SetCudaDebugSync", &deme::DEMSolver::SetCudaDebugSync,
             "Set the process-wide switch that synchronizes after each CUDA operation that enqueues stream work. "
             "Disable for normal asynchronous performance.",
             py::arg("enable") = true)
        .def("GetCudaDebugSync", &deme::DEMSolver::GetCudaDebugSync,
             "Return whether process-wide CUDA debug synchronization is enabled.")

        .def("SetInitBinSizeAsMultipleOfSmallestSphere", &deme::DEMSolver::SetInitBinSizeAsMultipleOfSmallestSphere,
             "Explicitly instruct the bin size (for contact detection) that the solver should use, as a multiple of "
             "the radius of the smallest sphere in simulation.")
        .def(
            "SetInitBinNumTarget", &deme::DEMSolver::SetInitBinNumTarget,
            "Set the target number of bins (for contact detection) at the start of the simulation upon initialization.")
        .def("InstructNumOwners", &deme::DEMSolver::InstructNumOwners,
             "Explicitly instruct the sizes for the arrays at initialization time. This is useful when the number of "
             "owners tends to change (especially gradually increase) frequently in the simulation.")
        .def("UseFrictionalHertzianModel", &deme::DEMSolver::UseFrictionalHertzianModel,
             py::return_value_policy::reference_internal,
             "Instruct the solver to use frictonal (history-based) Hertzian contact force model.")
        .def("UseFrictionlessHertzianModel", &deme::DEMSolver::UseFrictionlessHertzianModel,
             py::return_value_policy::reference_internal,
             "Instruct the solver to use frictonless Hertzian contact force model")
        .def("DefineContactForceModel", &deme::DEMSolver::DefineContactForceModel,
             py::return_value_policy::reference_internal,
             "Define a custom contact force model by a string. Returns a pointer to the force model in use.")
        .def("ReadContactForceModel", &deme::DEMSolver::ReadContactForceModel,
             py::return_value_policy::reference_internal,
             "Read user custom contact force model from a file (which by default should reside in "
             "kernel/DEMUserScripts). Returns a pointer to the force model in use.")

        .def("GetContactForceModel", &deme::DEMSolver::GetContactForceModel, "Get the current force model",
             py::return_value_policy::reference_internal)
        .def("SetSortContactPairs", &deme::DEMSolver::SetSortContactPairs,
             "Instruct the solver if contact pair arrays should be sorted (based on the types of contacts) before "
             "usage.")
        .def(
            "SetJitifyClumpTemplates", &deme::DEMSolver::SetJitifyClumpTemplates,
            "Instruct the solver to rearrange and consolidate clump templates information, then jitify it into GPU "
            "kernels (if set to true), rather than using flattened sphere component configuration arrays whose entries "
            "are associated with individual spheres.",
            py::arg("use") = true)
        .def("DisableJitifyClumpTemplates", &deme::DEMSolver::DisableJitifyClumpTemplates,
             "Disable jitification of clump templates (use flattened arrays instead).")
        .def("SetJitifyMassProperties", &deme::DEMSolver::SetJitifyMassProperties,
             "Instruct the solver to rearrange and consolidate mass property information (for all owner types), then "
             "jitify it into GPU kernels (if set to true), rather than using flattened mass property arrays whose "
             "entries are associated with individual owners.",
             py::arg("use") = true)
        .def("DisableJitifyMassProperties", &deme::DEMSolver::DisableJitifyMassProperties,
             "Disable jitification of mass properties (use flattened arrays instead).")
        .def("SetExpandFactor", &deme::DEMSolver::SetExpandFactor,
             "(Explicitly) set the amount by which the radii of the spheres (and the thickness of the boundaries) are "
             "expanded for the purpose of contact detection (safe, and creates false positives). If fix is set to "
             "true, then this expand factor does not change even if the user uses variable time step size. After "
             "initialization, call UpdateSimParams() for this change to take effect in the workers.",
             py::arg("beta"), py::arg("fix") = true)
        .def("SetMeshUniversalContact", &deme::DEMSolver::SetMeshUniversalContact,
             "Set whether mesh-mesh contacts should be universally detected. Set to false to speedup simulation if "
             "meshes are not expected to have contacts. Default is false.",
             py::arg("use") = true)
        .def("SetDEME2MeshBehavior", &deme::DEMSolver::SetDEME2MeshBehavior,
             "When enabled, assign every triangle of subsequently loaded meshes to its own patch. Disabling restores "
             "the normal behavior of leaving new meshes' patch assignments unchanged.",
             py::arg("use") = true)
        .def("SetPersistentContact", &deme::DEMSolver::SetPersistentContact,
             "Set whether the solver should expect the user to mark certain contacts as persistent across kT updates. "
             "Set this to true if you later will call MarkPersistentContact series of methods.",
             py::arg("use") = true)
        .def("SetSimplePatchCombination", &deme::DEMSolver::SetSimplePatchCombination,
             "Use simple patch-ID based triangle contact combination. Disable to opt into connected-component flooding.",
             py::arg("use") = true)
        .def("SetStablePatchIslandIDs", &deme::DEMSolver::SetStablePatchIslandIDs,
             "Stabilize flooded patch-island IDs across contact-detection steps.", py::arg("use") = true)
        .def("SetMeshParticlesLowPoly", &deme::DEMSolver::SetMeshParticlesLowPoly,
             "Declare that all meshed particles have a low polygon count (e.g. box, tetrahedron). When enabled, the "
             "per-triangle maxTriTriPenetration array is neither computed, transferred to kT, nor used to inflate "
             "contact-detection margins, saving compute time. Toggle this on only when you are confident that no "
             "triangle from one mesh will ever be completely submerged inside another mesh.",
             py::arg("use") = true)

        .def("SetExpandSafetyType", &deme::DEMSolver::SetExpandSafetyType,
             "Select the contact-margin velocity strategy. After initialization, call UpdateSimParams() for this "
             "change to take effect in the workers.")
        .def("SetExpandSafetyAdder", &deme::DEMSolver::SetExpandSafetyAdder,
             "Set a `base' velocity, which we will always add to our estimated maximum system velocity, when deriving "
             "the thickness of the contact `safety' margin. After initialization, call UpdateSimParams() for this "
             "change to take effect in the workers.")
        .def("SetUseAngularVelocityMargin", &deme::DEMSolver::SetUseAngularVelocityMargin,
             "Control whether angular velocity contributes to the contact detection margin.")
        .def("SetMaxSphereInBin", &deme::DEMSolver::SetMaxSphereInBin,
             "Used to force the solver to error out when there are too many spheres in a bin. A huge number can be "
             "used to discourage this error type")
        .def("SetMaxTriangleInBin", &deme::DEMSolver::SetMaxTriangleInBin,
             "Used to force the solver to error out when there are too many spheres in a bin. A huge number can be "
             "used to discourage this error type")

        .def(
            "SetErrorOutVelocity", &deme::DEMSolver::SetErrorOutVelocity,
            "Set the velocity which when exceeded, the solver errors out. A huge number can be used to discourage this "
            "error type. Defaulted to 1e3.")
        .def("SetErrorOutAngularVelocity", &deme::DEMSolver::SetErrorOutAngularVelocity,
             "Set the angular velocity which when exceeded, the solver errors out. A huge number can be used to "
             "discourage this error type. Defaulted to 1e4.")
        .def("SetErrorOutAvgContacts", &deme::DEMSolver::SetErrorOutAvgContacts,
             "Set the average number of contacts a sphere has, before the solver errors out. A huge number can be used "
             "to discourage this error type. Defaulted to 100")
        .def("SetTriTriPenetration", &deme::DEMSolver::SetTriTriPenetration,
             "Manually set the current triangle-triangle penetration value in dT. This allows the user to directly "
             "control the maxTriTriPenetration value which will ONLY be used in the NEXT contact detection run in kT.")
        .def("SetMaxTriTriPenetration", &deme::DEMSolver::SetMaxTriTriPenetration,
             "Set the maximum allowed triangle-triangle penetration used as the margin added in kT contact detection. "
             "This value caps the penetration margin added to prevent excessively large values.")
        .def("SetTriTriContactRejectionRatio", &deme::DEMSolver::SetTriTriContactRejectionRatio,
             "Set the ratio threshold for rejecting suspicious tri-tri contacts. A contact is rejected when the "
             "penetration depth exceeds this fraction of the contact-point-to-mesh-center distance for either mesh "
             "involved. A negative value disables the guard entirely.")
        .def("GetAvgSphContacts", &deme::DEMSolver::GetAvgSphContacts,
             "Get the current number of contacts each sphere has")
        .def("UseAdaptiveBinSize", &deme::DEMSolver::UseAdaptiveBinSize,
             "Enable or disable adaptive bin size. After initialization, call UpdateSimParams() for this change to "
             "take effect in the workers.", py::arg("use") = true)
        .def("DisableAdaptiveBinSize", &deme::DEMSolver::DisableAdaptiveBinSize,
             "Disable adaptive bin size. After initialization, call UpdateSimParams() for this change to take effect "
             "in the workers.")
        .def("UseAdaptiveUpdateFreq", &deme::DEMSolver::UseAdaptiveUpdateFreq,
             "Enable or disable adaptive contact-update frequency. After initialization, call UpdateSimParams() for "
             "this change to take effect in the workers.", py::arg("use") = true)
        .def("DisableAdaptiveUpdateFreq", &deme::DEMSolver::DisableAdaptiveUpdateFreq,
             "Disable adaptive contact-update frequency. After initialization, call UpdateSimParams() for this "
             "change to take effect in the workers.")
        .def("SetAdaptiveBinSizeDelaySteps", &deme::DEMSolver::SetAdaptiveBinSizeDelaySteps,
             "Adjust how frequently kT updates the bin size. After initialization, call UpdateSimParams().")
        .def("SetAdaptiveBinSizeMaxRate", &deme::DEMSolver::SetAdaptiveBinSizeMaxRate,
             "Set the maximum bin-size change rate. After initialization, call UpdateSimParams().")
        .def("SetAdaptiveBinSizeAcc", &deme::DEMSolver::SetAdaptiveBinSizeAcc,
             "Set the bin-size adjustment acceleration. After initialization, call UpdateSimParams().")
        .def("SetAdaptiveBinSizeUpperProactivity", &deme::DEMSolver::SetAdaptiveBinSizeUpperProactivity,
             "Set how proactive the solver is in avoiding the bin being too big (leading to too many geometries in a "
             "bin). After initialization, call UpdateSimParams().")
        .def(
            "SetAdaptiveBinSizeLowerProactivity", &deme::DEMSolver::SetAdaptiveBinSizeLowerProactivity,
            "Set how proactive the solver is in avoiding undersized bins. After initialization, call UpdateSimParams().")
        .def("GetBinSize", &deme::DEMSolver::GetBinSize,
             "Get the current bin (for contact detection) size. Must be called from synchronized stance.")
        .def("GetBinNum", &deme::DEMSolver::GetBinNum,
             "Get the current number of bins (for contact detection). Must be called from synchronized stance.")
        .def("SetCDMaxUpdateFreq", &deme::DEMSolver::SetCDMaxUpdateFreq,
             "Set the upper bound of kT update frequency. After initialization, call UpdateSimParams().")
        .def("SetCDNumStepsMaxDriftAheadOfAvg", &deme::DEMSolver::SetCDNumStepsMaxDriftAheadOfAvg,
             "Set dT's drift allowance above average. After initialization, call UpdateSimParams().")
        .def("SetCDNumStepsMaxDriftMultipleOfAvg", &deme::DEMSolver::SetCDNumStepsMaxDriftMultipleOfAvg,
             "Set dT's maximum-drift multiplier. After initialization, call UpdateSimParams().")
        .def("SetCDNumStepsMaxDriftHistorySize", &deme::DEMSolver::SetCDNumStepsMaxDriftHistorySize,
             "Set how many past kinematic-worker updates calibrate the maximum future drift limit. The default is "
             "recommended for normal use. After initialization, call UpdateSimParams().",
             py::arg("n"))
        .def("GetUpdateFreq", &deme::DEMSolver::GetUpdateFreq, "Get the current update frequency used by the solver.")
        .def("SetForceCalcThreadsPerBlock", &deme::DEMSolver::SetForceCalcThreadsPerBlock,
             "Set the number of threads per block in force calculation (default 256).")
        .def("Duplicate",
             static_cast<std::shared_ptr<deme::DEMMaterial> (deme::DEMSolver::*)(
                 const std::shared_ptr<deme::DEMMaterial>&)>(&deme::DEMSolver::Duplicate),
             "Duplicate a material that is loaded into the system.")
        .def("Duplicate",
             static_cast<std::shared_ptr<deme::DEMClumpTemplate> (deme::DEMSolver::*)(
                 const std::shared_ptr<deme::DEMClumpTemplate>&)>(&deme::DEMSolver::Duplicate),
             "Duplicate a clump template that is loaded into the system.")
        .def("Duplicate",
             static_cast<std::shared_ptr<deme::DEMClumpBatch> (deme::DEMSolver::*)(
                 const std::shared_ptr<deme::DEMClumpBatch>&)>(&deme::DEMSolver::Duplicate),
             "Duplicate a batch of clumps that is loaded into the system.")
        .def("AddExternalObject", &deme::DEMSolver::AddExternalObject,
             "Add an analytical object to the simulation system.")
        .def(
            "SetOutputContent",
            static_cast<void (deme::DEMSolver::*)(const std::vector<std::string>&)>(&deme::DEMSolver::SetOutputContent),
            "Specify the information that needs to go into the clump or sphere output files.")
        .def("SetMeshOutputFormat",
             static_cast<void (deme::DEMSolver::*)(const std::string&)>(&deme::DEMSolver::SetMeshOutputFormat),
             "Specify the output file format of meshes.")
        .def("SetMeshOutputContent",
             static_cast<void (deme::DEMSolver::*)(const std::vector<std::string>&)>(
                 &deme::DEMSolver::SetMeshOutputContent),
             "Specify per-triangle fields to include in mesh VTK output.", py::arg("content"))
        .def("SetMeshOutputContent",
             static_cast<void (deme::DEMSolver::*)(deme::MESH_OUTPUT_CONTENT)>(
                 &deme::DEMSolver::SetMeshOutputContent),
             "Specify one per-triangle field to include in mesh VTK output.", py::arg("content"))
        .def("EnableMeshPatchColorOutput", &deme::DEMSolver::EnableMeshPatchColorOutput,
             "Enable or disable patch color metadata in PLY mesh output.", py::arg("enable") = true)
        .def("SetContactOutputContent",
             static_cast<void (deme::DEMSolver::*)(const std::vector<std::string>&)>(
                 &deme::DEMSolver::SetContactOutputContent),
             "Specify the information that needs to go into the contact pair output files.")
        .def("SetContactOutputFormat",
             static_cast<void (deme::DEMSolver::*)(const std::string&)>(&deme::DEMSolver::SetContactOutputFormat),
             "Specify the file format of contact pairs.")
        .def("EnableOwnerWildcardOutput", &deme::DEMSolver::EnableOwnerWildcardOutput,
             "Enable or disable owner wildcard output.", py::arg("enable") = true)
        .def("EnableContactWildcardOutput", &deme::DEMSolver::EnableContactWildcardOutput,
             "Enable or disable contact wildcard output.", py::arg("enable") = true)
        .def("SetAllowIntraCombinedOwnerContacts", &deme::DEMSolver::SetAllowIntraCombinedOwnerContacts,
             "Allow or suppress contacts among owners belonging to the same combined owner group.",
             py::arg("allow") = true)
        .def("SetVerbosity", static_cast<void (deme::DEMSolver::*)(const std::string&)>(&deme::DEMSolver::SetVerbosity),
             "Set the verbosity level of the solver. Select from 'QUIET', 'ERROR', 'WARNING', 'INFO', 'METRIC' or "
             "'DEBUG'. Recommend 'INFO'.")
        .def("AddKernelInclude", &deme::DEMSolver::AddKernelInclude,
             "Add a library that the kernels will be compiled with (so that the user can use the provided methods in "
             "their customized code, like force model).")
        .def("SetKernelInclude", &deme::DEMSolver::SetKernelInclude,
             "Set the kernels' headers' extra include lines. Useful for customization.")
        .def("RemoveKernelInclude", &deme::DEMSolver::RemoveKernelInclude,
             "Remove all extra libraries that the kernels `include' in their headers.")

        .def("LoadMaterial",
             static_cast<std::shared_ptr<deme::DEMMaterial> (deme::DEMSolver::*)(
                 const std::unordered_map<std::string, float>&)>(&deme::DEMSolver::LoadMaterial),
             "Load materials properties (Young's modulus, Poisson's ratio...) into the system.")
        .def("LoadMaterial",
             static_cast<std::shared_ptr<deme::DEMMaterial> (deme::DEMSolver::*)(deme::DEMMaterial&)>(
                 &deme::DEMSolver::LoadMaterial),
             "Load materials properties into the system.")
        .def("InstructBoxDomainDimension",
             static_cast<void (deme::DEMSolver::*)(float, float, float, const std::string&)>(
                 &deme::DEMSolver::InstructBoxDomainDimension),
             "Set the Box Domain Dimension", py::arg("x"), py::arg("y"), py::arg("z"), py::arg("dir_exact") = "none")
        .def("InstructBoxDomainDimension",
             static_cast<void (deme::DEMSolver::*)(const std::pair<float, float>&, const std::pair<float, float>&,
                                                   const std::pair<float, float>&, const std::string& dir_exact)>(
                 &deme::DEMSolver::InstructBoxDomainDimension),
             "Set the span of the Box Domain", py::arg("x"), py::arg("y"), py::arg("z"), py::arg("dir_exact") = "none")
        .def("LoadMeshType",
             static_cast<std::shared_ptr<deme::DEMMesh> (deme::DEMSolver::*)(
                 const std::string&, const std::shared_ptr<deme::DEMMaterial>&, bool, bool)>(
                 &deme::DEMSolver::LoadMeshType),
             "Load a mesh template into cache so it can be instantiated repeatedly.", py::arg("filename"),
             py::arg("mat"), py::arg("load_normals") = true, py::arg("load_uv") = false)
        .def("LoadMeshType",
             static_cast<std::shared_ptr<deme::DEMMesh> (deme::DEMSolver::*)(const std::string&, bool, bool)>(
                 &deme::DEMSolver::LoadMeshType),
             "Load a mesh template into cache (without explicitly assigning material here).", py::arg("filename"),
             py::arg("load_normals") = true, py::arg("load_uv") = false)
        .def("LoadMeshType",
             static_cast<std::shared_ptr<deme::DEMMesh> (deme::DEMSolver::*)(deme::DEMMesh&)>(
                 &deme::DEMSolver::LoadMeshType),
             "Load a user-constructed mesh object as a reusable template.", py::arg("mesh"))
        .def("AddMeshFromTemplate",
             static_cast<std::shared_ptr<deme::DEMMesh> (deme::DEMSolver::*)(
                 const std::shared_ptr<deme::DEMMesh>&, const float3&)>(&deme::DEMSolver::AddMeshFromTemplate),
             "Instantiate a mesh from a cached template at the requested initial position.", py::arg("mesh_template"),
             py::arg("init_pos") = make_float3(0))
        .def("InstructBoxDomainBoundingBC", &deme::DEMSolver::InstructBoxDomainBoundingBC,
             "Instruct if and how we should add boundaries to the simulation world upon initialization. Choose between "
             "`none', `all' (add 6 boundary planes) and `top_open' (add 5 boundary planes and leave the z-directon top "
             "open). Also specifies the material that should be assigned to those bounding boundaries.")
        .def("SetMaterialPropertyPair", &deme::DEMSolver::SetMaterialPropertyPair,
             "Set the value for a material property that by nature involves a pair of a materials (e.g. friction "
             "coefficient).")
        .def("AddBCPlane",
             static_cast<std::shared_ptr<deme::DEMExternObj> (deme::DEMSolver::*)(
                 const std::vector<float>&, const std::vector<float>&, const std::shared_ptr<deme::DEMMaterial>&)>(
                 &deme::DEMSolver::AddBCPlane),
             "Add an analytical plane to the simulation.")
        .def("Track", (&deme::DEMSolver::PythonTrack),
             "Create a DEMTracker to allow direct control/modification/query to this external object/batch of "
             "clumps/triangle mesh object.")
        .def("Track",
             static_cast<std::shared_ptr<deme::DEMTracker> (deme::DEMSolver::*)(
                 const std::shared_ptr<deme::DEMCombinedInstances>&)>(&deme::DEMSolver::Track),
             "Create a single tracker that tracks all member owners in a combined-instances batch.",
             py::arg("combined_inst"))
        .def("AddWavefrontMeshObject",
             static_cast<std::shared_ptr<deme::DEMMeshConnected> (deme::DEMSolver::*)(
                 const std::string&, const std::shared_ptr<deme::DEMMaterial>&, bool, bool)>(
                 &deme::DEMSolver::AddWavefrontMeshObject),
             "Load a mesh-represented object.", py::arg("filename"), py::arg("mat"), py::arg("load_normals") = true,
             py::arg("load_uv") = false)
        .def("AddWavefrontMeshObject",
             static_cast<std::shared_ptr<deme::DEMMeshConnected> (deme::DEMSolver::*)(deme::DEMMeshConnected&)>(
                 &deme::DEMSolver::AddWavefrontMeshObject),
             "Load a mesh-represented object.")
        .def("AddWavefrontMeshObject",
             static_cast<std::shared_ptr<deme::DEMMeshConnected> (deme::DEMSolver::*)(
                 const std::string& filename, bool, bool)>(&deme::DEMSolver::AddWavefrontMeshObject),
             "Load a mesh-represented object.", py::arg("filename"), py::arg("load_normals") = true,
             py::arg("load_uv") = false)
        .def("AddShellMesh",
             static_cast<std::shared_ptr<deme::DEMMesh> (deme::DEMSolver::*)(deme::DEMMesh&, float)>(
                 &deme::DEMSolver::AddShellMesh),
             "Load a finite-thickness shell mesh from a user-constructed mesh object.", py::arg("mesh"),
             py::arg("shell_thickness"))
        .def("AddWavefrontShellObject",
             static_cast<std::shared_ptr<deme::DEMMesh> (deme::DEMSolver::*)(
                 const std::string&, const std::shared_ptr<deme::DEMMaterial>&, float, bool, bool)>(
                 &deme::DEMSolver::AddWavefrontShellObject),
             "Load a finite-thickness shell mesh from a mesh file.", py::arg("filename"), py::arg("mat"),
             py::arg("shell_thickness"), py::arg("load_normals") = true, py::arg("load_uv") = false)
        .def("AddWavefrontShellObject",
             static_cast<std::shared_ptr<deme::DEMMesh> (deme::DEMSolver::*)(const std::string&, float, bool, bool)>(
                 &deme::DEMSolver::AddWavefrontShellObject),
             "Load a finite-thickness shell mesh from a mesh file without assigning material immediately.",
             py::arg("filename"), py::arg("shell_thickness"), py::arg("load_normals") = true,
             py::arg("load_uv") = false)
        .def("LoadClumpType",
             static_cast<std::shared_ptr<deme::DEMClumpTemplate> (deme::DEMSolver::*)(
                 float, const std::vector<float>&, const std::string, const std::shared_ptr<deme::DEMMaterial>&)>(
                 &deme::DEMSolver::LoadClumpType),
             "Load a clump type into the API-level cache")
        .def("LoadClumpType",
             static_cast<std::shared_ptr<deme::DEMClumpTemplate> (deme::DEMSolver::*)(
                 float mass, const std::vector<float>&, const std::vector<float>&,
                 const std::vector<std::vector<float>>&, const std::shared_ptr<deme::DEMMaterial>&)>(
                 &deme::DEMSolver::LoadClumpType),
             "Load a clump type into the API-level cache")
        .def("LoadClumpType",
             static_cast<std::shared_ptr<deme::DEMClumpTemplate> (deme::DEMSolver::*)(deme::DEMClumpTemplate&)>(
                 &deme::DEMSolver::LoadClumpType),
             "Load a clump type into the API-level cache")
        .def("LoadClumpType",
             static_cast<std::shared_ptr<deme::DEMClumpTemplate> (deme::DEMSolver::*)(
                 float, const std::vector<float>&, const std::string,
                 const std::vector<std::shared_ptr<deme::DEMMaterial>>&)>(&deme::DEMSolver::LoadClumpType),
             "Load a clump type into the API-level cache")
        .def("LoadClumpType",
             static_cast<std::shared_ptr<deme::DEMClumpTemplate> (deme::DEMSolver::*)(
                 float, const std::vector<float>&, const std::string, const std::shared_ptr<deme::DEMMaterial>&)>(
                 &deme::DEMSolver::LoadClumpType),
             "Load a clump type into the API-level cache")

        .def("GetOwnerContactClumps", &deme::DEMSolver::GetOwnerContactClumps,
             "Get the clumps that are in contact with this owner as a vector.")
        .def("GetOwnerPosition", &deme::DEMSolver::GetOwnerPosition, "Get position of n consecutive owners.",
             py::arg("ownerID"), py::arg("n") = 1)
        .def("GetClumpPositionsHandover", &deme::DEMSolver::GetClumpPositionsHandover,
             "Get all clump-owner center positions ordered by owner ID.")
        .def("GetOwnerAngVel", &deme::DEMSolver::GetOwnerAngVel, "Get angular velocity of n consecutive owners.",
             py::arg("ownerID"), py::arg("n") = 1)
        .def("GetOwnerOriQ", &deme::DEMSolver::GetOwnerOriQ, "Get quaternion of n consecutive owners.",
             py::arg("ownerID"), py::arg("n") = 1)
        .def("GetOwnerVelocity", &deme::DEMSolver::GetOwnerVelocity, "Get velocity of n consecutive owners.",
             py::arg("ownerID"), py::arg("n") = 1)
        .def("GetOwnerAcc", &deme::DEMSolver::GetOwnerAcc, "Get the acceleration of n consecutive owners.",
             py::arg("ownerID"), py::arg("n") = 1)
        .def("GetOwnerAngAcc", &deme::DEMSolver::GetOwnerAngAcc,
             "Get the angular acceleration of n consecutive owners.", py::arg("ownerID"), py::arg("n") = 1)
        .def("GetOwnerFamily", &deme::DEMSolver::GetOwnerFamily, "Get the family number of n consecutive owners.",
             py::arg("ownerID"), py::arg("n") = 1)
        .def("RequestContactUpdate", &deme::DEMSolver::RequestContactUpdate,
             "Request an immediate contact detection update.")
        .def("SetTrianglePVTrackingOwners", &deme::DEMSolver::SetTrianglePVTrackingOwners,
             "Enable per-triangle P/V/PxV tracking for the specified mesh owners.")
        .def("DisableTrianglePVTracking", &deme::DEMSolver::DisableTrianglePVTracking,
             "Disable per-triangle P/V/PxV tracking and clear tracking state.")
        .def("GetTrackedOwnerTrianglePV",
             [](deme::DEMSolver& solver, deme::bodyID_t ownerID, bool reset_window) {
                 std::vector<float> avgP;
                 std::vector<float> avgV;
                 std::vector<float> avgPV;
                 bool ok = solver.GetTrackedOwnerTrianglePV(ownerID, avgP, avgV, avgPV, reset_window);
                 return py::make_tuple(ok, avgP, avgV, avgPV);
             },
             "Get frame-window averaged per-triangle P, V and P*V for one tracked owner.", py::arg("ownerID"),
             py::arg("reset_window") = true)
        .def("EnableMeshWearModel", &deme::DEMSolver::EnableMeshWearModel,
             "Enable a mesh wear model driven by per-triangle P*V.", py::arg("ownerID"), py::arg("wear_rate"),
             py::arg("update_interval"), py::arg("start_time") = 0.0, py::arg("end_time") = -1.0,
             py::arg("normal_sign") = -1.0f)
        .def("DisableMeshWearModel", &deme::DEMSolver::DisableMeshWearModel,
             "Disable mesh wear model for one owner.")
        .def("DisableAllMeshWearModels", &deme::DEMSolver::DisableAllMeshWearModels,
             "Disable all active mesh wear models.")
        .def("FlushMeshWearModels", &deme::DEMSolver::FlushMeshWearModels,
             "Force-apply any pending wear deformation immediately.")
        .def("GetOwnerMass", &deme::DEMSolver::GetOwnerMass, "Get the mass of n consecutive owners.",
             py::arg("ownerID"), py::arg("n") = 1)
        .def("GetOwnerMOI", &deme::DEMSolver::GetOwnerMOI,
             "Get the moment of inertia (in principal axis frame) of n consecutive owners.", py::arg("ownerID"),
             py::arg("n") = 1)

        .def("SetOwnerPosition", &deme::DEMSolver::SetOwnerPosition,
             "Set position of consecutive owners starting from ownerID, based on input position vector. N (the size of "
             "the input vector) elements will be modified.")
        .def("SetOwnerAngVel", &deme::DEMSolver::SetOwnerAngVel,
             "Set angular velocity of consecutive owners starting from ownerID, based on input angular velocity "
             "vector. N (the size of the input vector) elements will be modified.")
        .def("SetOwnerVelocity", &deme::DEMSolver::SetOwnerVelocity,
             "Set velocity of consecutive owners starting from ownerID, based on input velocity vector. N (the size of "
             "the input vector) elements will be modified.")
        .def("SetOwnerOriQ", &deme::DEMSolver::SetOwnerOriQ,
             "Set quaternion of consecutive owners starting from ownerID, based on input quaternion vector. N (the "
             "size of the input vector) elements will be modified.")
        .def("SetOwnerFamily", &deme::DEMSolver::SetOwnerFamily, "Set the family number of consecutive owners.",
             py::arg("ownerID"), py::arg("fam"), py::arg("n") = 1)

        .def("SetTriNodeRelPos", &deme::DEMSolver::SetTriNodeRelPos,
             "Rewrite the relative positions of the flattened triangle soup.")
        .def("UpdateTriNodeRelPos", &deme::DEMSolver::UpdateTriNodeRelPos,
             "Update the relative positions of the flattened triangle soup.")
        .def("GetCachedMesh", &deme::DEMSolver::GetCachedMesh, "Get a handle for the mesh this tracker is tracking.")
        .def("GetMeshNodesGlobal", &deme::DEMSolver::GetMeshNodesGlobal,
             "Get the current locations of all the nodes in the mesh being tracked.")

        .def("GetClumpContacts",
             static_cast<std::vector<std::pair<deme::bodyID_t, deme::bodyID_t>> (deme::DEMSolver::*)() const>(
                 &deme::DEMSolver::GetClumpContacts),
             "Get all clump--clump contact ID pairs in the simulation system. Note all GetContact-like methods reports "
             "potential contacts (not necessarily confirmed contacts), meaning they are similar to what "
             "WriteContactFileIncludingPotentialPairs does, not what WriteContactFile does.")
        .def("GetClumpContacts",
             static_cast<std::vector<std::pair<deme::bodyID_t, deme::bodyID_t>> (deme::DEMSolver::*)(
                 const std::set<deme::family_t>&) const>(&deme::DEMSolver::GetClumpContacts),
             "Get all clump--clump contact ID pairs in the simulation system. Note all GetContact-like methods reports "
             "potential contacts (not necessarily confirmed contacts), meaning they are similar to what "
             "WriteContactFileIncludingPotentialPairs does, not what WriteContactFile does. Use the argument to "
             "include only these families in the output.")
        .def("GetClumpContacts",
             static_cast<std::vector<std::pair<deme::bodyID_t, deme::bodyID_t>> (deme::DEMSolver::*)(
                 std::vector<std::pair<deme::family_t, deme::family_t>>&) const>(&deme::DEMSolver::GetClumpContacts),
             "Get all clump--clump contact ID pairs in the simulation system. Note all GetContact-like methods reports "
             "potential contacts (not necessarily confirmed contacts), meaning they are similar to what "
             "WriteContactFileIncludingPotentialPairs does, not what WriteContactFile does.")

        .def("GetContacts",
             static_cast<std::vector<std::pair<deme::bodyID_t, deme::bodyID_t>> (deme::DEMSolver::*)() const>(
                 &deme::DEMSolver::GetContacts),
             "Get all contact ID pairs in the simulation system. Note all GetContact-like methods reports potential "
             "contacts (not necessarily confirmed contacts), meaning they are similar to what "
             "WriteContactFileIncludingPotentialPairs does, not what WriteContactFile does.")
        .def("GetContacts",
             static_cast<std::vector<std::pair<deme::bodyID_t, deme::bodyID_t>> (deme::DEMSolver::*)(
                 const std::set<deme::family_t>&) const>(&deme::DEMSolver::GetContacts),
             "Get all contact ID pairs in the simulation system. Note all GetContact-like methods reports potential "
             "contacts (not necessarily confirmed contacts), meaning they are similar to what "
             "WriteContactFileIncludingPotentialPairs does, not what WriteContactFile does. Use the argument to "
             "include only these families in the output.")
        .def("GetContacts",
             static_cast<std::vector<std::pair<deme::bodyID_t, deme::bodyID_t>> (deme::DEMSolver::*)(
                 std::vector<std::pair<deme::family_t, deme::family_t>>&) const>(&deme::DEMSolver::GetContacts),
             "Get all contact ID pairs in the simulation system. Note all GetContact-like methods reports potential "
             "contacts (not necessarily confirmed contacts), meaning they are similar to what "
             "WriteContactFileIncludingPotentialPairs does, not what WriteContactFile does.")

        .def("GetContactDetailedInfo", &deme::DEMSolver::GetContactDetailedInfo,
             "Get all contact pairs' detailed information (actual content based on the setting with "
             "SetContactOutputContent; default are owner IDs, contact point location, contact force, and associated "
             "wildcard values) in the simulation system. Note all GetContact-like methods reports potential contacts "
             "(not necessarily confirmed contacts), meaning they are similar to what "
             "WriteContactFileIncludingPotentialPairs does, not what WriteContactFile does.",
             py::arg("force_thres") = -1.0)

        .def("GetHostMemUsageDynamic", &deme::DEMSolver::GetHostMemUsageDynamic,
             "Get the host memory usage (in bytes) on dT.")
        .def("GetDeviceMemUsageDynamic", &deme::DEMSolver::GetDeviceMemUsageDynamic,
             "Get the device memory usage (in bytes) on dT.")
        .def("GetHostMemUsageKinematic", &deme::DEMSolver::GetHostMemUsageKinematic,
             "Get the host memory usage (in bytes) on kT.")
        .def("GetDeviceMemUsageKinematic", &deme::DEMSolver::GetDeviceMemUsageKinematic,
             "Get the device memory usage (in bytes) on kT.")
        .def("ShowMemStats", &deme::DEMSolver::ShowMemStats, "Print the current memory usage in pretty format.")

        .def("AddClumps",
             static_cast<std::shared_ptr<deme::DEMClumpBatch> (deme::DEMSolver::*)(deme::DEMClumpBatch&)>(
                 &deme::DEMSolver::AddClumps),
             "Load input clumps (topology types and initial locations) on a per-pair basis. Note that the initial "
             "location means the location of the clumps' CoM coordinates in the global frame.")
        .def("AddClumps",
             static_cast<std::shared_ptr<deme::DEMClumpBatch> (deme::DEMSolver::*)(
                 std::shared_ptr<deme::DEMClumpTemplate>&, const std::vector<std::vector<float>>&)>(
                 &deme::DEMSolver::AddClumps),
             "Load input clumps (topology types and initial locations) on a per-pair basis. Note that the initial "
             "location means the location of the clumps' CoM coordinates in the global frame.")
        .def("AddClumps",
             static_cast<std::shared_ptr<deme::DEMClumpBatch> (deme::DEMSolver::*)(
                 const std::vector<std::shared_ptr<deme::DEMClumpTemplate>>&, const std::vector<std::vector<float>>&)>(
                 &deme::DEMSolver::AddClumps),
             "Load input clumps (topology types and initial locations) on a per-pair basis. Note that the initial "
             "location means the location of the clumps' CoM coordinates in the global frame.")

        .def("SetFamilyFixed", &deme::DEMSolver::SetFamilyFixed, "Mark all entities in this family to be fixed.")

        .def("SetFamilyPrescribedAngVel",
             static_cast<void (deme::DEMSolver::*)(unsigned int, const std::string&, const std::string&,
                                                   const std::string&, bool, const std::string&)>(
                 &deme::DEMSolver::SetFamilyPrescribedAngVel),
             "Set the prescribed angular velocity to all entities in a family. If dictate is set to true, then this "
             "family will not be influenced by the force exerted from other simulation entites (both linear and "
             "rotational motions). If false, only specified components (that is, not specified with 'none') will not "
             "be influenced by the force exerted from other simulation entites.",
             py::arg("ID"), py::arg("velX"), py::arg("velY"), py::arg("velZ"), py::arg("dictate") = true,
             py::arg("pre") = "none")
        .def("SetFamilyPrescribedAngVel",
             static_cast<void (deme::DEMSolver::*)(unsigned int)>(&deme::DEMSolver::SetFamilyPrescribedAngVel),
             "Let the angular velocities of all entites in this family always keep `as is', and not influenced by the "
             "force exerted from other simulation entites.")
        .def("SetFamilyPrescribedAngVelX", &deme::DEMSolver::SetFamilyPrescribedAngVelX,
             "Let the X component of the angular velocities of all entites in this family always keep `as is', and not "
             "influenced by the force exerted from other simulation entites.")
        .def("SetFamilyPrescribedAngVelY", &deme::DEMSolver::SetFamilyPrescribedAngVelY,
             "Let the X component of the angular velocities of all entites in this family always keep `as is', and not "
             "influenced by the force exerted from other simulation entites.")
        .def("SetFamilyPrescribedAngVelZ", &deme::DEMSolver::SetFamilyPrescribedAngVelZ,
             "Let the X component of the angular velocities of all entites in this family always keep `as is', and not "
             "influenced by the force exerted from other simulation entites.")

        .def("SetFamilyPrescribedLinVel",
             static_cast<void (deme::DEMSolver::*)(unsigned int, const std::string&, const std::string&,
                                                   const std::string&, bool, const std::string&)>(
                 &deme::DEMSolver::SetFamilyPrescribedLinVel),
             "Set the prescribed linear velocity to all entities in a family. If dictate is set to true, then this "
             "family will not be influenced by the force exerted from other simulation entites (both linear and "
             "rotational motions). If false, only specified components (that is, not specified with 'none') will not "
             "be influenced by the force exerted from other simulation entites.",
             py::arg("ID"), py::arg("velX"), py::arg("velY"), py::arg("velZ"), py::arg("dictate") = true,
             py::arg("pre") = "none")
        .def("SetFamilyPrescribedLinVel",
             static_cast<void (deme::DEMSolver::*)(unsigned int)>(&deme::DEMSolver::SetFamilyPrescribedLinVel),
             "Let the linear velocities of all entites in this family always keep `as is', and not influenced by the "
             "force exerted from other simulation entites.")
        .def("SetFamilyPrescribedLinVelX", &deme::DEMSolver::SetFamilyPrescribedLinVelX,
             "Let the X component of the linear velocities of all entites in this family always keep `as is', and not "
             "influenced by the force exerted from other simulation entites.")
        .def("SetFamilyPrescribedLinVelY", &deme::DEMSolver::SetFamilyPrescribedLinVelY,
             "Let the Y component of the linear velocities of all entites in this family always keep `as is', and not "
             "influenced by the force exerted from other simulation entites.")
        .def("SetFamilyPrescribedLinVelZ", &deme::DEMSolver::SetFamilyPrescribedLinVelZ,
             "Let the Z component of the linear velocities of all entites in this family always keep `as is', and not "
             "influenced by the force exerted from other simulation entites.")

        .def("SetFamilyPrescribedPosition",
             static_cast<void (deme::DEMSolver::*)(unsigned int, const std::string&, const std::string&,
                                                   const std::string&, bool, const std::string&)>(
                 &deme::DEMSolver::SetFamilyPrescribedPosition),
             "Keep the positions of all entites in this family to remain exactly the user-specified values.",
             py::arg("ID"), py::arg("X"), py::arg("Y"), py::arg("Z"), py::arg("dictate") = true,
             py::arg("pre") = "none")
        .def("SetFamilyPrescribedPosition",
             static_cast<void (deme::DEMSolver::*)(unsigned int)>(&deme::DEMSolver::SetFamilyPrescribedPosition),
             "Keep the positions of all entites in this family to remain as is.")
        .def("SetFamilyPrescribedPositionX", &deme::DEMSolver::SetFamilyPrescribedPositionX,
             "Let the X component of the linear positions of all entites in this family always keep `as is'.")
        .def("SetFamilyPrescribedPositionY", &deme::DEMSolver::SetFamilyPrescribedPositionY,
             "Let the Y component of the linear positions of all entites in this family always keep `as is'.")
        .def("SetFamilyPrescribedPositionZ", &deme::DEMSolver::SetFamilyPrescribedPositionZ,
             "Let the Z component of the linear positions of all entites in this family always keep `as is'.")

        .def("SetFamilyPrescribedQuaternion",
             static_cast<void (deme::DEMSolver::*)(unsigned int, const std::string&, bool)>(
                 &deme::DEMSolver::SetFamilyPrescribedQuaternion),
             "Keep the orientation quaternions of all entites in this family to remain exactly the user-specified "
             "values.",
             py::arg("ID"), py::arg("q_formula"), py::arg("dictate") = true)
        .def("SetFamilyPrescribedQuaternion",
             static_cast<void (deme::DEMSolver::*)(unsigned int)>(&deme::DEMSolver::SetFamilyPrescribedQuaternion),
             "Let the orientation quaternions of all entites in this family always keep `as is'.")

        .def("AddFamilyPrescribedAcc", &deme::DEMSolver::AddFamilyPrescribedAcc,
             "The entities in this family will always experienced an extra acceleration defined using this method.",
             py::arg("ID"), py::arg("X"), py::arg("Y"), py::arg("Z"), py::arg("pre") = "none")
        .def("AddFamilyPrescribedAngAcc", &deme::DEMSolver::AddFamilyPrescribedAngAcc,
             "The entities in this family will always experienced an extra angular acceleration defined using this "
             "method.",
             py::arg("ID"), py::arg("X"), py::arg("Y"), py::arg("Z"), py::arg("pre") = "none")

        .def("CorrectFamilyLinVel", &deme::DEMSolver::CorrectFamilyLinVel,
             "The entities in this family will always experience an added linear-velocity correction defined using "
             "this method. At the same time, they are still subject to the `simulation physics'.",
             py::arg("ID"), py::arg("X"), py::arg("Y"), py::arg("Z"), py::arg("pre") = "none")
        .def("CorrectFamilyAngVel", &deme::DEMSolver::CorrectFamilyAngVel,
             "The entities in this family will always experience an added angular-velocity correction defined using "
             "this method. At the same time, they are still subject to the `simulation physics'.",
             py::arg("ID"), py::arg("X"), py::arg("Y"), py::arg("Z"), py::arg("pre") = "none")

        .def("CorrectFamilyPosition", &deme::DEMSolver::CorrectFamilyPosition,
             "The entities in this family will always experience an added positional correction defined using this "
             "method. At the same time, they are still subject to the `simulation physics'.",
             py::arg("ID"), py::arg("X"), py::arg("Y"), py::arg("Z"), py::arg("pre") = "none")
        .def("CorrectFamilyQuaternion", &deme::DEMSolver::CorrectFamilyQuaternion,
             "The entities in this family will always experience an added quaternion correction defined using this "
             "method. At the same time, they are still subject to the `simulation physics'.",
             py::arg("ID"), py::arg("q_formula"))

        .def("SetContactWildcards", &deme::DEMSolver::SetContactWildcards,
             "Set the names for the extra quantities that will be associated with each contact pair.")
        .def("SetOwnerWildcards", &deme::DEMSolver::SetOwnerWildcards,
             "Set the names for the extra quantities that will be associated with each owner.")

        .def("SetFamilyContactWildcardValueEither", &deme::DEMSolver::SetFamilyContactWildcardValueEither,
             "Change the value of contact wildcards to val if either of the contact geometries is in family N.")
        .def("SetFamilyContactWildcardValueBoth", &deme::DEMSolver::SetFamilyContactWildcardValueBoth,
             "Change the value of contact wildcards to val if both of the contact geometries are in family N.")
        .def("SetFamilyContactWildcardValue", &deme::DEMSolver::SetFamilyContactWildcardValue,
             "Change the value of contact wildcards to val if one of the contact geometry is in family N1, and the "
             "other is in N2.")
        .def("SetContactWildcardValue", &deme::DEMSolver::SetContactWildcardValue,
             "Change the value of contact wildcards to val. Apply to all simulation bodies that are present.")

        .def("MarkFamilyPersistentContactEither", &deme::DEMSolver::MarkFamilyPersistentContactEither,
             "Make it so that for any currently-existing contact, if one of its contact geometries is in family N, "
             "then this contact will never be removed.")
        .def("MarkFamilyPersistentContactBoth", &deme::DEMSolver::MarkFamilyPersistentContactBoth,
             "Make it so that for any currently-existing contact, if both of its contact geometries are in family N, "
             "then this contact will never be removed.")
        .def("MarkFamilyPersistentContact", &deme::DEMSolver::MarkFamilyPersistentContact,
             "Make it so that if for any currently-existing contact, if its two contact geometries are in family N1 "
             "and N2 respectively, this contact will never be removed.")
        .def("MarkPersistentContact", &deme::DEMSolver::MarkPersistentContact,
             "Make it so that all currently-existing contacts in this simulation will never be removed.")

        .def("RemoveFamilyPersistentContactEither", &deme::DEMSolver::RemoveFamilyPersistentContactEither,
             "Cancel contact persistence qualification. Work like the inverse of MarkFamilyPersistentContactEither.")
        .def("RemoveFamilyPersistentContactBoth", &deme::DEMSolver::RemoveFamilyPersistentContactBoth,
             "Cancel contact persistence qualification. Work like the inverse of MarkFamilyPersistentContactBoth.")
        .def("RemoveFamilyPersistentContact", &deme::DEMSolver::RemoveFamilyPersistentContact,
             "Cancel contact persistence qualification. Work like the inverse of MarkFamilyPersistentContact.")
        .def("RemovePersistentContact", &deme::DEMSolver::RemovePersistentContact,
             "Cancel contact persistence qualification. Work like the inverse of MarkPersistentContact.")

        .def("GetOwnerContactForces",
             static_cast<size_t (deme::DEMSolver::*)(const std::vector<deme::bodyID_t>& ownerIDs,
                                                     std::vector<float3>& points, std::vector<float3>& forces)>(
                 &deme::DEMSolver::GetOwnerContactForces),
             "Get all contact forces that concern a list of owners.")
        .def("GetOwnerContactForces",
             static_cast<size_t (deme::DEMSolver::*)(
                 const std::vector<deme::bodyID_t>& ownerIDs, std::vector<float3>& points, std::vector<float3>& forces,
                 std::vector<float3>& torques, bool torque_in_local)>(&deme::DEMSolver::GetOwnerContactForces),
             "Get all contact forces and torque that concern a list of owners.", py::arg("ownerIDs"), py::arg("points"),
             py::arg("forces"), py::arg("torques"), py::arg("torque_in_local") = false)

        .def("SetOwnerWildcardValue",
             static_cast<void (deme::DEMSolver::*)(deme::bodyID_t ownerID, const std::string& name,
                                                   const std::vector<float>& vals)>(
                 &deme::DEMSolver::SetOwnerWildcardValue),
             "Set the wildcard values of some owners using a list.", py::arg("ownerIDs"), py::arg("name"),
             py::arg("vals"))
        .def("SetOwnerWildcardValue",
             static_cast<void (deme::DEMSolver::*)(deme::bodyID_t ownerID, const std::string& name, float val,
                                                   size_t n)>(&deme::DEMSolver::SetOwnerWildcardValue),
             "Set the wildcard values of some owners using a list.", py::arg("ownerIDs"), py::arg("name"),
             py::arg("val"), py::arg("n") = 1)

        .def("GetOwnerWildcardValue",
             static_cast<std::vector<float> (deme::DEMSolver::*)(deme::bodyID_t ownerID, const std::string& name,
                                                                 deme::bodyID_t n)>(
                 &deme::DEMSolver::GetOwnerWildcardValue),
             "Get the owner wildcard's values of some owners.", py::arg("ownerID"), py::arg("name"), py::arg("n") = 1)
        .def("GetAllOwnerWildcardValue", &deme::DEMSolver::GetAllOwnerWildcardValue,
             "Get the owner wildcard's values of all entities.")
        .def("GetFamilyOwnerWildcardValue", &deme::DEMSolver::GetFamilyOwnerWildcardValue,
             "Get the owner wildcard's values of all entities in family N.")

        .def("SyncMemoryTransfer", &deme::DEMSolver::SyncMemoryTransfer,
             "If the user used async-ed version of a tracker's get/set methods (to get a speed boost in many piecemeal "
             "accesses of a long array), this method should be called to mark the end of to-host transactions.")

        .def("SetFamilyOwnerWildcardValue",
             static_cast<void (deme::DEMSolver::*)(unsigned int N, const std::string& name,
                                                   const std::vector<float>& vals)>(
                 &deme::DEMSolver::SetFamilyOwnerWildcardValue),
             "Modify the owner wildcard's values of all entities in family N.")
        .def("SetFamilyOwnerWildcardValue",
             static_cast<void (deme::DEMSolver::*)(unsigned int N, const std::string& name, float val)>(
                 &deme::DEMSolver::SetFamilyOwnerWildcardValue),
             "Modify the owner wildcard's values of all entities in family N.")

        .def("SetFamilyClumpMaterial", &deme::DEMSolver::SetFamilyClumpMaterial,
             "Set all clumps in this family to have this material.")
        .def("SetFamilyMeshMaterial", &deme::DEMSolver::SetFamilyMeshMaterial,
             "Set all meshes in this family to have this material.")
        .def("SetFamilyExtraMargin", &deme::DEMSolver::SetFamilyExtraMargin,
             "Add an extra contact margin to entities in a family so they are registered as potential contact pairs "
             "earlier.")

        .def("ClearCache", &deme::DEMSolver::ClearCache,
             "Remove host-side cached vectors (so you can re-define them, and then re-initialize system).")

        .def("CreateInspector",
             static_cast<std::shared_ptr<deme::DEMInspector> (deme::DEMSolver::*)(const std::string&)>(
                 &deme::DEMSolver::CreateInspector),
             "Create a inspector object that can help query some statistical info of the clumps in the simulation.",
             py::arg("quantity") = "clump_max_z")
        .def("GetNumClumps", &deme::DEMSolver::GetNumClumps,
             "Return the number of clumps that are currently in the simulation. Must be used after initialization.")
        .def("GetNumOwners", &deme::DEMSolver::GetNumOwners,
             "Return the total number of owners (clumps + meshes + analytical objects) that are currently in the "
             "simulation. Must be used after initialization.")
        .def("CreateInspector",
             static_cast<std::shared_ptr<deme::DEMInspector> (deme::DEMSolver::*)(
                 const std::string&, const std::string&)>(&deme::DEMSolver::CreateInspector),
             "Create a inspector object that can help query some statistical info of the clumps in the simulation.")
        .def("SetInitTimeStep", &deme::DEMSolver::SetInitTimeStep,
             "Legacy name for SetTimeStepSize, kept for backward compatibility.",
             py::arg("ts"))
        .def("SetTimeStepSize", &deme::DEMSolver::SetTimeStepSize,
             "Set the timestep in seconds before or after initialization. A post-initialization call takes effect on "
             "the next step and must be made while the solver is synchronized.",
             py::arg("ts"))
        .def("SetGravitationalAcceleration",
             static_cast<void (deme::DEMSolver::*)(const std::vector<float>&)>(
                 &deme::DEMSolver::SetGravitationalAcceleration),
             "Set the global gravitational acceleration as ``[x, y, z]`` in length/time^2 units. This takes effect "
             "immediately after initialization when called while the solver is synchronized.",
             py::arg("acc"))
        .def("SetMaxVelocity", &deme::DEMSolver::SetMaxVelocity,
             "Set the maximum expected particle velocity. The solver will not use a velocity larger than this for "
             "determining the margin thickness, and velocity larger than this will be considered a system anomaly. "
             "After initialization, call UpdateSimParams() for this change to take effect in the workers.")
        .def("SetErrorOutVelocity", &deme::DEMSolver::SetErrorOutVelocity,
             "Set the velocity which when exceeded, the solver errors out. A huge number can be used to discourage "
             "this error type. Defaulted to 5e4.")
        .def("SetExpandSafetyMultiplier", &deme::DEMSolver::SetExpandSafetyMultiplier,
             "Assign a multiplier to our estimated maximum system velocity, when deriving the thinckness of the "
             "contact `safety' margin. After initialization, call UpdateSimParams() for this change to take effect in "
             "the workers.")
        .def("Initialize", &deme::DEMSolver::Initialize,
             "Finalize cached setup data, allocate runtime state, and prepare JIT CUDA kernels. Call once after "
             "materials, geometry, particles, and solver options have been configured. ``dry_run=True`` performs "
             "initialization without starting normal dynamics.",
             py::arg("dry_run") = false)
        .def("WriteSphereFile",
             static_cast<void (deme::DEMSolver::*)(const std::string&) const>(&deme::DEMSolver::WriteSphereFile),
             "Write clumps as component spheres using the selected CSV or VTK output format.",
             py::arg("outfilename"))
        .def("WriteAnalyticalFile",
             static_cast<void (deme::DEMSolver::*)(const std::string&, unsigned int) const>(
                 &deme::DEMSolver::WriteAnalyticalFile),
             "Write analytical boundary surfaces as VTK clipped to the user-specified domain.",
             py::arg("outfilename"), py::arg("circumferential_resolution") = 32)
        .def("WriteMeshFile",
             static_cast<void (deme::DEMSolver::*)(const std::string&) const>(&deme::DEMSolver::WriteMeshFile),
             "Write the current status of all meshes to a file.")
        .def("WriteClumpFile",
             static_cast<void (deme::DEMSolver::*)(const std::string&, unsigned int) const>(
                 &deme::DEMSolver::WriteClumpFile),
             "Write the current status of clumps to a file.", py::arg("outfilename"), py::arg("accuracy") = 10)
        .def(
            "WriteContactFile",
            static_cast<void (deme::DEMSolver::*)(const std::string&, float) const>(&deme::DEMSolver::WriteContactFile),
            "Write all contact pairs to a file. Forces smaller than threshold will not be outputted.",
            py::arg("outfilename"), py::arg("force_thres") = 1e-30)
        .def("WriteContactFileIncludingPotentialPairs",
             static_cast<void (deme::DEMSolver::*)(const std::string&) const>(
                 &deme::DEMSolver::WriteContactFileIncludingPotentialPairs),
             "Write all contact pairs kT-supplied to a file, thus including the potential ones (those are not yet in "
             "contact, or recently used to be in contact).",
             py::arg("outfilename"))
        .def("WaitForPendingOutput", &deme::DEMSolver::WaitForPendingOutput,
             "Wait for any in-flight async output to finish.")

        // Maybe add checkpoint-reading methods here...
        .def("DoDynamics", &deme::DEMSolver::DoDynamics,
             "Advance by ``duration`` seconds without forcing the dynamic and kinematic workers to synchronize at "
             "the call boundary. This supports short calls and co-simulation; synchronize before immediately reading "
             "state.",
             py::arg("duration"))
        .def("DoStepDynamics", &deme::DEMSolver::DoStepDynamics,
             "Advance one current timestep without forcing a final dynamic/kinematic worker synchronization.")
        .def("DoDynamicsThenSync", &deme::DEMSolver::DoDynamicsThenSync,
             "Advance by ``duration`` seconds, then synchronize the dynamic and kinematic workers. Use this before "
             "immediately querying or modifying runtime state.",
             py::arg("duration"))
        .def("UpdateSimParams", &deme::DEMSolver::UpdateSimParams,
             "Transfer the cached sim params to the workers. Used for sim environment modification after system "
             "initialization.")
        .def("ChangeFamily", &deme::DEMSolver::ChangeFamily,
             "Change all entities with family number ID_from to have a new number ID_to, when the condition defined by "
             "the string is satisfied by the entities in question. This should be called before initialization, and "
             "will be baked into the solver, so the conditions will be checked and changes applied every time step.")
        .def("ChangeFamilyWhen", &deme::DEMSolver::ChangeFamilyWhen,
             "Change all entities with family number ID_from to have a new number ID_to, when the condition defined by "
             "the string is satisfied by the entities in question. This should be called before initialization, and "
             "will be baked into the solver, so the conditions will be checked and changes applied every time step.")
        .def("ChangeClumpFamily", &deme::DEMSolver::ChangeClumpFamily,
             "Change the family number for the clumps in a box region to the specified value.", py::arg("fam_num"),
             py::arg("X") = std::pair<double, double>(-DEME_HUGE_FLOAT, DEME_HUGE_FLOAT),
             py::arg("Y") = std::pair<double, double>(-DEME_HUGE_FLOAT, DEME_HUGE_FLOAT),
             py::arg("Z") = std::pair<double, double>(-DEME_HUGE_FLOAT, DEME_HUGE_FLOAT),
             py::arg("orig_fam") = std::set<unsigned int>())

        .def("ShowThreadCollaborationStats", &deme::DEMSolver::ShowThreadCollaborationStats,
             "Show the collaboration stats between dT and kT. This is more useful for tweaking the number of time "
             "steps that dT should be allowed to be in advance of kT.")
        .def("ShowTimingStats", &deme::DEMSolver::ShowTimingStats,
             "Show the wall time and percentages of wall time spend on various solver tasks.")
        .def("SetGPUTimersEnabled", &deme::DEMSolver::SetGPUTimersEnabled,
             "Enable or disable event-based GPU timing.", py::arg("enabled"))
        .def("GetGPUTimersEnabled", &deme::DEMSolver::GetGPUTimersEnabled,
             "Return whether event-based GPU timing is enabled.")
        .def("PrintKinematicScratchSpaceUsage", &deme::DEMSolver::PrintKinematicScratchSpaceUsage,
             "Print kT's scratch space usage. This is a debug method.")
        .def("ShowAnomalies", &deme::DEMSolver::ShowAnomalies,
             "Show potential anomalies that may have been there in the simulation, then clear the anomaly log.")
        .def("ClearThreadCollaborationStats", &deme::DEMSolver::ClearThreadCollaborationStats,
             "Reset the collaboration stats between dT and kT back to the initial value (0). You should call this if "
             "you want to start over and re-inspect the stats of the new run; otherwise, it is generally not needed, "
             "you can go ahead and destroy DEMSolver.")
        .def("ClearTimingStats", &deme::DEMSolver::ClearTimingStats,
             "Reset the recordings of the wall time and percentages of wall time spend on various solver tasks.")
        .def("PurgeFamily", &deme::DEMSolver::PurgeFamily,
             "Remove all entities in a family from runtime arrays to reclaim memory.", py::arg("family_num"))
        .def("ReleaseFlattenedArrays", &deme::DEMSolver::ReleaseFlattenedArrays,
             "Release setup-time flattened arrays used for preprocessing and worker transfer.")
        .def("GetWhetherForceCollectInKernel", &deme::DEMSolver::GetWhetherForceCollectInKernel,
             "Return whether the solver is currently reducing force in the force calculation kernel.")
        .def("AddOwnerNextStepAcc", &deme::DEMSolver::AddOwnerNextStepAcc,
             "Add an extra acceleration to a owner for the next time step.")
        .def("AddOwnerNextStepAngAcc", &deme::DEMSolver::AddOwnerNextStepAngAcc,
             " Add an extra angular acceleration to a owner for the next time step.")
        .def("DisableContactBetweenFamilies", &deme::DEMSolver::DisableContactBetweenFamilies,
             "Instruct the solver that the 2 input families should not have contacts (a.k.a. ignored, if such a pair "
             "is encountered in contact detection). These 2 families can be the same (which means no contact within "
             "members of that family).")
        .def("EnableContactBetweenFamilies", &deme::DEMSolver::EnableContactBetweenFamilies,
             "Re-enable contact between 2 families after the system is initialized.")
        .def("DisableFamilyOutput", &deme::DEMSolver::DisableFamilyOutput,
             "Prevent entites associated with this family to be outputted to files.");

#ifdef DEME_HAS_VISUALIZER
    py::class_<deme::DEMVisualizerColor>(obj, "VisualizerColor", "RGBA color used by DEMVisualizer.")
        .def(py::init<std::uint8_t, std::uint8_t, std::uint8_t, std::uint8_t>(), py::arg("r") = 255, py::arg("g") = 255,
             py::arg("b") = 255, py::arg("a") = 255)
        .def_readwrite("r", &deme::DEMVisualizerColor::r)
        .def_readwrite("g", &deme::DEMVisualizerColor::g)
        .def_readwrite("b", &deme::DEMVisualizerColor::b)
        .def_readwrite("a", &deme::DEMVisualizerColor::a);

    py::enum_<deme::DEMVisualizerColorMode>(obj, "VisualizerColorMode")
        .value("FAMILY", deme::DEMVisualizerColorMode::FAMILY)
        .value("HEIGHT", deme::DEMVisualizerColorMode::HEIGHT)
        .value("SPEED", deme::DEMVisualizerColorMode::SPEED);

    py::class_<deme::DEMVisualizer>(
        obj, "DEMVisualizer",
        "Step-wise interactive viewer. Render() draws the current solver state without advancing the simulation.")
        .def(py::init<deme::DEMSolver&>(), py::arg("solver"), py::keep_alive<1, 2>(),
             "Create a visualizer and keep its solver alive for the lifetime of the viewer.")
        .def("Initialize", &deme::DEMVisualizer::Initialize, "Create the visualization window and graphics resources.")
        .def("Run", &deme::DEMVisualizer::Run, "Return true while the initialized window remains open.")
        .def("Render", &deme::DEMVisualizer::Render,
             "Synchronously capture the current solver state and draw one frame without advancing the solver.")
        .def("Close", &deme::DEMVisualizer::Close, "Close the visualization window.")
        .def("SetWindowSize", &deme::DEMVisualizer::SetWindowSize, py::arg("width"), py::arg("height"))
        .def("SetWindowTitle", &deme::DEMVisualizer::SetWindowTitle, py::arg("title"))
        .def("SetTargetFPS", &deme::DEMVisualizer::SetTargetFPS, py::arg("fps"))
        .def("SetCameraPosition", &deme::DEMVisualizer::SetCameraPosition, py::arg("position"))
        .def("SetCameraTarget", &deme::DEMVisualizer::SetCameraTarget, py::arg("target"))
        .def("SetBackgroundColor", &deme::DEMVisualizer::SetBackgroundColor, py::arg("color"))
        .def("SetFamilyColor", &deme::DEMVisualizer::SetFamilyColor, py::arg("family"), py::arg("color"))
        .def("SetRenderSpheres", &deme::DEMVisualizer::SetRenderSpheres, py::arg("render") = true,
             "Enable or disable component-sphere rendering; enabled by default.")
        .def("SetRenderTriangles", &deme::DEMVisualizer::SetRenderTriangles, py::arg("render") = true,
             "Enable or disable triangle rendering; enabled by default.")
        .def("IsRenderingSpheres", &deme::DEMVisualizer::IsRenderingSpheres)
        .def("IsRenderingTriangles", &deme::DEMVisualizer::IsRenderingTriangles)
        .def("ShouldStep", &deme::DEMVisualizer::ShouldStep)
        .def("SetPaused", &deme::DEMVisualizer::SetPaused, py::arg("paused"))
        .def("IsPaused", &deme::DEMVisualizer::IsPaused)
        .def("RequestStep", &deme::DEMVisualizer::RequestStep)
        .def("SetFamilyVisible", &deme::DEMVisualizer::SetFamilyVisible, py::arg("family"), py::arg("visible"))
        .def("SetColorMode", &deme::DEMVisualizer::SetColorMode, py::arg("mode"))
        .def("FrameAll", &deme::DEMVisualizer::FrameAll)
        .def("FrameSelected", &deme::DEMVisualizer::FrameSelected)
        .def("PickAt", &deme::DEMVisualizer::PickAt, py::arg("x"), py::arg("y"))
        .def("GetSelectedOwner", &deme::DEMVisualizer::GetSelectedOwner)
        .def("GetSelectedGeometryID", &deme::DEMVisualizer::GetSelectedGeometryID)
        .def("IsSelectedSphere", &deme::DEMVisualizer::IsSelectedSphere)
        .def("RequestScreenshot", &deme::DEMVisualizer::RequestScreenshot, py::arg("path"));
#endif

    py::class_<deme::DEMMaterial, std::shared_ptr<deme::DEMMaterial>>(
        obj, "DEMMaterial", "Material property name/value pairs used by a contact force model.")
        .def(py::init<const std::unordered_map<std::string, float>&>(), py::arg("properties"),
             "Create material properties. Normally use ``DEMSolver.LoadMaterial`` so the material is registered.")
        .def_readwrite("mat_prop", &deme::DEMMaterial::mat_prop)
        .def_readwrite("load_order", &deme::DEMMaterial::load_order);

    py::class_<deme::DEMClumpTemplate, std::shared_ptr<deme::DEMClumpTemplate>>(
        obj, "DEMClumpTemplate", "Reusable rigid clump topology composed of one or more sphere components.")
        .def(py::init<>(), "Create an empty clump template.")
        .def("Mass", &deme::DEMClumpTemplate::GetMass, "Return the clump mass.")
        .def("MOI", &deme::DEMClumpTemplate::GetMOI,
             "Return principal moments of inertia in the clump-local principal frame.")
        .def("SetMass", &deme::DEMClumpTemplate::SetMass, "Set the clump mass.", py::arg("mass"))
        .def("SetMOI",
             static_cast<void (deme::DEMClumpTemplate::*)(const std::vector<float>&)>(&deme::DEMClumpTemplate::SetMOI),
             "Set the three principal moments of inertia in the clump-local frame.", py::arg("MOI"))
        .def("SetMaterial",
             static_cast<void (deme::DEMClumpTemplate::*)(const std::vector<std::shared_ptr<deme::DEMMaterial>>&)>(
                 &deme::DEMClumpTemplate::SetMaterial),
             "Assign one material per sphere component. The sequence length must equal the component count.",
             py::arg("materials"))
        .def("SetMaterial",
             static_cast<void (deme::DEMClumpTemplate::*)(const std::shared_ptr<deme::DEMMaterial>& input)>(
                 &deme::DEMClumpTemplate::SetMaterial),
             "Assign the same material to every sphere component.", py::arg("material"))
        .def("SetVolume", &deme::DEMClumpTemplate::SetVolume,
             "Set the clump volume, which is required for void-ratio queries.", py::arg("volume"))
        .def("ReadComponentFromFile", &deme::DEMClumpTemplate::ReadComponentFromFile,
             "Load sphere-component positions and radii from a delimited file.", py::arg("filename"),
             py::arg("x_id") = "x", py::arg("y_id") = "y", py::arg("z_id") = "z", py::arg("r_id") = "r")
        .def("InformCentroidPrincipal",
             static_cast<void (deme::DEMClumpTemplate::*)(const std::vector<float>&, const std::vector<float>&)>(
                 &deme::DEMClumpTemplate::InformCentroidPrincipal),
             "Re-express component positions about the supplied center of mass and principal orientation.",
             py::arg("center"), py::arg("orientation"))
        .def("Move",
             static_cast<void (deme::DEMClumpTemplate::*)(const std::vector<float>&, const std::vector<float>&)>(
                 &deme::DEMClumpTemplate::Move),
             "Apply a translation and quaternion rotation to the component coordinates.", py::arg("translation"),
             py::arg("quaternion"))
        .def("Scale", &deme::DEMClumpTemplate::Scale, "Uniformly scale component radii and relative positions.",
             py::arg("factor"))
        .def("AssignName", &deme::DEMClumpTemplate::AssignName, "Assign the clump type name written to output files.",
             py::arg("name"));

    py::class_<deme::DEMCombinedTemplate, std::shared_ptr<deme::DEMCombinedTemplate>>(
        obj, "DEMCombinedTemplate",
        "Reusable rigid grouping of clump or mesh templates with fixed member-relative transforms.")
        .def(py::init<>(), "Create an empty combined-owner template.");

    py::class_<deme::DEMCombinedInstances, std::shared_ptr<deme::DEMCombinedInstances>>(
        obj, "DEMCombinedInstances", "A batch of instantiated combined owners and their member-owner metadata.")
        .def(py::init<>(), "Create an empty combined-instance batch.")
        .def("GetNumOwners", &deme::DEMCombinedInstances::GetNumOwners,
             "Get total number of member owners in this combined batch.")
        .def("AddOwnerWildcard",
             static_cast<void (deme::DEMCombinedInstances::*)(const std::string&, const std::vector<float>&)>(
                 &deme::DEMCombinedInstances::AddOwnerWildcard),
             "Add an owner wildcard to all member owners with per-owner values.", py::arg("name"), py::arg("vals"))
        .def("AddOwnerWildcard",
             static_cast<void (deme::DEMCombinedInstances::*)(const std::string&, float)>(
                 &deme::DEMCombinedInstances::AddOwnerWildcard),
             "Add an owner wildcard to all member owners with a uniform value.", py::arg("name"), py::arg("val"));

    py::class_<deme::DEMClumpBatch, deme::DEMInitializer, std::shared_ptr<deme::DEMClumpBatch>>(
        obj, "DEMClumpBatch", "Cached setup data for a batch of clump instances.")
        .def(py::init<size_t&>(), py::arg("count"), "Create setup storage for ``count`` clumps.")
        .def("GetNumClumps", &deme::DEMClumpBatch::GetNumClumps, "Return the number of clumps in this batch.")
        .def("GetNumSpheres", &deme::DEMClumpBatch::GetNumSpheres,
             "Return the total number of component spheres represented by this batch.")
        .def("SetTypes",
             static_cast<void (deme::DEMClumpBatch::*)(const std::vector<std::shared_ptr<deme::DEMClumpTemplate>>&)>(
                 &deme::DEMClumpBatch::SetTypes),
             "Assign one clump template per instance. The sequence length must equal the batch size.", py::arg("types"))
        .def("SetTypes",
             static_cast<void (deme::DEMClumpBatch::*)(const std::shared_ptr<deme::DEMClumpTemplate>&)>(
                 &deme::DEMClumpBatch::SetTypes),
             "Assign the same clump template to every instance.", py::arg("type"))
        .def("SetType", &deme::DEMClumpBatch::SetType, "Assign the same clump template to every instance.",
             py::arg("type"))
        .def("SetVel",
             static_cast<void (deme::DEMClumpBatch::*)(const std::vector<std::vector<float>>&)>(
                 &deme::DEMClumpBatch::SetVel),
             "Set one global initial linear velocity vector per clump.", py::arg("velocities"))
        .def("SetVel",
             static_cast<void (deme::DEMClumpBatch::*)(const std::vector<float>&)>(&deme::DEMClumpBatch::SetVel),
             "Set the same global initial linear velocity for every clump.", py::arg("velocity"))
        //    .def("SetAngVel",
        //         static_cast<void (deme::DEMClumpBatch::*)(const
        //         std::vector<float3>&)>(&deme::DEMClumpBatch::SetAngVel))
        .def("SetFamilies",
             static_cast<void (deme::DEMClumpBatch::*)(const std::vector<unsigned int>&)>(
                 &deme::DEMClumpBatch::SetFamilies),
             "Set one family number per clump. Family behavior is configured on the solver.", py::arg("families"))
        .def("SetFamilies", static_cast<void (deme::DEMClumpBatch::*)(unsigned int)>(&deme::DEMClumpBatch::SetFamilies),
             "Set the same family number for every clump.", py::arg("family"))
        .def("SetFamily", &deme::DEMClumpBatch::SetFamily, "Set the same family number for every clump.",
             py::arg("family"))
        .def("SetExistingContacts", &deme::DEMClumpBatch::SetExistingContacts,
             "Supply sphere-sphere contact owner-ID pairs relative to this batch when restoring a simulation.",
             py::arg("pairs"))
        .def("SetExistingContactWildcards", &deme::DEMClumpBatch::SetExistingContactWildcards,
             "Supply named restart wildcard arrays after ``SetExistingContacts``. Every array must match the "
             "pre-existing contact count.",
             py::arg("wildcards"))
        .def("AddExistingContactWildcard", &deme::DEMClumpBatch::AddExistingContactWildcard,
             "Add one named restart contact-wildcard array after ``SetExistingContacts``.", py::arg("name"),
             py::arg("values"))
        .def("SetOwnerWildcards", &deme::DEMClumpBatch::SetOwnerWildcards,
             "Set named initial owner-wildcard arrays. Every array must match the clump count.", py::arg("wildcards"))
        .def("AddOwnerWildcard",
             static_cast<void (deme::DEMClumpBatch::*)(const std::string&, const std::vector<float>&)>(
                 &deme::DEMClumpBatch::AddOwnerWildcard),
             "Add one initial owner wildcard with one value per clump.", py::arg("name"), py::arg("values"))
        .def("AddOwnerWildcard",
             static_cast<void (deme::DEMClumpBatch::*)(const std::string&, float)>(
                 &deme::DEMClumpBatch::AddOwnerWildcard),
             "Add one initial owner wildcard with the same value for every clump.", py::arg("name"), py::arg("value"))
        .def("GetNumContacts", &deme::DEMClumpBatch::GetNumContacts,
             "Return the number of pre-existing contacts supplied for restart initialization.");

    py::class_<deme::DEMExternObj, deme::DEMInitializer, std::shared_ptr<deme::DEMExternObj>>(
        obj, "DEMExternObj", "Analytical external owner composed of planes, cylinders, and related primitives.")
        .def(py::init<>(), "Create an empty analytical external object.")
        .def("Mass", &deme::DEMExternObj::GetMass, "Return the object's mass.")
        .def("MOI", &deme::DEMExternObj::GetMOI, "Return principal moments of inertia in the object-local frame.")
        .def("SetFamily", &deme::DEMExternObj::SetFamily, "Set the contact family number before initialization.",
             py::arg("family"))
        .def("SetMass", &deme::DEMExternObj::SetMass, "Set the object mass.", py::arg("mass"))
        .def("SetMOI",
             static_cast<void (deme::DEMExternObj::*)(const std::vector<float>&)>(&deme::DEMExternObj::SetMOI),
             "Set the three principal moments of inertia in the object-local frame.", py::arg("MOI"))
        .def("SetInitQuat",
             static_cast<void (deme::DEMExternObj::*)(const std::vector<float>&)>(&deme::DEMExternObj::SetInitQuat),
             "Set the initial local-to-global orientation quaternion ``(x, y, z, w)`` before initialization.",
             py::arg("quaternion"))
        .def("SetInitPos",
             static_cast<void (deme::DEMExternObj::*)(const std::vector<float>&)>(&deme::DEMExternObj::SetInitPos),
             "Set the initial global center-of-mass position before initialization.", py::arg("position"))
        .def("AddPlane",
             static_cast<void (deme::DEMExternObj::*)(const std::vector<float>&, const std::vector<float>&,
                                                      const std::shared_ptr<deme::DEMMaterial>&)>(
                 &deme::DEMExternObj::AddPlane),
             "Add an infinite plane specified by a point and normal in the object's local frame.", py::arg("point"),
             py::arg("normal"), py::arg("material"))
        //.def("AddPlate", static_cast<void (&deme::DEMExternObj::AddPlate, "Add a plate with finite size.")
        .def("AddZCylinder",
             static_cast<void (deme::DEMExternObj::*)(const std::vector<float>&, const float,
                                                      const std::shared_ptr<deme::DEMMaterial>&,
                                                      const deme::objNormal_t)>(&deme::DEMExternObj::AddZCylinder),
             "Add an infinite cylinder aligned with the object-local Z axis.", py::arg("pos"), py::arg("rad"),
             py::arg("material"), py::arg("normal") = deme::ENTITY_NORMAL_INWARD)
        .def("AddCylinder",
             static_cast<void (deme::DEMExternObj::*)(const std::vector<float>&, const std::vector<float>&, const float,
                                                      const std::shared_ptr<deme::DEMMaterial>&,
                                                      const deme::objNormal_t)>(&deme::DEMExternObj::AddCylinder),
             "Add an infinite cylinder along a user-specified object-local axis.", py::arg("pos"), py::arg("axis"),
             py::arg("rad"), py::arg("material"), py::arg("normal") = deme::ENTITY_NORMAL_INWARD)
        .def("AddCone",
             static_cast<void (deme::DEMExternObj::*)(const std::vector<float>&, const std::vector<float>&, const float,
                                                      const std::shared_ptr<deme::DEMMaterial>&,
                                                      const deme::objNormal_t)>(&deme::DEMExternObj::AddCone),
             "Add an analytical cone side extending indefinitely from its tip.", py::arg("tip"), py::arg("axis"),
             py::arg("slope"), py::arg("material"), py::arg("normal") = deme::ENTITY_NORMAL_INWARD)
        .def("AddConeSegment",
             static_cast<void (deme::DEMExternObj::*)(
                 const std::vector<float>&, const std::vector<float>&, const float, const float, const float,
                 const std::shared_ptr<deme::DEMMaterial>&, const deme::objNormal_t)>(
                 &deme::DEMExternObj::AddConeSegment),
             "Add an analytical cone or frustum side clipped between two axial distances.", py::arg("tip"),
             py::arg("axis"), py::arg("slope"), py::arg("hmin"), py::arg("hmax"), py::arg("material"),
             py::arg("normal") = deme::ENTITY_NORMAL_INWARD)
        .def_readwrite("types", &deme::DEMExternObj::types)
        .def_readwrite("materials", &deme::DEMExternObj::materials)
        .def_readwrite("family_code", &deme::DEMExternObj::family_code)
        .def_readwrite("init_pos", &deme::DEMExternObj::init_pos)
        .def_readwrite("init_oriQ", &deme::DEMExternObj::init_oriQ)
        .def_readwrite("mass", &deme::DEMExternObj::mass)
        .def_readwrite("MOI", &deme::DEMExternObj::MOI)
        .def_readwrite("load_order", &deme::DEMExternObj::load_order)
        .def_readwrite("entity_params", &deme::DEMExternObj::entity_params);

    py::class_<deme::DEMMeshConnected, deme::DEMInitializer, std::shared_ptr<deme::DEMMeshConnected>>(
        obj, "DEMMesh", "Connected triangle-mesh template or external mesh owner.")
        .def(py::init<>(), "Create an empty mesh.")
        .def(py::init<std::string&>(), py::arg("filename"), "Load a mesh from a Wavefront OBJ file.")
        .def(py::init<std::string, const std::shared_ptr<deme::DEMMaterial>&>(), py::arg("filename"),
             py::arg("material"), "Load a Wavefront OBJ mesh and assign one material to every triangle.")
        .def("Mass", &deme::DEMMeshConnected::GetMass, "Return the mesh owner's mass.")
        .def("MOI", &deme::DEMMeshConnected::GetMOI, "Return principal moments of inertia in the mesh-local frame.")
        .def("Clear", &deme::DEMMeshConnected::Clear, "Remove all mesh geometry and cached attributes.")
        .def("LoadWavefrontMesh", &deme::DEMMeshConnected::LoadWavefrontMesh,
             "Load a triangle mesh saved as a Wavefront .obj file", py::arg("input_file"),
             py::arg("load_normals") = true, py::arg("load_uv") = false)
        .def("WriteWavefront", &deme::DEMMeshConnected::WriteWavefront,
             "Write the mesh geometry to a Wavefront OBJ file.", py::arg("output_file"))
        .def("GetNumTriangles", &deme::DEMMeshConnected::GetNumTriangles,
             "Get the number of triangles already added to this mesh")
        .def("GetNumNodes", &deme::DEMMeshConnected::GetNumNodes, "Get the number of nodes in the mesh")
        .def("UseNormals", &deme::DEMMeshConnected::UseNormals,
             "Instruct that when the mesh is initialized into the system, it will re-order the nodes of each triangle "
             "so that the normals derived from right-hand-rule are the same as the normals in the mesh file",
             py::arg("use") = true)
        .def("SetShellThickness", &deme::DEMMeshConnected::SetShellThickness,
             "Treat this mesh as a finite-thickness shell.", py::arg("thickness"))
        .def("DisableShell", &deme::DEMMeshConnected::DisableShell, "Disable finite-thickness shell mode.")
        .def("IsShell", &deme::DEMMeshConnected::IsShell, "Return whether this mesh is configured as a shell.")
        .def("GetShellThickness", &deme::DEMMeshConnected::GetShellThickness, "Return the full shell thickness.")
        .def("GetTriangle", &deme::DEMMeshConnected::GetTriangle, "Access the n-th triangle in mesh")
        .def("SetPatchIDs", &deme::DEMMeshConnected::SetPatchIDs, "Set one patch ID per triangle.",
             py::arg("patch_ids"))
        .def("SetEachTriangleAsPatch", &deme::DEMMeshConnected::SetEachTriangleAsPatch,
             "Assign every triangle to its own patch using consecutive patch IDs.")
        .def("GetPatchIDs", &deme::DEMMeshConnected::GetPatchIDs, "Get one patch ID per triangle.")
        .def("GetNumPatches", &deme::DEMMeshConnected::GetNumPatches, "Get the number of mesh patches.")
        .def("ArePatchesExplicitlySet", &deme::DEMMeshConnected::ArePatchesExplicitlySet,
             "Return whether patch IDs were explicitly supplied or computed.")
        .def("SplitIntoConvexPatches",
             static_cast<unsigned int (deme::DEMMeshConnected::*)(float)>(
                 &deme::DEMMeshConnected::SplitIntoConvexPatches),
             "Split the mesh into connected angle-threshold patches.", py::arg("hard_angle_deg") = 30.0f)
        .def("SetMass", &deme::DEMMeshConnected::SetMass, "Set the mesh owner's mass.", py::arg("mass"))
        .def("SetMOI",
             static_cast<void (deme::DEMMeshConnected::*)(const std::vector<float>&)>(&deme::DEMMeshConnected::SetMOI),
             "Set the three principal moments of inertia in the mesh-local frame.", py::arg("MOI"))
        .def("SetFamily", &deme::DEMMeshConnected::SetFamily, "Set the contact family number before initialization.",
             py::arg("family"))
        .def("SetMaterial",
             static_cast<void (deme::DEMMeshConnected::*)(const std::shared_ptr<deme::DEMMaterial>&)>(
                 &deme::DEMMeshConnected::SetMaterial),
             "Assign the same material to every mesh triangle.", py::arg("material"))
        .def("SetMaterial",
             static_cast<void (deme::DEMMeshConnected::*)(const std::vector<std::shared_ptr<deme::DEMMaterial>>&)>(
                 &deme::DEMMeshConnected::SetMaterial),
             "Assign one material per triangle. The sequence length must equal the triangle count.",
             py::arg("materials"))
        .def("SetInitQuat",
             static_cast<void (deme::DEMMeshConnected::*)(const std::vector<float>&)>(
                 &deme::DEMMeshConnected::SetInitQuat),
             "Set the initial local-to-global orientation quaternion ``(x, y, z, w)``.", py::arg("quaternion"))
        .def("SetInitPos",
             static_cast<void (deme::DEMMeshConnected::*)(const std::vector<float>&)>(
                 &deme::DEMMeshConnected::SetInitPos),
             "Set the initial global center-of-mass position.", py::arg("position"))
        .def("InformCentroidPrincipal",
             static_cast<void (deme::DEMMeshConnected::*)(const std::vector<float>&, const std::vector<float>&)>(
                 &deme::DEMMeshConnected::InformCentroidPrincipal),
             "Re-express mesh vertices about the supplied center of mass and principal orientation.", py::arg("center"),
             py::arg("orientation"))
        .def("Move",
             static_cast<void (deme::DEMMeshConnected::*)(const std::vector<float>&, const std::vector<float>&)>(
                 &deme::DEMMeshConnected::Move),
             "Apply a translation and quaternion rotation to mesh vertices.", py::arg("translation"),
             py::arg("quaternion"))
        .def("Mirror",
             static_cast<void (deme::DEMMeshConnected::*)(const std::vector<float>&, const std::vector<float>&)>(
                 &deme::DEMMeshConnected::Mirror),
             "Mirror vertices about a plane specified by a point and normal.", py::arg("point"), py::arg("normal"))
        .def("Scale", static_cast<void (deme::DEMMeshConnected::*)(float)>(&deme::DEMMeshConnected::Scale),
             "Uniformly scale mesh vertices.", py::arg("factor"))
        .def("Scale",
             static_cast<void (deme::DEMMeshConnected::*)(const std::vector<float>&)>(&deme::DEMMeshConnected::Scale),
             "Scale mesh vertices independently along X, Y, and Z.", py::arg("factors"))
        .def("ClearWildcards", &deme::DEMMeshConnected::ClearWildcards, "Remove cached triangle wildcard values.")
        .def("GetCoordsVertices", &deme::DEMMeshConnected::GetCoordsVerticesAsVectorOfVectors,
             "Return mesh vertex coordinates as ``[[x, y, z], ...]``.")
        //.def("GetCoordsUV", &deme::DEMMeshConnected::GetCoordsUVPython)
        //.def("GetCoordsColors", &deme::DEMMeshConnected::GetCoordsColorsPython)
        .def("GetIndicesVertexes", &deme::DEMMeshConnected::GetIndicesVertexesAsVectorOfVectors,
             "Return triangle vertex-index triplets.");
    //    .def("GetIndicesNormals", &deme::DEMMeshConnected::GetIndicesNormalsPython)
    //    .def("GetIndicesUV", &deme::DEMMeshConnected::GetIndicesUVPython)
    //    .def("GetIndicesColors", &deme::DEMMeshConnected::GetIndicesColorsPython);

    //// TODO: Insert readwrite functions to access all public class objects!

    py::enum_<deme::TIME_INTEGRATOR>(obj, "TIME_INTEGRATOR")
        .value("FORWARD_EULER", deme::TIME_INTEGRATOR::FORWARD_EULER)
        .value("CENTERED_DIFFERENCE", deme::TIME_INTEGRATOR::CENTERED_DIFFERENCE)
        .value("EXTENDED_TAYLOR", deme::TIME_INTEGRATOR::EXTENDED_TAYLOR)
        .value("CHUNG", deme::TIME_INTEGRATOR::CHUNG)
        .export_values();

    py::enum_<deme::OWNER_TYPE>(obj, "OWNER_TYPE")
        .value("CLUMP", deme::OWNER_TYPE::CLUMP)
        .value("ANALYTICAL", deme::OWNER_TYPE::ANALYTICAL)
        .value("MESH", deme::OWNER_TYPE::MESH)
        .export_values();

    py::enum_<deme::FORCE_MODEL>(obj, "FORCE_MODEL")
        .value("HERTZIAN", deme::FORCE_MODEL::HERTZIAN)
        .value("HERTZIAN_FRICTIONLES", deme::FORCE_MODEL::HERTZIAN_FRICTIONLESS)
        .value("CUSTOM", deme::FORCE_MODEL::CUSTOM)
        .export_values();

    py::enum_<deme::OUTPUT_CONTENT>(obj, "OUTPUT_CONTENT")
        .value("XYZ", deme::OUTPUT_CONTENT::XYZ)
        .value("QUAT", deme::OUTPUT_CONTENT::QUAT)
        .value("ABSV", deme::OUTPUT_CONTENT::ABSV)
        .value("VEL", deme::OUTPUT_CONTENT::VEL)
        .value("ANG_VEL", deme::OUTPUT_CONTENT::ANG_VEL)
        .value("ABS_ACC", deme::OUTPUT_CONTENT::ABS_ACC)
        .value("ACC", deme::OUTPUT_CONTENT::ACC)
        .value("ANG_ACC", deme::OUTPUT_CONTENT::ANG_ACC)
        .value("FAMILY", deme::OUTPUT_CONTENT::FAMILY)
        .value("MAT", deme::OUTPUT_CONTENT::MAT)
        .value("OWNER_WILDCARD", deme::OUTPUT_CONTENT::OWNER_WILDCARD)
        .export_values();

    py::enum_<deme::MESH_OUTPUT_CONTENT>(obj, "MESH_OUTPUT_CONTENT")
        .value("XYZ", deme::MESH_OUTPUT_CONTENT::XYZ)
        .value("QUAT", deme::MESH_OUTPUT_CONTENT::QUAT)
        .value("ABSV", deme::MESH_OUTPUT_CONTENT::ABSV)
        .value("VEL", deme::MESH_OUTPUT_CONTENT::VEL)
        .value("ANG_VEL", deme::MESH_OUTPUT_CONTENT::ANG_VEL)
        .value("ABS_ACC", deme::MESH_OUTPUT_CONTENT::ABS_ACC)
        .value("ACC", deme::MESH_OUTPUT_CONTENT::ACC)
        .value("ANG_ACC", deme::MESH_OUTPUT_CONTENT::ANG_ACC)
        .value("FAMILY", deme::MESH_OUTPUT_CONTENT::FAMILY)
        .value("MAT", deme::MESH_OUTPUT_CONTENT::MAT)
        .value("OWNER_WILDCARD", deme::MESH_OUTPUT_CONTENT::OWNER_WILDCARD)
        .value("GEO_WILDCARD", deme::MESH_OUTPUT_CONTENT::GEO_WILDCARD)
        .value("OWNER", deme::MESH_OUTPUT_CONTENT::OWNER)
        .value("MESH_ID", deme::MESH_OUTPUT_CONTENT::MESH_ID)
        .value("TRI_ID", deme::MESH_OUTPUT_CONTENT::TRI_ID)
        .value("PATCH_ID", deme::MESH_OUTPUT_CONTENT::PATCH_ID);

    py::enum_<deme::OUTPUT_FORMAT>(obj, "OUTPUT_FORMAT")
        .value("CSV", deme::OUTPUT_FORMAT::CSV)
        .value("BINARY", deme::OUTPUT_FORMAT::BINARY)
        .value("VTK", deme::OUTPUT_FORMAT::VTK)
        .export_values();

    py::enum_<deme::MESH_FORMAT>(obj, "MESH_FORMAT")
        .value("VTK", deme::MESH_FORMAT::VTK)
        .value("OBJ", deme::MESH_FORMAT::OBJ)
        .value("STL", deme::MESH_FORMAT::STL)
        .value("PLY", deme::MESH_FORMAT::PLY)
        .export_values();

    py::enum_<deme::SPATIAL_DIR>(obj, "SPATIAL_DIR")
        .value("X", deme::SPATIAL_DIR::X)
        .value("Y", deme::SPATIAL_DIR::Y)
        .value("Z", deme::SPATIAL_DIR::Z)
        .value("NONE", deme::SPATIAL_DIR::NONE)
        .export_values();

    py::enum_<deme::CNT_OUTPUT_CONTENT>(obj, "CNT_OUTPUT_CONTENT")
        .value("CNT_TYPE", deme::CNT_OUTPUT_CONTENT::CNT_TYPE)
        .value("FORCE", deme::CNT_OUTPUT_CONTENT::FORCE)
        .value("POINT", deme::CNT_OUTPUT_CONTENT::CNT_POINT)
        .value("COMPONENT", deme::CNT_OUTPUT_CONTENT::COMPONENT)
        .value("NORMAL", deme::CNT_OUTPUT_CONTENT::NORMAL)
        .value("TORQUE", deme::CNT_OUTPUT_CONTENT::TORQUE)
        .value("CNT_WILDCARD", deme::CNT_OUTPUT_CONTENT::CNT_WILDCARD)
        .value("OWNER", deme::CNT_OUTPUT_CONTENT::OWNER)
        .value("GEO_ID", deme::CNT_OUTPUT_CONTENT::GEO_ID)
        .value("NICKNAME", deme::CNT_OUTPUT_CONTENT::NICKNAME)
        .export_values();
}
