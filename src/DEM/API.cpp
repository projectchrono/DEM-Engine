#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <memory>
#include "API.h"

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

PYBIND11_MODULE(DEME, obj) {
    py::class_<deme::DEMSolver>(obj, "DEMSolver")
        .def(py::init<unsigned int&>())
        .def("__del__", &deme::DEMSolver::~DEMSolver, "Links python deletion function to DEMSolver's destructor")
        .def("SetVerbosity", &deme::DEMSolver::SetVerbosity
                             "Defines the desired verbosity level to be chosen from the VERBOSITY enumeration object")
        .def("LoadMaterial", &deme::DEMSolver::LoadMaterial, "Loads in a DEMMaterial")
        .def("InstructBoxDomainDimension", &deme::DEMSolver::InstructBoxDomainDimension,
             "Sets the Box Domain Dimension")
        .def("InstructBoxDomainBoundingBC", &deme::DEMSolver::InstructBoxDomainBoundingBC,
             "Instruct if and how we should add boundaries to the simulation world upon initialization. Choose between "
             "`none', `all' (add 6 boundary planes) and `top_open' (add 5 boundary planes and leave the z-directon top "
             "open). Also specifies the material that should be assigned to those bounding boundaries.")
        .def("SetMaterialPropertyPair", &deme::DEMSolver::SetMaterialPropertyPair,
             "Defines a pair of DEMMaterial objects")
        .def("AddBCPlane", &deme::DEMSolver::AddBCPlane,
             "Add an (analytical or clump-represented) external object to the simulation system")
        .def("Track", &deme::DEMSolver::Track,
             "Create a DEMTracker to allow direct control/modification/query to this external object")
        .def("AddWavefrontMeshObject", &deme::DEMSolver::AddWavefrontMeshObject, "Load a mesh-represented object")
        .def("LoadClumpType", &deme::DEMSolver::LoadClumpType, "Load a clump type into the API-level cache")
        .def("AddClumps", &deme::DEMSolver::AddClumps,
             "Load input clumps (topology types and initial locations) on a per-pair basis. Note that the initial "
             "location means the location of the clumps' CoM coordinates in the global frame")
        .def("SetFamilyPrescribedAngVel", &deme::DEMSolver::SetFamilyPrescribedAngVel,
             "Set the prescribed angular velocity to all entities in a family. If dictate is set to true, then this "
             "family will not be fluenced by the force exerted from other simulation entites (both linear and "
             "rotational motions).")
        .def("SetFamilyPrescribedLinVel", &deme::DEMSolver::SetFamilyPrescribedLinVel,
             "Set the prescribed linear velocity to all entities in a family. If dictate is set to true, then this "
             "family will not be influenced by the force exerted from other simulation entites (both linear and "
             "rotational motions).")
        .def("SetFamilyPrescribedPosition", &deme::DEMSolver::SetFamilyPrescribedPosition,
             "Keep the positions of all entites in this family to remain exactly the user-specified values.")
        .def("SetFamilyPrescribedQuaternion", &deme::DEMSolver::SetFamilyPrescribedQuaternion,
             "Keep the orientation quaternions of all entites in this family to remain exactly the user-specified "
             "values")
        .def("AddFamilyPrescribedAcc", &deme::DEMSolver::AddFamilyPrescribedAcc,
             "The entities in this family will always experienced an extra acceleration defined using this method")
        .def("CreateInspector", &deme::DEMSolver::CreateInspector,
             "Create a inspector object that can help query some statistical info of the clumps in the simulation")
        .def("SetInitTimeStep", &deme::DEMSolver::SetInitTimeStep,
             "Set the initial time step size. If using constant step size, then this will be used throughout; "
             "otherwise, the actual step size depends on the variable step strategy.")
        .def("SetGravitationalAcceleration", &deme::DEMSolver::SetGravitationalAcceleration, "Set gravitational pull")
        .def("SetMaxVelocity", &deme::DEMSolver::SetMaxVelocity, "Sets the maximum velocity")
        .def("SetErrorOutVelocity", &deme::DEMSolver::SetErrorOutVelocity, "Sets the Error out velocity")
        .def("SetExpandSafetyMultiplier", &deme::DEMSolver::SetExpandSafetyMultiplier,
             "Assign a multiplier to our estimated maximum system velocity, when deriving the thinckness of the "
             "contact `safety' margin")
        .def("Initialize", &deme::DEMSolver::Initialize, "Initializes the system")
        .def("WriteSphereFile", &deme::DEMSolver::WriteSphereFile, "Writes a sphere file")
        .def("WriteMeshFile", &deme::DEMSolver::WriteMeshFile, "Write the current status of all meshes to a file")
        .def("DoDynamics", &deme::DEMSolver::DoDynamics,
             "Advance simulation by this amount of time, and at the end of this call, synchronize kT and dT. This is "
             "suitable for a longer call duration and without co-simulation.")
        .def("DoDynamicsThenSync", &deme::DEMSolver::DoDynamicsThenSync,
             "Advance simulation by this amount of time, and at the end of this call, synchronize kT and dT. This is "
             "suitable for a longer call duration and without co-simulation.")
        .def("ChangeFamily", &deme::DEMSolver::ChangeFamily,
             "Change all entities with family number ID_from to have a new number ID_to, when the condition defined by "
             "the string is satisfied by the entities in question. This should be called before initialization, and "
             "will be baked into the solver, so the conditions will be checked and changes applied every time step.")
        .def("ShowThreadCollaborationStats", &deme::DEMSolver::ShowThreadCollaborationStats,
             "Show the collaboration stats between dT and kT. This is more useful for tweaking the number of time "
             "steps that dT should be allowed to be in advance of kT.")
        .def("ShowTimingStats", &deme::DEMSolver::ShowTimingStats,
             "Show the wall time and percentages of wall time spend on various solver tasks")
        .def("ShowAnomalies", &deme::DEMSolver::ShowAnomalies,
             "Show potential anomalies that may have been there in the simulation, then clear the anomaly log.")
        .def_readwrite("types", &deme::DEMSolver::types)
        .def_readwrite("materials", &deme::DEMSolver::materials)
        .def_readwrite("family_code", &deme::DEMSolver::family_code)
        .def_readwrite("init_pos", &deme::DEMSolver::init_pos)
        .def_readwrite("init_oriQ", &deme::DEMSolver::init_oriQ)
        .def_readwrite("mass", &deme::DEMSolver::mass)
        .def_readwrite("MOI", &deme::DEMSolver::MOI)
        .def_readwrite("load_order", &deme::DEMSolver::load_order)
        .def_readwrite("entity_params", &deme::DEMSolver::entity_params);
}