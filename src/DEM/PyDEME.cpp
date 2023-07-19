#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <memory>
#include "API.h"
#include "AuxClasses.h"
#include <core/utils/DEMEPaths.h>
#include "VariableTypes.h"
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
#include <DEM/utils/Samplers.hpp>

namespace py = pybind11;

// Defining PyBind11 inclusion code

PYBIND11_MODULE(DEME, obj) {
    obj.def("GetDEMEDataFile", &deme::GetDEMEDataFile);

    /*
    py::class_<deme::Sampler>(obj, "Sampler")
        .def(py::init<float>())
        .def("SampleBox", static_cast<std::vector<std::vector<float>> (deme::Sampler::*)(
                              const std::vector<float>&, const std::vector<float>&)>(&deme::Sampler::SampleBox))
        .def("SampleSphere",
             static_cast<std::vector<std::vector<float>> (deme::Sampler::*)(const std::vector<float>&, float)>(
                 &deme::Sampler::SampleSphere))
        .def("SampleCylinderX",
             static_cast<std::vector<std::vector<float>> (deme::Sampler::*)(const std::vector<float>&, float, float)>(
                 &deme::Sampler::SampleCylinderX))
        .def("SampleCylinderY",
             static_cast<std::vector<std::vector<float>> (deme::Sampler::*)(const std::vector<float>&, float, float)>(
                 &deme::Sampler::SampleCylinderY))
        .def("SampleCylinderZ",
             static_cast<std::vector<std::vector<float>> (deme::Sampler::*)(const std::vector<float>&, float, float)>(
                 &deme::Sampler::SampleCylinderZ))
        .def("GetSeparation", &deme::Sampler::GetSeparation)
        .def("SetSeparation", &deme::Sampler::SetSeparation);
     */
    py::class_<deme::GridSampler>(obj, "GridSampler")
        .def(py::init<float>())
        .def("SetSeparation", &deme::GridSampler::SetSeparation)
        .def("SampleBox",
             static_cast<std::vector<std::vector<float>> (deme::GridSampler::*)(
                 const std::vector<float>& center, const std::vector<float>& halfDim)>(&deme::GridSampler::SampleBox),
             "Generates a sample box")
        .def("SampleSphere",
             static_cast<std::vector<std::vector<float>> (deme::GridSampler::*)(const std::vector<float>&, float)>(
                 &deme::GridSampler::SampleSphere))
        .def("SampleCylinderX", static_cast<std::vector<std::vector<float>> (deme::GridSampler::*)(
                                    const std::vector<float>&, float, float)>(&deme::GridSampler::SampleCylinderX))
        .def("SampleCylinderY", static_cast<std::vector<std::vector<float>> (deme::GridSampler::*)(
                                    const std::vector<float>&, float, float)>(&deme::GridSampler::SampleCylinderY))
        .def("SampleCylinderZ", static_cast<std::vector<std::vector<float>> (deme::GridSampler::*)(
                                    const std::vector<float>&, float, float)>(&deme::GridSampler::SampleCylinderZ))
        .def("GetSeparation", &deme::Sampler::GetSeparation);

    py::class_<deme::HCPSampler>(obj, "HCPSampler")
        .def(py::init<float>())

        .def("SampleBox",
             static_cast<std::vector<std::vector<float>> (deme::HCPSampler::*)(
                 const std::vector<float>& center, const std::vector<float>& halfDim)>(&deme::HCPSampler::SampleBox),
             "Generates a sample box")
        .def("SetSeparation", &deme::HCPSampler::SetSeparation)
        .def("SampleSphere",
             static_cast<std::vector<std::vector<float>> (deme::HCPSampler::*)(const std::vector<float>&, float)>(
                 &deme::HCPSampler::SampleSphere))
        .def(
            "SampleCylinderX",
            static_cast<std::vector<std::vector<float>> (deme::HCPSampler::*)(const std::vector<float>&, float, float)>(
                &deme::HCPSampler::SampleCylinderX))
        .def(
            "SampleCylinderY",
            static_cast<std::vector<std::vector<float>> (deme::HCPSampler::*)(const std::vector<float>&, float, float)>(
                &deme::HCPSampler::SampleCylinderY))
        .def(
            "SampleCylinderZ",
            static_cast<std::vector<std::vector<float>> (deme::HCPSampler::*)(const std::vector<float>&, float, float)>(
                &deme::HCPSampler::SampleCylinderZ))
        .def("GetSeparation", &deme::HCPSampler::GetSeparation);

    py::class_<deme::DEMInspector, std::shared_ptr<deme::DEMInspector>>(obj, "DEMInspector")
        .def(py::init<deme::DEMSolver*, deme::DEMDynamicThread*, const std::string&>())
        .def("GetValue", &deme::DEMInspector::GetValue);

    py::class_<deme::DEMInitializer, std::shared_ptr<deme::DEMInitializer>>(obj, "DEMInitializer").def(py::init<>());

    py::class_<deme::DEMTrackedObj, deme::DEMInitializer, std::shared_ptr<deme::DEMTrackedObj>>(obj, "DEMTrackedObj")
        .def(py::init<deme::DEMTrackedObj>());

    py::class_<deme::DEMTracker, std::shared_ptr<deme::DEMTracker>>(obj, "Tracker")
        .def(py::init<deme::DEMSolver*>())
        .def("SetPos",
             static_cast<void (deme::DEMTracker::*)(const std::vector<float>&, size_t)>(&deme::DEMTracker::SetPos))
        .def("GetContactAcc",
             static_cast<std::vector<float> (deme::DEMTracker::*)(size_t)>(&deme::DEMTracker::GetContactAcc));

    py::class_<deme::DEMForceModel>(obj, "DEMForceModel")
        .def(py::init<deme::FORCE_MODEL>())
        .def("SetForceModelType", &deme::DEMForceModel::SetForceModelType)
        .def("DefineCustomModel", &deme::DEMForceModel::DefineCustomModel)
        .def("ReadCustomModelFile", &deme::DEMForceModel::ReadCustomModelFile)
        .def("SetMustHaveMatProp", &deme::DEMForceModel::SetMustHaveMatProp)
        .def("SetMustPairwiseMatProp", &deme::DEMForceModel::SetMustPairwiseMatProp)
        .def("SetPerContactWildcards", &deme::DEMForceModel::SetPerContactWildcards)
        .def("SetPerOwnerWildcards", &deme::DEMForceModel::SetPerOwnerWildcards)
        .def("SetPerGeometryWildcards", &deme::DEMForceModel::SetPerGeometryWildcards);

    py::class_<deme::DEMSolver>(obj, "DEMSolver")
        .def(py::init<unsigned int>())
        .def("SetInitBinSize", &deme::DEMSolver::SetInitBinSize)
        .def("SetOutputFormat", &deme::DEMSolver::SetOutputFormat)
        .def("GetNumContacts", &deme::DEMSolver::GetNumContacts)
        .def("SetCDUpdateFreq", &deme::DEMSolver::SetCDUpdateFreq)
        .def("GetSimTime", &deme::DEMSolver::GetSimTime)
        .def("SetSimTime", &deme::DEMSolver::SetSimTime)
        .def("UpdateClumps", &deme::DEMSolver::UpdateClumps)
        .def("SetAdaptiveTimeStepType", &deme::DEMSolver::SetAdaptiveTimeStepType)
        .def("SetIntegrator",
             static_cast<void (deme::DEMSolver::*)(const std::string&)>(&deme::DEMSolver::SetIntegrator))
        .def("SetIntegrator",
             static_cast<void (deme::DEMSolver::*)(deme::TIME_INTEGRATOR)>(&deme::DEMSolver::SetIntegrator))
        .def("GetInitStatus", &deme::DEMSolver::GetInitStatus)
        .def("GetJitStringSubs", &deme::DEMSolver::GetJitStringSubs)
        .def("SetInitBinSize", &deme::DEMSolver::SetInitBinSize)
        .def("SetInitBinSizeAsMultipleOfSmallestSphere", &deme::DEMSolver::SetInitBinSizeAsMultipleOfSmallestSphere)
        .def("InstructNumOwners", &deme::DEMSolver::InstructNumOwners)
        .def("UseFrictionalHertzianModel", &deme::DEMSolver::UseFrictionalHertzianModel)
        .def("UseFrictionlessHertzianModel", &deme::DEMSolver::UseFrictionlessHertzianModel)
        .def("DefineContactForceModel", &deme::DEMSolver::DefineContactForceModel)
        .def("ReadContactForceModel", &deme::DEMSolver::ReadContactForceModel)
        .def("GetContactForceModel", &deme::DEMSolver::GetContactForceModel)
        .def("SetSortContactPairs", &deme::DEMSolver::SetSortContactPairs)
        .def("SetJitifyClumpTemplates", &deme::DEMSolver::SetJitifyClumpTemplates)
        .def("SetJitifyMassProperties", &deme::DEMSolver::SetJitifyMassProperties)
        .def("SetExpandFactor", &deme::DEMSolver::SetExpandFactor)
        .def("SetMaxVelocity", &deme::DEMSolver::SetMaxVelocity)
        .def("SetExpandSafetyType", &deme::DEMSolver::SetExpandSafetyType)
        .def("SetExpandSafetyMultiplier", &deme::DEMSolver::SetExpandSafetyMultiplier)
        .def("SetExpandSafetyAdder", &deme::DEMSolver::SetExpandSafetyAdder)
        .def("SetMaxSphereInBin", &deme::DEMSolver::SetMaxSphereInBin)
        .def("SetMaxTriangleInBin", &deme::DEMSolver::SetMaxTriangleInBin)
        .def("SetErrorOutVelocity", &deme::DEMSolver::SetErrorOutVelocity)
        .def("SetErrorOutAvgContacts", &deme::DEMSolver::SetErrorOutAvgContacts)
        .def("GetAvgSphContacts", &deme::DEMSolver::GetAvgSphContacts)
        .def("UseAdaptiveBinSize", &deme::DEMSolver::UseAdaptiveBinSize)
        .def("DisableAdaptiveBinSize", &deme::DEMSolver::DisableAdaptiveBinSize)
        .def("UseAdaptiveUpdateFreq", &deme::DEMSolver::UseAdaptiveUpdateFreq)
        .def("DisableAdaptiveUpdateFreq", &deme::DEMSolver::DisableAdaptiveUpdateFreq)
        .def("SetAdaptiveBinSizeDelaySteps", &deme::DEMSolver::SetAdaptiveBinSizeDelaySteps)
        .def("SetAdaptiveBinSizeMaxRate", &deme::DEMSolver::SetAdaptiveBinSizeMaxRate)
        .def("SetAdaptiveBinSizeAcc", &deme::DEMSolver::SetAdaptiveBinSizeAcc)
        .def("SetAdaptiveBinSizeUpperProactivity", &deme::DEMSolver::SetAdaptiveBinSizeUpperProactivity)
        .def("SetAdaptiveBinSizeLowerProactivity", &deme::DEMSolver::SetAdaptiveBinSizeLowerProactivity)
        .def("SetCDMaxUpdateFreq", &deme::DEMSolver::SetCDMaxUpdateFreq)
        .def("SetCDNumStepsMaxDriftAheadOfAvg", &deme::DEMSolver::SetCDNumStepsMaxDriftAheadOfAvg)
        .def("SetCDNumStepsMaxDriftMultipleOfAvg", &deme::DEMSolver::SetCDNumStepsMaxDriftMultipleOfAvg)
        .def("SetCDNumStepsMaxDriftHistorySize", &deme::DEMSolver::SetCDNumStepsMaxDriftHistorySize)
        .def("GetUpdateFreq", &deme::DEMSolver::GetUpdateFreq)
        .def("SetForceCalcThreadsPerBlock", &deme::DEMSolver::SetForceCalcThreadsPerBlock)
        .def("Duplicate", static_cast<std::shared_ptr<deme::DEMMaterial> (deme::DEMSolver::*)(
                              const std::shared_ptr<deme::DEMMaterial>&)>(&deme::DEMSolver::Duplicate))
        .def("Duplicate", static_cast<std::shared_ptr<deme::DEMClumpTemplate> (deme::DEMSolver::*)(
                              const std::shared_ptr<deme::DEMClumpTemplate>&)>(&deme::DEMSolver::Duplicate))
        .def("Duplicate", static_cast<std::shared_ptr<deme::DEMClumpBatch> (deme::DEMSolver::*)(
                              const std::shared_ptr<deme::DEMClumpBatch>&)>(&deme::DEMSolver::Duplicate))
        .def("AddExternalObject", &deme::DEMSolver::AddExternalObject)
        .def("SetOutputContent", &deme::DEMSolver::SetOutputContent)
        .def("SetMeshOutputFormat", &deme::DEMSolver::SetMeshOutputFormat)
        .def("SetContactOutputContent", &deme::DEMSolver::SetContactOutputContent)
        .def("SetVerbosity", &deme::DEMSolver::SetVerbosity,
             "Defines the desired verbosity level to be chosen from the VERBOSITY enumeration object")
        .def("DefineContactForceModel", &deme::DEMSolver::DefineContactForceModel)
        .def("LoadMaterial",
             static_cast<std::shared_ptr<deme::DEMMaterial> (deme::DEMSolver::*)(
                 const std::unordered_map<std::string, float>&)>(&deme::DEMSolver::LoadMaterial),
             "Loads in a DEMMaterial")
        .def("LoadMaterial",
             static_cast<std::shared_ptr<deme::DEMMaterial> (deme::DEMSolver::*)(deme::DEMMaterial&)>(
                 &deme::DEMSolver::LoadMaterial),
             "Loads in a DEMMaterial")
        .def("InstructBoxDomainDimension",
             static_cast<void (deme::DEMSolver::*)(float, float, float, const std::string&)>(
                 &deme::DEMSolver::InstructBoxDomainDimension),
             "Sets the Box Domain Dimension", py::arg("x"), py::arg("y"), py::arg("z"), py::arg("dir_exact") = "none")
        .def("InstructBoxDomainDimension",
             static_cast<void (deme::DEMSolver::*)(const std::pair<float, float>&, const std::pair<float, float>&,
                                                   const std::pair<float, float>&, const std::string& dir_exact)>(
                 &deme::DEMSolver::InstructBoxDomainDimension),
             "Sets the Box Domain Dimension")
        .def("InstructBoxDomainBoundingBC", &deme::DEMSolver::InstructBoxDomainBoundingBC,
             "Instruct if and how we should add boundaries to the simulation world upon initialization. Choose between "
             "`none', `all' (add 6 boundary planes) and `top_open' (add 5 boundary planes and leave the z-directon top "
             "open). Also specifies the material that should be assigned to those bounding boundaries.")
        .def("SetMaterialPropertyPair", &deme::DEMSolver::SetMaterialPropertyPair,
             "Defines a pair of DEMMaterial objects")
        .def("AddBCPlane",
             static_cast<std::shared_ptr<deme::DEMExternObj> (deme::DEMSolver::*)(
                 const std::vector<float>&, const std::vector<float>&, const std::shared_ptr<deme::DEMMaterial>&)>(
                 &deme::DEMSolver::AddBCPlane),
             "Add an (analytical or clump-represented) external object to the simulation system")
        .def("Track", (&deme::DEMSolver::PythonTrack), "Tracker object")
        .def("AddWavefrontMeshObject",
             static_cast<std::shared_ptr<deme::DEMMeshConnected> (deme::DEMSolver::*)(
                 const std::string&, const std::shared_ptr<deme::DEMMaterial>&, bool, bool)>(
                 &deme::DEMSolver::AddWavefrontMeshObject),
             "Load a mesh-represented object")
        .def("LoadClumpType",
             static_cast<std::shared_ptr<deme::DEMClumpTemplate> (deme::DEMSolver::*)(
                 float, const std::vector<float>&, const std::string, const std::shared_ptr<deme::DEMMaterial>&)>(
                 &deme::DEMSolver::LoadClumpType),
             "Load a clump type into the API-level cache")
        .def("AddClumps",
             static_cast<std::shared_ptr<deme::DEMClumpBatch> (deme::DEMSolver::*)(deme::DEMClumpBatch&)>(
                 &deme::DEMSolver::AddClumps),
             "Load input clumps (topology types and initial locations) on a per-pair basis. Note that the initial "
             "location means the location of the clumps' CoM coordinates in the global frame")
        .def("AddClumps",
             static_cast<std::shared_ptr<deme::DEMClumpBatch> (deme::DEMSolver::*)(
                 const std::vector<std::shared_ptr<deme::DEMClumpTemplate>>&, const std::vector<std::vector<float>>&)>(
                 &deme::DEMSolver::AddClumps),
             "Load input clumps (topology types and initial locations) on a per-pair basis. Note that the initial "
             "location means the location of the clumps' CoM coordinates in the global frame")

        .def("SetFamilyPrescribedAngVel",
             static_cast<void (deme::DEMSolver::*)(unsigned int ID, const std::string&, const std::string&,
                                                   const std::string&, bool dictate)>(
                 &deme::DEMSolver::SetFamilyPrescribedAngVel),
             "Set the prescribed angular velocity to all entities in a family. If dictate is set to true, then this "
             "family will not be fluenced by the force exerted from other simulation entites (both linear and "
             "rotational motions).")
        .def("SetFamilyPrescribedAngVel",
             static_cast<void (deme::DEMSolver::*)(unsigned int ID)>(&deme::DEMSolver::SetFamilyPrescribedAngVel),
             "Set the prescribed angular velocity to all entities in a family. If dictate is set to true, then this "
             "family will not be fluenced by the force exerted from other simulation entites (both linear and "
             "rotational motions).")
        .def("SetFamilyPrescribedLinVel",
             static_cast<void (deme::DEMSolver::*)(unsigned int ID)>(&deme::DEMSolver::SetFamilyPrescribedLinVel),
             "Set the prescribed linear velocity to all entities in a family. If dictate is set to true, then this "
             "family will not be influenced by the force exerted from other simulation entites (both linear and "
             "rotational motions).")
        .def("SetFamilyPrescribedLinVel",
             static_cast<void (deme::DEMSolver::*)(unsigned int, const std::string&, const std::string&,
                                                   const std::string&, bool)>(
                 &deme::DEMSolver::SetFamilyPrescribedLinVel),
             "Set the prescribed linear velocity to all entities in a family. If dictate is set to true, then this "
             "family will not be influenced by the force exerted from other simulation entites (both linear and "
             "rotational motions).")
        .def("SetFamilyPrescribedPosition",
             static_cast<void (deme::DEMSolver::*)(unsigned int ID)>(&deme::DEMSolver::SetFamilyPrescribedPosition),
             "Keep the positions of all entites in this family to remain exactly the user-specified values")
        .def("SetFamilyPrescribedPosition",
             static_cast<void (deme::DEMSolver::*)(unsigned int, const std::string&, const std::string&,
                                                   const std::string&, bool)>(
                 &deme::DEMSolver::SetFamilyPrescribedPosition),
             "Keep the positions of all entites in this family to remain exactly the user-specified values")
        .def("AddFamilyPrescribedAcc", &deme::DEMSolver::AddFamilyPrescribedAcc,
             "The entities in this family will always experienced an extra acceleration defined using this method")
        .def("CreateInspector",
             static_cast<std::shared_ptr<deme::DEMInspector> (deme::DEMSolver::*)(const std::string&)>(
                 &deme::DEMSolver::CreateInspector),
             "Create a inspector object that can help query some statistical info of the clumps in the simulation")
        .def("GetNumClumps", &deme::DEMSolver::GetNumClumps)
        .def("CreateInspector",
             static_cast<std::shared_ptr<deme::DEMInspector> (deme::DEMSolver::*)(
                 const std::string&, const std::string&)>(&deme::DEMSolver::CreateInspector),
             "Create a inspector object that can help query some statistical info of the clumps in the simulation")
        .def("SetInitTimeStep", &deme::DEMSolver::SetInitTimeStep,
             "Set the initial time step size. If using constant step size, then this will be used throughout; "
             "otherwise, the actual step size depends on the variable step strategy.")
        .def("SetGravitationalAcceleration",
             static_cast<void (deme::DEMSolver::*)(const std::vector<float>&)>(
                 &deme::DEMSolver::SetGravitationalAcceleration),
             "Set gravitational pull")
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
             "Show potential anomalies that may have been there in the simulation, then clear the anomaly log.");

    py::class_<deme::DEMMaterial, std::shared_ptr<deme::DEMMaterial>>(obj, "DEMMaterial")
        .def(py::init<const std::unordered_map<std::string, float>&>())
        .def_readwrite("mat_prop", &deme::DEMMaterial::mat_prop)
        .def_readwrite("load_order", &deme::DEMMaterial::load_order);

    py::class_<deme::DEMClumpTemplate, std::shared_ptr<deme::DEMClumpTemplate>>(obj, "DEMClumpTemplate")
        .def(py::init<>())
        .def("SetVolume", &deme::DEMClumpTemplate::SetVolume)
        .def("ReadComponentFromFile", &deme::DEMClumpTemplate::ReadComponentFromFile)
        .def("InformCentroidPrincipal",
             static_cast<void (deme::DEMClumpTemplate::*)(const std::vector<float>&, const std::vector<float>&)>(
                 &deme::DEMClumpTemplate::InformCentroidPrincipal))
        .def("Move",
             static_cast<void (deme::DEMClumpTemplate::*)(const std::vector<float>&, const std::vector<float>&)>(
                 &deme::DEMClumpTemplate::Move))
        .def("Scale", &deme::DEMClumpTemplate::Scale)
        .def("AssignName", &deme::DEMClumpTemplate::AssignName);

    py::class_<deme::DEMClumpBatch, deme::DEMInitializer, std::shared_ptr<deme::DEMClumpBatch>>(obj, "DEMClumpBatch")
        .def(py::init<size_t&>())
        .def("SetTypes",
             static_cast<void (deme::DEMClumpBatch::*)(const std::vector<std::shared_ptr<deme::DEMClumpTemplate>>&)>(
                 &deme::DEMClumpBatch::SetTypes))
        .def("SetTypes", static_cast<void (deme::DEMClumpBatch::*)(const std::shared_ptr<deme::DEMClumpTemplate>&)>(
                             &deme::DEMClumpBatch::SetTypes))
        .def("SetType", &deme::DEMClumpBatch::SetType)
        .def("SetVel", static_cast<void (deme::DEMClumpBatch::*)(const std::vector<std::vector<float>>&)>(
                           &deme::DEMClumpBatch::SetVel))
        .def("SetVel",
             static_cast<void (deme::DEMClumpBatch::*)(const std::vector<float>&)>(&deme::DEMClumpBatch::SetVel))
        //    .def("SetAngVel",
        //         static_cast<void (deme::DEMClumpBatch::*)(const
        //         std::vector<float3>&)>(&deme::DEMClumpBatch::SetAngVel))
        .def("SetFamilies", static_cast<void (deme::DEMClumpBatch::*)(const std::vector<unsigned int>&)>(
                                &deme::DEMClumpBatch::SetFamilies))
        .def("SetFamilies", static_cast<void (deme::DEMClumpBatch::*)(unsigned int)>(&deme::DEMClumpBatch::SetFamilies))
        .def("SetFamily", &deme::DEMClumpBatch::SetFamily)
        .def("SetExistingContacts", &deme::DEMClumpBatch::SetExistingContacts)
        .def("SetExistingContactWildcards", &deme::DEMClumpBatch::SetExistingContactWildcards)
        .def("AddExistingContactWildcard", &deme::DEMClumpBatch::AddExistingContactWildcard)
        .def("SetOwnerWildcards", &deme::DEMClumpBatch::SetOwnerWildcards)
        .def("AddOwnerWildcard",
             static_cast<void (deme::DEMClumpBatch::*)(const std::string&, const std::vector<float>&)>(
                 &deme::DEMClumpBatch::AddOwnerWildcard))
        .def("AddOwnerWildcard", static_cast<void (deme::DEMClumpBatch::*)(const std::string&, float)>(
                                     &deme::DEMClumpBatch::AddOwnerWildcard))
        .def("SetGeometryWildcards", &deme::DEMClumpBatch::SetGeometryWildcards)
        .def("AddGeometryWildcard",
             static_cast<void (deme::DEMClumpBatch::*)(const std::string&, const std::vector<float>&)>(
                 &deme::DEMClumpBatch::AddGeometryWildcard))
        .def("AddGeometryWildcard", static_cast<void (deme::DEMClumpBatch::*)(const std::string&, float)>(
                                        &deme::DEMClumpBatch::AddGeometryWildcard))
        .def("GetNumContacts", &deme::DEMClumpBatch::GetNumContacts);

    py::class_<deme::DEMExternObj, deme::DEMInitializer, std::shared_ptr<deme::DEMExternObj>>(obj, "DEMExternObj")
        .def(py::init<>())
        .def("SetFamily", &deme::DEMExternObj::SetFamily, "Defines an object contact family number")
        .def("SetMass", &deme::DEMExternObj::SetMass, "Sets the mass of this object")
        .def("SetMOI",
             static_cast<void (deme::DEMExternObj::*)(const std::vector<float>&)>(&deme::DEMExternObj::SetMOI),
             "Sets the MOI (in the principal frame)")
        .def("SetInitQuat",
             static_cast<void (deme::DEMExternObj::*)(const std::vector<float>&)>(&deme::DEMExternObj::SetInitQuat),
             "Set the initial quaternion for this object (before simulation initializes).")
        .def("SetInitPos",
             static_cast<void (deme::DEMExternObj::*)(const std::vector<float>&)>(&deme::DEMExternObj::SetInitPos),
             "Set the initial position for this object (before simulation initializes).")
        .def("AddPlane",
             static_cast<void (deme::DEMExternObj::*)(const std::vector<float>&, const std::vector<float>&,
                                                      const std::shared_ptr<deme::DEMMaterial>&)>(
                 &deme::DEMExternObj::AddPlane),
             "Add a plane with infinite size.")
        //.def("AddPlate", static_cast<void (&deme::DEMExternObj::AddPlate, "Add a plate with finite size.")
        .def("AddZCylinder",
             static_cast<void (deme::DEMExternObj::*)(const std::vector<float>&, const float,
                                                      const std::shared_ptr<deme::DEMMaterial>&,
                                                      const deme::objNormal_t)>(&deme::DEMExternObj::AddZCylinder),
             "Add a z-axis-aligned cylinder of infinite length")
        .def("AddCylinder",
             static_cast<void (deme::DEMExternObj::*)(const std::vector<float>&, const std::vector<float>&, const float,
                                                      const std::shared_ptr<deme::DEMMaterial>&,
                                                      const deme::objNormal_t)>(&deme::DEMExternObj::AddCylinder),
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

    py::class_<deme::DEMMeshConnected, deme::DEMInitializer, std::shared_ptr<deme::DEMMeshConnected>>(
        obj, "DEMMeshConnected")
        .def(py::init<>())
        .def(py::init<std::string&>())
        .def(py::init<std::string, const std::shared_ptr<deme::DEMMaterial>&>())
        .def("Clear", &deme::DEMMeshConnected::Clear, "Clears everything from memory")
        .def("LoadWavefrontMesh", &deme::DEMMeshConnected::LoadWavefrontMesh,
             "Load a triangle mesh saved as a Wavefront .obj file")
        .def("WriteWavefront", &deme::DEMMeshConnected::WriteWavefront,
             "Write the specified meshes in a Wavefront .obj file")
        .def("GetNumTriangles", &deme::DEMMeshConnected::GetNumTriangles,
             "Get the number of triangles already added to this mesh")
        .def("GetNumNodes", &deme::DEMMeshConnected::GetNumNodes, "Get the number of nodes in the mesh")
        .def("UseNormals", &deme::DEMMeshConnected::UseNormals,
             "Instruct that when the mesh is initialized into the system, it will re-order the nodes of each triangle "
             "so that the normals derived from right-hand-rule are the same as the normals in the mesh file")
        .def("GetTriangle", &deme::DEMMeshConnected::GetTriangle, "Access the n-th triangle in mesh")
        .def("SetMass", &deme::DEMMeshConnected::SetMass)
        .def("SetMOI",
             static_cast<void (deme::DEMMeshConnected::*)(const std::vector<float>&)>(&deme::DEMMeshConnected::SetMOI))
        .def("SetFamily", &deme::DEMMeshConnected::SetFamily)
        .def("SetMaterial", static_cast<void (deme::DEMMeshConnected::*)(const std::shared_ptr<deme::DEMMaterial>&)>(
                                &deme::DEMMeshConnected::SetMaterial))
        .def("SetMaterial",
             static_cast<void (deme::DEMMeshConnected::*)(const std::vector<std::shared_ptr<deme::DEMMaterial>>&)>(
                 &deme::DEMMeshConnected::SetMaterial))
        .def("SetInitQuat", static_cast<void (deme::DEMMeshConnected::*)(const std::vector<float>&)>(
                                &deme::DEMMeshConnected::SetInitQuat))
        .def("SetInitPos", static_cast<void (deme::DEMMeshConnected::*)(const std::vector<float>&)>(
                               &deme::DEMMeshConnected::SetInitPos))
        .def("InformCentroidPrincipal",
             static_cast<void (deme::DEMMeshConnected::*)(const std::vector<float>&, const std::vector<float>&)>(
                 &deme::DEMMeshConnected::InformCentroidPrincipal))
        .def("Move",
             static_cast<void (deme::DEMMeshConnected::*)(const std::vector<float>&, const std::vector<float>&)>(
                 &deme::DEMMeshConnected::Move))
        .def("Mirror",
             static_cast<void (deme::DEMMeshConnected::*)(const std::vector<float>&, const std::vector<float>&)>(
                 &deme::DEMMeshConnected::Mirror))
        .def("Scale", static_cast<void (deme::DEMMeshConnected::*)(float)>(&deme::DEMMeshConnected::Scale))
        .def("ClearWildcards", &deme::DEMMeshConnected::ClearWildcards)
        .def("SetGeometryWildcards", &deme::DEMMeshConnected::SetGeometryWildcards)
        .def("AddGeometryWildcard",
             static_cast<void (deme::DEMMeshConnected::*)(const std::string&, const std::vector<float>&)>(
                 &deme::DEMMeshConnected::AddGeometryWildcard))
        .def("AddGeometryWildcard", static_cast<void (deme::DEMMeshConnected::*)(const std::string&, float)>(
                                        &deme::DEMMeshConnected::AddGeometryWildcard));

    //// TODO: Insert readwrite functions to access all public class objects!

    py::enum_<deme::VERBOSITY>(obj, "VERBOSITY")
        .value("QUIET", deme::VERBOSITY::QUIET)
        .value("ERROR", deme::VERBOSITY::ERROR)
        .value("WARNING", deme::VERBOSITY::WARNING)
        .value("INFO", deme::VERBOSITY::INFO)
        .value("STEP_ANOMALY", deme::VERBOSITY::STEP_ANOMALY)
        .value("STEP_METRIC", deme::VERBOSITY::STEP_METRIC)
        .value("DEBUG", deme::VERBOSITY::DEBUG)
        .value("STEP_DEBUG", deme::VERBOSITY::STEP_DEBUG)
        .export_values();

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
        .value("GEO_WILDCARD", deme::OUTPUT_CONTENT::GEO_WILDCARD)
        .value("EXP_FACTOR", deme::OUTPUT_CONTENT::EXP_FACTOR)
        .export_values();

    py::enum_<deme::OUTPUT_FORMAT>(obj, "OUTPUT_FORMAT")
        .value("CSV", deme::OUTPUT_FORMAT::CSV)
        .value("BINARY", deme::OUTPUT_FORMAT::BINARY)
        .value("CHPF", deme::OUTPUT_FORMAT::CHPF)
        .export_values();

    py::enum_<deme::MESH_FORMAT>(obj, "MESH_FORMAT")
        .value("VTK", deme::MESH_FORMAT::VTK)
        .value("OBJ", deme::MESH_FORMAT::OBJ)
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
        .value("POINT", deme::CNT_OUTPUT_CONTENT::POINT)
        .value("COMPONENT", deme::CNT_OUTPUT_CONTENT::COMPONENT)
        .value("NORMAL", deme::CNT_OUTPUT_CONTENT::NORMAL)
        .value("TORQUE", deme::CNT_OUTPUT_CONTENT::TORQUE)
        .value("CNT_WILDCARD", deme::CNT_OUTPUT_CONTENT::CNT_WILDCARD)
        .value("OWNER", deme::CNT_OUTPUT_CONTENT::OWNER)
        .value("GEO_ID", deme::CNT_OUTPUT_CONTENT::GEO_ID)
        .value("NICKNAME", deme::CNT_OUTPUT_CONTENT::NICKNAME)
        .export_values();
}