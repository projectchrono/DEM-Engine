//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <DEM/API.h>
#include <DEM/DEMAuxClasses.h>
#include <DEM/HostSideHelpers.hpp>

namespace sgps {

// =============================================================================
// DEMInspector class
// =============================================================================

const std::string DEM_INSP_CODE_SPHERE_HIGH_Z = R"V0G0N(
    quantity[sphereID] = sphereZ + myRadius;
)V0G0N";

const std::string DEM_INSP_CODE_SPHERE_LOW_Z = R"V0G0N(
    quantity[sphereID] = sphereZ - myRadius;
)V0G0N";

void DEMInspector::switch_quantity_type(const std::string& quantity) {
    switch (hash_charr(quantity.c_str())) {
        case ("max_z"_):
            inspection_code = DEM_INSP_CODE_SPHERE_HIGH_Z;
            reduce_flavor = DEM_CUB_REDUCE_FLAVOR::MAX;
            break;
        case ("min_z"_):
            inspection_code = DEM_INSP_CODE_SPHERE_LOW_Z;
            reduce_flavor = DEM_CUB_REDUCE_FLAVOR::MIN;
            break;
        //// TODO: void ratio will have its own query function
        // case "void_ratio"_:
        //     break;
        case ("absv"_):
            // inspection_code = ;
            reduce_flavor = DEM_CUB_REDUCE_FLAVOR::NONE;
            break;
        default:
            std::stringstream ss;
            ss << quantity << " is not a known query type." << std::endl;
            throw std::runtime_error(ss.str());
    }
}
void DEMInspector::switch_kernel_to_call(DEM_INSPECT_ENTITY_TYPE insp_type) {
    switch (insp_type) {
        case (DEM_INSPECT_ENTITY_TYPE::SPHERE):
            kernel_name = "inspectSphereProperty";
            break;
        case (DEM_INSPECT_ENTITY_TYPE::CLUMP):
            kernel_name = "inspectClumpProperty";
            break;
        default:
            std::stringstream ss;
            ss << "You should select inspection entity type between SPHERE and CLUMP." << std::endl;
            throw std::runtime_error(ss.str());
    }
}

float DEMInspector::GetValue() {
    return sys->dTInspectReduce(inspection_kernel, kernel_name, thing_to_insp, reduce_flavor, all_domain);
}

void DEMInspector::initializeInspector(const std::unordered_map<std::string, std::string>& Subs) {
    // Add own substitutions to it
    {
        std::unordered_map<std::string, std::string> my_subs = Subs;
        my_subs["_inRegionPolicy_"] = in_region_code;
        my_subs["_quantityQueryProcess_"] = inspection_code;
        inspection_kernel = std::make_shared<jitify::Program>(
            std::move(JitHelper::buildProgram("DEMQueryKernels", JitHelper::KERNEL_DIR / "DEMQueryKernels.cu", my_subs,
                                              {"-I" + (JitHelper::KERNEL_DIR / "..").string()})));
    }
    initialized = true;
}

// =============================================================================
// DEMTracker class
// =============================================================================

float3 DEMTracker::Pos(size_t offset) {
    return sys->GetOwnerPosition(obj->ownerID + offset);
}
float3 DEMTracker::AngVel(size_t offset) {
    return sys->GetOwnerAngVel(obj->ownerID + offset);
}
float3 DEMTracker::Vel(size_t offset) {
    return sys->GetOwnerVelocity(obj->ownerID + offset);
}
float4 DEMTracker::OriQ(size_t offset) {
    return sys->GetOwnerOriQ(obj->ownerID + offset);
}
void DEMTracker::SetPos(float3 pos, size_t offset) {
    sys->SetOwnerPosition(obj->ownerID + offset, pos);
}
void DEMTracker::SetAngVel(float3 angVel, size_t offset) {
    sys->SetOwnerAngVel(obj->ownerID + offset, angVel);
}
void DEMTracker::SetVel(float3 vel, size_t offset) {
    sys->SetOwnerVelocity(obj->ownerID + offset, vel);
}
void DEMTracker::SetOriQ(float4 oriQ, size_t offset) {
    sys->SetOwnerOriQ(obj->ownerID + offset, oriQ);
}
void DEMTracker::ChangeClumpSizes(const std::vector<bodyID_t>& IDs, const std::vector<float>& factors) {
    std::vector<bodyID_t> offsetted_IDs(IDs);
    size_t offset = obj->ownerID;
    std::for_each(offsetted_IDs.begin(), offsetted_IDs.end(), [offset](bodyID_t& x) { x += offset; });
    sys->ChangeClumpSizes(offsetted_IDs, factors);
}

}  // END namespace sgps
