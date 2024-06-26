//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <DEM/API.h>
#include <DEM/AuxClasses.h>
#include <DEM/HostSideHelpers.hpp>
#include <DEM/Models.h>

namespace deme {

// =============================================================================
// DEMInspector class
// =============================================================================

const std::string INSP_CODE_SPHERE_HIGH_Z = R"V0G0N(
    quantity[sphereID] = Z + myRadius;
)V0G0N";

const std::string INSP_CODE_SPHERE_LOW_Z = R"V0G0N(
    quantity[sphereID] = Z - myRadius;
)V0G0N";

const std::string INSP_CODE_SPHERE_HIGH_ABSV = R"V0G0N(
    float3 relPos = myRelPos;
    // Get owner's velocity
    float3 rotVel, linVel;
    linVel.x = granData->vX[myOwner];
    linVel.y = granData->vY[myOwner];
    linVel.z = granData->vZ[myOwner];
    // rotVel is local
    rotVel.x = granData->omgBarX[myOwner];
    rotVel.y = granData->omgBarY[myOwner];
    rotVel.z = granData->omgBarZ[myOwner];

    float vel;
    {
        // It is indeed an estimation, since it accounts for center of sphere
        // only. But interestingly, a sphere's rotation about its own CoM does
        // not contribute to the size of contact detection margin, which is the
        // main reason for querying max absv for us. So, it should be fine.
        float3 pRotVel = cross(rotVel, relPos);
        // Map rotational contribution back to global
        applyOriQToVector3<float, deme::oriQ_t>(pRotVel.x, pRotVel.y, pRotVel.z, 
                                                oriQw, oriQx, oriQy, oriQz);
        vel = length(pRotVel + linVel);
    }
    quantity[sphereID] = vel;
)V0G0N";

const std::string INSP_CODE_EVERYTHING_ABSV = R"V0G0N(
    double myVX = granData->vX[myOwner];
    double myVY = granData->vY[myOwner];
    double myVZ = granData->vZ[myOwner];
    double myABSV = sqrt(myVX * myVX + myVY * myVY + myVZ * myVZ);

    quantity[myOwner] = myABSV;
)V0G0N";

const std::string INSP_CODE_CLUMP_KE = R"V0G0N(
    // First lin energy
    double myVX = granData->vX[myOwner];
    double myVY = granData->vY[myOwner];
    double myVZ = granData->vZ[myOwner];
    double myKE = 0.5 * myMass * (myVX * myVX + myVY * myVY + myVZ * myVZ);
    // Then rot energy
    myVX = granData->omgBarX[myOwner];
    myVY = granData->omgBarY[myOwner];
    myVZ = granData->omgBarZ[myOwner];
    myKE += 0.5 * ((double)myMOI.x * myVX * myVX + (double)myMOI.y * myVY * myVY + (double)myMOI.z * myVZ * myVZ);

    quantity[myOwner] = myKE;
)V0G0N";

const std::string INSP_CODE_CLUMP_APPROX_VOL = R"V0G0N(
    size_t myVolOffset = granData->inertiaPropOffsets[myOwner];
    float myVol = volumeProperties[myVolOffset];
    quantity[myOwner] = myVol;
)V0G0N";

const std::string INSP_CODE_CLUMP_APPROX_MASS = R"V0G0N(
    quantity[myOwner] = myMass;
)V0G0N";

void DEMInspector::switch_quantity_type(const std::string& quantity) {
    switch (hash_charr(quantity.c_str())) {
        case ("clump_max_z"_):
            inspection_code = INSP_CODE_SPHERE_HIGH_Z;
            reduce_flavor = CUB_REDUCE_FLAVOR::MAX;
            kernel_name = "inspectSphereProperty";
            thing_to_insp = INSPECT_ENTITY_TYPE::SPHERE;
            index_name = "sphereID";
            break;
        case ("clump_min_z"_):
            inspection_code = INSP_CODE_SPHERE_LOW_Z;
            reduce_flavor = CUB_REDUCE_FLAVOR::MIN;
            kernel_name = "inspectSphereProperty";
            thing_to_insp = INSPECT_ENTITY_TYPE::SPHERE;
            index_name = "sphereID";
            break;
        case ("clump_max_absv"_):
            inspection_code = INSP_CODE_SPHERE_HIGH_ABSV;
            reduce_flavor = CUB_REDUCE_FLAVOR::MAX;
            kernel_name = "inspectSphereProperty";
            thing_to_insp = INSPECT_ENTITY_TYPE::SPHERE;
            index_name = "sphereID";
            break;
        // case ("mesh_max_z"_):
        //     reduce_flavor = CUB_REDUCE_FLAVOR::MAX;
        //     break;
        // case ("clump_com_max_z"_):
        //     // inspection_code = INSP_CODE_SPHERE_HIGH_Z;
        //     reduce_flavor = CUB_REDUCE_FLAVOR::MAX;
        //     kernel_name = "inspectOwnerProperty";
        //     break;
        //// TODO: Void ration here is a very rough approximation and should only work when domain is large and
        /// particles are small
        case ("clump_volume"_):
            inspection_code = INSP_CODE_CLUMP_APPROX_VOL;
            reduce_flavor = CUB_REDUCE_FLAVOR::SUM;
            kernel_name = "inspectOwnerProperty";
            thing_to_insp = INSPECT_ENTITY_TYPE::CLUMP;
            index_name = "myOwner";
            all_domain = false;  // Only clumps, so not all domain owners
            break;
        case ("clump_mass"_):
            inspection_code = INSP_CODE_CLUMP_APPROX_MASS;
            reduce_flavor = CUB_REDUCE_FLAVOR::SUM;
            kernel_name = "inspectOwnerProperty";
            thing_to_insp = INSPECT_ENTITY_TYPE::CLUMP;
            index_name = "myOwner";
            all_domain = false;  // Only clumps, so not all domain owners
            break;
        case ("max_absv"_):
            inspection_code = INSP_CODE_EVERYTHING_ABSV;
            reduce_flavor = CUB_REDUCE_FLAVOR::MAX;
            kernel_name = "inspectOwnerProperty";
            thing_to_insp = INSPECT_ENTITY_TYPE::EVERYTHING;
            index_name = "myOwner";
            break;
        case ("absv"_):
            inspection_code = INSP_CODE_EVERYTHING_ABSV;
            reduce_flavor = CUB_REDUCE_FLAVOR::NONE;
            kernel_name = "inspectOwnerProperty";
            thing_to_insp = INSPECT_ENTITY_TYPE::EVERYTHING;
            index_name = "myOwner";
            break;
        case ("clump_kinetic_energy"_):
            inspection_code = INSP_CODE_CLUMP_KE;
            reduce_flavor = CUB_REDUCE_FLAVOR::SUM;
            kernel_name = "inspectOwnerProperty";
            thing_to_insp = INSPECT_ENTITY_TYPE::CLUMP;
            index_name = "myOwner";
            all_domain = false;  // Only clumps, so not all domain owners
            break;
        default:
            std::stringstream ss;
            ss << quantity << " is not a known query type." << std::endl;
            throw std::runtime_error(ss.str());
    }
}

float DEMInspector::GetValue() {
    assertInit();
    float reduce_result =
        sys->dTInspectReduce(inspection_kernel, kernel_name, thing_to_insp, reduce_flavor, all_domain);
    return reduce_result;
}

float* DEMInspector::GetValues() {
    assertInit();
    float* reduce_result =
        sys->dTInspectNoReduce(inspection_kernel, kernel_name, thing_to_insp, reduce_flavor, all_domain);
    return reduce_result;
}

float* DEMInspector::dT_GetValue() {
    // assertInit(); // This one the user should not use
    return dT->inspectCall(inspection_kernel, kernel_name, thing_to_insp, reduce_flavor, all_domain);
}

void DEMInspector::assertInit() {
    if (!initialized) {
        Initialize(sys->GetJitStringSubs());
    }
}

void DEMInspector::Initialize(const std::unordered_map<std::string, std::string>& Subs, bool force) {
    if (!(sys->GetInitStatus()) && !force) {
        std::stringstream ss;
        ss << "Inspector should only be initialized or used after the simulation system is initialized (because it "
              "uses device-side data)!"
           << std::endl;
        throw std::runtime_error(ss.str());
    }
    // We want to make sure if the in_region_code is legit, if it is not an all_domain query
    std::string in_region_specifier = in_region_code, placeholder;
    // But if the in_region_code is all spaces, it's fine, probably they don't care
    if ((!all_domain) && (!is_all_spaces(in_region_code))) {
        if (!any_whole_word_match(in_region_code, {"X", "Y", "Z"}) ||
            !all_whole_word_match(in_region_code, {"return"}, placeholder)) {
            std::stringstream ss;
            ss << "One of your insepctors is set to query a specific region, but the domian is not properly "
                  "defined.\nIt needs to return a bool variable that is a result of logical operations involving X, Y "
                  "and Z.\nYou can remove the region argument if all simulation entities should be considered."
               << std::endl;
            throw std::runtime_error(ss.str());
        }
        // Replace the return with our own variable
        in_region_specifier = replace_pattern(in_region_specifier, "return", "bool isInRegion = ");
        in_region_specifier += "if (!isInRegion) { not_in_region[" + index_name + "] = 1; return; }\n";
    }

    // Add own substitutions to it
    std::unordered_map<std::string, std::string> my_subs = Subs;
    my_subs["_inRegionPolicy_"] = in_region_specifier;
    my_subs["_quantityQueryProcess_"] = inspection_code;
    if (thing_to_insp == INSPECT_ENTITY_TYPE::SPHERE) {
        inspection_kernel = std::make_shared<jitify::Program>(std::move(
            JitHelper::buildProgram("DEMSphereQueryKernels", JitHelper::KERNEL_DIR / "DEMSphereQueryKernels.cu",
                                    my_subs, DEME_JITIFY_OPTIONS)));
    } else if (thing_to_insp == INSPECT_ENTITY_TYPE::CLUMP || thing_to_insp == INSPECT_ENTITY_TYPE::EVERYTHING) {
        inspection_kernel = std::make_shared<jitify::Program>(std::move(JitHelper::buildProgram(
            "DEMOwnerQueryKernels", JitHelper::KERNEL_DIR / "DEMOwnerQueryKernels.cu", my_subs, DEME_JITIFY_OPTIONS)));
    } else {
        std::stringstream ss;
        ss << "Sorry, an inspector object you are using is not implemented yet.\nConsider letting the developers know "
              "this and they may help you."
           << std::endl;
        throw std::runtime_error(ss.str());
    }
    initialized = true;
}

// =============================================================================
// DEMTracker class
// =============================================================================

void DEMTracker::assertThereIsForcePairs(const std::string& name) {
    if (sys->GetWhetherForceCollectInKernel()) {
        std::stringstream ss;
        ss << "The solver is currently set to not record force pair info, so you cannot query force pairs using "
           << name
           << ".\nYou can call SetCollectAccRightAfterForceCalc(false) before system initialization and try "
              "again."
           << std::endl;
        throw std::runtime_error(ss.str());
    }
}
void DEMTracker::assertMesh(const std::string& name) {
    if (obj->obj_type != OWNER_TYPE::MESH) {
        std::stringstream ss;
        ss << name << " is only callable for trackers tracking a mesh!" << std::endl;
        throw std::runtime_error(ss.str());
    }
}
void DEMTracker::assertGeoSize(size_t input_length, const std::string& func_name, const std::string& geo_type) {
    if (input_length != obj->nGeos) {
        std::stringstream ss;
        ss << func_name << " is called with an input not the same size (number of " << geo_type
           << ") as the original tracked object!\nThe input has " << input_length << " " << geo_type
           << " while the original object has " << obj->nGeos << "." << std::endl;
        throw std::runtime_error(ss.str());
    }
}
void DEMTracker::assertOwnerSize(size_t input_length, const std::string& name) {
    if (input_length != obj->nSpanOwners) {
        std::stringstream ss;
        ss << name << " is called with an input not the same size of the number of owners it tracks!\nThe input has "
           << input_length << " elements while the tracker tracks " << obj->nSpanOwners << " entities." << std::endl;
        throw std::runtime_error(ss.str());
    }
}

std::vector<bodyID_t> DEMTracker::GetContactClumps(size_t offset) {
    return sys->GetOwnerContactClumps(obj->ownerID + offset);
}

bodyID_t DEMTracker::GetOwnerID(size_t offset) {
    return obj->ownerID + offset;
}

float3 DEMTracker::Pos(size_t offset) {
    return sys->GetOwnerPosition(obj->ownerID + offset);
}
std::vector<float> DEMTracker::GetPos(size_t offset) {
    float3 res = Pos(offset);
    return {res.x, res.y, res.z};
}

float3 DEMTracker::AngVelLocal(size_t offset) {
    return sys->GetOwnerAngVel(obj->ownerID + offset);
}
std::vector<float> DEMTracker::GetAngVelLocal(size_t offset) {
    float3 res = AngVelLocal(offset);
    return {res.x, res.y, res.z};
}

float3 DEMTracker::AngVelGlobal(size_t offset) {
    float3 ang_v = sys->GetOwnerAngVel(obj->ownerID + offset);
    float4 oriQ = sys->GetOwnerOriQ(obj->ownerID + offset);
    hostApplyOriQToVector3(ang_v.x, ang_v.y, ang_v.z, oriQ.w, oriQ.x, oriQ.y, oriQ.z);
    return ang_v;
}
std::vector<float> DEMTracker::GetAngVelGlobal(size_t offset) {
    float3 res = AngVelGlobal(offset);
    return {res.x, res.y, res.z};
}

float3 DEMTracker::Vel(size_t offset) {
    return sys->GetOwnerVelocity(obj->ownerID + offset);
}
std::vector<float> DEMTracker::GetVel(size_t offset) {
    float3 res = Vel(offset);
    return {res.x, res.y, res.z};
}

float4 DEMTracker::OriQ(size_t offset) {
    return sys->GetOwnerOriQ(obj->ownerID + offset);
}
std::vector<float> DEMTracker::GetOriQ(size_t offset) {
    float4 res = OriQ(offset);
    return {res.x, res.y, res.z, res.w};
}

unsigned int DEMTracker::GetFamily(size_t offset) {
    return sys->GetOwnerFamily(obj->ownerID + offset);
}

float DEMTracker::Mass(size_t offset) {
    return sys->GetOwnerMass(obj->ownerID + offset);
}

float3 DEMTracker::MOI(size_t offset) {
    return sys->GetOwnerMOI(obj->ownerID + offset);
}
std::vector<float> DEMTracker::GetMOI(size_t offset) {
    float3 res = MOI(offset);
    return {res.x, res.y, res.z};
}

// float3 DEMTracker::Acc(size_t offset) {
//     float3 contact_acc = sys->GetOwnerAcc(obj->ownerID + offset);
//     // Contact acceleration is not total acc, we need to add gravity and manually added forces
//     //// TODO: How to do that?
// }
// float3 DEMTracker::AngAcc(size_t offset) {
//     float3 contact_angAcc = sys->GetOwnerAngAcc(obj->ownerID + offset);
//     // Contact angAcc is not total angAcc, we need to add manually added angular acc
//     //// TODO: How to do that?
// }

float3 DEMTracker::ContactAcc(size_t offset) {
    return sys->GetOwnerAcc(obj->ownerID + offset);
}
std::vector<float> DEMTracker::GetContactAcc(size_t offset) {
    float3 res = ContactAcc(offset);
    return {res.x, res.y, res.z};
}

float3 DEMTracker::ContactAngAccLocal(size_t offset) {
    return sys->GetOwnerAngAcc(obj->ownerID + offset);
}
std::vector<float> DEMTracker::GetContactAngAccLocal(size_t offset) {
    float3 res = ContactAngAccLocal(offset);
    return {res.x, res.y, res.z};
}

float3 DEMTracker::ContactAngAccGlobal(size_t offset) {
    float3 ang_acc = sys->GetOwnerAngAcc(obj->ownerID + offset);
    float4 oriQ = sys->GetOwnerOriQ(obj->ownerID + offset);
    hostApplyOriQToVector3(ang_acc.x, ang_acc.y, ang_acc.z, oriQ.w, oriQ.x, oriQ.y, oriQ.z);
    return ang_acc;
}
std::vector<float> DEMTracker::GetContactAngAccGlobal(size_t offset) {
    float3 res = ContactAngAccGlobal(offset);
    return {res.x, res.y, res.z};
}

float DEMTracker::GetOwnerWildcardValue(const std::string& name, size_t offset) {
    return sys->GetOwnerWildcardValue(obj->ownerID + offset, name);
}

float DEMTracker::GetGeometryWildcardValue(const std::string& name, size_t offset) {
    std::vector<float> res;
    switch (obj->obj_type) {
        case (OWNER_TYPE::CLUMP):
            res = sys->GetSphereWildcardValue(obj->geoID + offset, name, 1);
            break;
        case (OWNER_TYPE::ANALYTICAL):
            res = sys->GetAnalWildcardValue(obj->geoID + offset, name, 1);
            break;
        case (OWNER_TYPE::MESH):
            res = sys->GetTriWildcardValue(obj->geoID + offset, name, 1);
            break;
    }
    return res[0];
}

std::vector<float> DEMTracker::GetGeometryWildcardValues(const std::string& name) {
    std::vector<float> res;
    switch (obj->obj_type) {
        case (OWNER_TYPE::CLUMP):
            res = sys->GetSphereWildcardValue(obj->geoID, name, obj->nGeos);
            break;
        case (OWNER_TYPE::ANALYTICAL):
            res = sys->GetAnalWildcardValue(obj->geoID, name, obj->nGeos);
            break;
        case (OWNER_TYPE::MESH):
            res = sys->GetTriWildcardValue(obj->geoID, name, obj->nGeos);
            break;
    }
    return res;
}

size_t DEMTracker::GetContactForces(std::vector<float3>& points, std::vector<float3>& forces, size_t offset) {
    assertThereIsForcePairs("GetContactForces");
    points.clear();
    forces.clear();
    return sys->GetOwnerContactForces(obj->ownerID + offset, points, forces);
}
size_t DEMTracker::GetContactForces(std::vector<std::vector<float>>& points,
                                    std::vector<std::vector<float>>& forces,
                                    size_t offset) {
    std::vector<float3> float3_points, float3_forces;
    size_t nPairs = GetContactForces(float3_points, float3_forces, offset);

    points.resize(nPairs);
    forces.resize(nPairs);
    for (size_t i = 0; i < nPairs; i++) {
        std::vector<float> float_point = {float3_points[i].x, float3_points[i].y, float3_points[i].z};
        std::vector<float> float_force = {float3_forces[i].x, float3_forces[i].y, float3_forces[i].z};
        points[i] = float_point;
        forces[i] = float_force;
    }
    return nPairs;
}

size_t DEMTracker::GetContactForcesAndLocalTorque(std::vector<float3>& points,
                                                  std::vector<float3>& forces,
                                                  std::vector<float3>& torques,
                                                  size_t offset) {
    assertThereIsForcePairs("GetContactForces");
    points.clear();
    forces.clear();
    torques.clear();
    return sys->GetOwnerContactForces(obj->ownerID + offset, points, forces, torques, true);
}
size_t DEMTracker::GetContactForcesAndLocalTorque(std::vector<std::vector<float>>& points,
                                                  std::vector<std::vector<float>>& forces,
                                                  std::vector<std::vector<float>>& torques,
                                                  size_t offset) {
    std::vector<float3> float3_points, float3_forces, float3_torques;
    size_t nPairs = GetContactForcesAndLocalTorque(float3_points, float3_forces, float3_torques, offset);

    points.resize(nPairs);
    forces.resize(nPairs);
    torques.resize(nPairs);
    for (size_t i = 0; i < nPairs; i++) {
        std::vector<float> float_point = {float3_points[i].x, float3_points[i].y, float3_points[i].z};
        std::vector<float> float_force = {float3_forces[i].x, float3_forces[i].y, float3_forces[i].z};
        std::vector<float> float_torque = {float3_torques[i].x, float3_torques[i].y, float3_torques[i].z};
        points[i] = float_point;
        forces[i] = float_force;
        torques[i] = float_torque;
    }
    return nPairs;
}

size_t DEMTracker::GetContactForcesAndGlobalTorque(std::vector<float3>& points,
                                                   std::vector<float3>& forces,
                                                   std::vector<float3>& torques,
                                                   size_t offset) {
    assertThereIsForcePairs("GetContactForces");
    points.clear();
    forces.clear();
    torques.clear();
    return sys->GetOwnerContactForces(obj->ownerID + offset, points, forces, torques, false);
}
size_t DEMTracker::GetContactForcesAndGlobalTorque(std::vector<std::vector<float>>& points,
                                                   std::vector<std::vector<float>>& forces,
                                                   std::vector<std::vector<float>>& torques,
                                                   size_t offset) {
    std::vector<float3> float3_points, float3_forces, float3_torques;
    size_t nPairs = GetContactForcesAndGlobalTorque(float3_points, float3_forces, float3_torques, offset);

    points.resize(nPairs);
    forces.resize(nPairs);
    torques.resize(nPairs);
    for (size_t i = 0; i < nPairs; i++) {
        std::vector<float> float_point = {float3_points[i].x, float3_points[i].y, float3_points[i].z};
        std::vector<float> float_force = {float3_forces[i].x, float3_forces[i].y, float3_forces[i].z};
        std::vector<float> float_torque = {float3_torques[i].x, float3_torques[i].y, float3_torques[i].z};
        points[i] = float_point;
        forces[i] = float_force;
        torques[i] = float_torque;
    }
    return nPairs;
}

void DEMTracker::AddAcc(float3 acc, size_t offset) {
    sys->AddOwnerNextStepAcc(obj->ownerID + offset, acc);
}
void DEMTracker::AddAcc(const std::vector<float>& acc, size_t offset) {
    assertThreeElements(acc, "AddAcc", "acc");
    AddAcc(host_make_float3(acc[0], acc[1], acc[2]), offset);
}

void DEMTracker::AddAngAcc(float3 angAcc, size_t offset) {
    sys->AddOwnerNextStepAngAcc(obj->ownerID + offset, angAcc);
}
void DEMTracker::AddAngAcc(const std::vector<float>& angAcc, size_t offset) {
    assertThreeElements(angAcc, "AddAngAcc", "angAcc");
    AddAngAcc(host_make_float3(angAcc[0], angAcc[1], angAcc[2]), offset);
}

void DEMTracker::SetPos(float3 pos, size_t offset) {
    sys->SetOwnerPosition(obj->ownerID + offset, pos);
}
void DEMTracker::SetPos(const std::vector<float>& pos, size_t offset) {
    assertThreeElements(pos, "SetPos", "pos");
    SetPos(host_make_float3(pos[0], pos[1], pos[2]), offset);
}

void DEMTracker::SetAngVel(float3 angVel, size_t offset) {
    sys->SetOwnerAngVel(obj->ownerID + offset, angVel);
}
void DEMTracker::SetAngVel(const std::vector<float>& angVel, size_t offset) {
    assertThreeElements(angVel, "SetAngVel", "angVel");
    SetAngVel(host_make_float3(angVel[0], angVel[1], angVel[2]), offset);
}

void DEMTracker::SetVel(float3 vel, size_t offset) {
    sys->SetOwnerVelocity(obj->ownerID + offset, vel);
}
void DEMTracker::SetVel(const std::vector<float>& vel, size_t offset) {
    assertThreeElements(vel, "SetVel", "vel");
    SetVel(host_make_float3(vel[0], vel[1], vel[2]), offset);
}

void DEMTracker::SetOriQ(float4 oriQ, size_t offset) {
    sys->SetOwnerOriQ(obj->ownerID + offset, oriQ);
}
void DEMTracker::SetOriQ(const std::vector<float>& oriQ, size_t offset) {
    assertFourElements(oriQ, "SetOriQ", "oriQ");
    SetOriQ(host_make_float4(oriQ[0], oriQ[1], oriQ[2], oriQ[3]), offset);
}

void DEMTracker::SetFamily(const std::vector<unsigned int>& fam_nums) {
    assertOwnerSize(fam_nums.size(), "SetFamily");
    for (size_t i = 0; i < fam_nums.size(); i++) {
        sys->SetOwnerFamily(obj->ownerID + i, fam_nums[i]);
    }
}
void DEMTracker::SetFamily(unsigned int fam_num) {
    for (size_t i = 0; i < obj->nSpanOwners; i++) {
        sys->SetOwnerFamily(obj->ownerID + i, fam_num);
    }
}
void DEMTracker::SetFamily(unsigned int fam_num, size_t offset) {
    sys->SetOwnerFamily(obj->ownerID + offset, fam_num);
}
void DEMTracker::ChangeClumpSizes(const std::vector<bodyID_t>& IDs, const std::vector<float>& factors) {
    std::vector<bodyID_t> offsetted_IDs(IDs);
    size_t offset = obj->ownerID;
    std::for_each(offsetted_IDs.begin(), offsetted_IDs.end(), [offset](bodyID_t& x) { x += offset; });
    sys->ChangeClumpSizes(offsetted_IDs, factors);
}

void DEMTracker::UpdateMesh(const std::vector<float3>& new_nodes) {
    assertMesh("UpdateMesh");
    // assertGeoSize(new_mesh->GetNumTriangles(), "UpdateMesh", "triangles");
    // Outsource to API system to handle...
    sys->SetTriNodeRelPos(obj->ownerID, obj->geoID, new_nodes);
}
void DEMTracker::UpdateMesh(const std::vector<std::vector<float>>& new_nodes) {
    assertThreeElementsVector(new_nodes, "UpdateMesh", "new_nodes");
    std::vector<float3> float3_nodes(new_nodes.size());
    for (size_t i = 0; i < new_nodes.size(); i++) {
        float3_nodes[i] = host_make_float3(new_nodes[i][0], new_nodes[i][1], new_nodes[i][2]);
    }
    UpdateMesh(float3_nodes);
}

// Deformation is per-node, yet UpdateTriNodeRelPos need per-triangle info.
void DEMTracker::UpdateMeshByIncrement(const std::vector<float3>& deformation) {
    assertMesh("UpdateMeshByIncrement");
    // Outsource to API system to handle...
    sys->UpdateTriNodeRelPos(obj->ownerID, obj->geoID, deformation);
}
void DEMTracker::UpdateMeshByIncrement(const std::vector<std::vector<float>>& deformation) {
    assertThreeElementsVector(deformation, "UpdateMeshByIncrement", "deformation");
    std::vector<float3> float3_nodes(deformation.size());
    for (size_t i = 0; i < deformation.size(); i++) {
        float3_nodes[i] = host_make_float3(deformation[i][0], deformation[i][1], deformation[i][2]);
    }
    UpdateMeshByIncrement(float3_nodes);
}

std::shared_ptr<DEMMeshConnected>& DEMTracker::GetMesh() {
    assertMesh("GetMesh");
    return sys->GetCachedMesh(obj->ownerID);
}

std::vector<float3> DEMTracker::GetMeshNodesGlobal() {
    assertMesh("GetMeshNodesGlobal");
    return sys->GetMeshNodesGlobal(obj->ownerID);
}
std::vector<std::vector<float>> DEMTracker::GetMeshNodesGlobalAsVectorOfVector() {
    std::vector<float3> res = GetMeshNodesGlobal();
    std::vector<std::vector<float>> vector_nodes(res.size());
    for (size_t i = 0; i < res.size(); i++) {
        std::vector<float> float_elem = {res[i].x, res[i].y, res[i].z};
        vector_nodes[i] = float_elem;
    }
    return vector_nodes;
}

void DEMTracker::SetOwnerWildcardValue(const std::string& name, float wc, size_t offset) {
    sys->SetOwnerWildcardValue(obj->ownerID + offset, name, wc, 1);
}

void DEMTracker::SetOwnerWildcardValues(const std::string& name, const std::vector<float>& wc) {
    assertOwnerSize(wc.size(), "SetOwnerWildcardValues");
    sys->SetOwnerWildcardValue(obj->ownerID, name, wc);
}

void DEMTracker::SetGeometryWildcardValue(const std::string& name, float wc, size_t offset) {
    switch (obj->obj_type) {
        case (OWNER_TYPE::CLUMP):
            sys->SetSphereWildcardValue(obj->geoID + offset, name, std::vector<float>(1, wc));
            break;
        case (OWNER_TYPE::ANALYTICAL):
            sys->SetAnalWildcardValue(obj->geoID + offset, name, std::vector<float>(1, wc));
            break;
        case (OWNER_TYPE::MESH):
            sys->SetTriWildcardValue(obj->geoID + offset, name, std::vector<float>(1, wc));
            break;
    }
}

void DEMTracker::SetGeometryWildcardValues(const std::string& name, const std::vector<float>& wc) {
    switch (obj->obj_type) {
        case (OWNER_TYPE::CLUMP):
            assertGeoSize(wc.size(), "SetGeometryWildcardValues", "spheres");
            sys->SetSphereWildcardValue(obj->geoID, name, wc);
            break;
        case (OWNER_TYPE::ANALYTICAL):
            assertGeoSize(wc.size(), "SetGeometryWildcardValues", "analytical components");
            sys->SetAnalWildcardValue(obj->geoID, name, wc);
            break;
        case (OWNER_TYPE::MESH):
            assertGeoSize(wc.size(), "SetGeometryWildcardValues", "triangles");
            sys->SetTriWildcardValue(obj->geoID, name, wc);
            break;
    }
}

// =============================================================================
// DEMForceModel class
// =============================================================================

void DEMForceModel::SetForceModelType(FORCE_MODEL model_type) {
    type = model_type;
    switch (model_type) {
        case (FORCE_MODEL::HERTZIAN):
            m_must_have_mat_props = {"E", "nu", "CoR", "mu", "Crr"};
            m_pairwise_mat_props = {"CoR", "mu", "Crr"};
            m_force_model = HERTZIAN_FORCE_MODEL();
            // History-based model uses these history-related arrays
            m_contact_wildcards = {"delta_time", "delta_tan_x", "delta_tan_y", "delta_tan_z"};
            break;
        case (FORCE_MODEL::HERTZIAN_FRICTIONLESS):
            m_must_have_mat_props = {"E", "nu", "CoR"};
            m_pairwise_mat_props = {"CoR"};
            m_force_model = HERTZIAN_FORCE_MODEL_FRICTIONLESS();
            // No contact history needed for frictionless
            m_contact_wildcards.clear();
            break;
        case (FORCE_MODEL::CUSTOM):
            m_must_have_mat_props.clear();
            m_pairwise_mat_props.clear();
    }
}

void DEMForceModel::DefineCustomModel(const std::string& model) {
    // If custom model is set, we don't care what materials needs to be set
    m_must_have_mat_props.clear();
    type = FORCE_MODEL::CUSTOM;
    m_force_model = model;
}

int DEMForceModel::ReadCustomModelFile(const std::filesystem::path& sourcefile) {
    if (!std::filesystem::exists(sourcefile)) {
        return 1;
    }
    // If custom model is set, we don't care what materials needs to be set
    m_must_have_mat_props.clear();
    type = FORCE_MODEL::CUSTOM;
    m_force_model = read_file_to_string(sourcefile);
    return 0;
}

void DEMForceModel::SetPerContactWildcards(const std::set<std::string>& wildcards) {
    for (const auto& a_str : wildcards) {
        if (match_pattern(a_str, " ")) {
            // Wildcard array names cannot have spaces in them
            std::stringstream ss;
            ss << "Contact wildcard " << a_str << " is not valid: no spaces allowed in its name." << std::endl;
            throw std::runtime_error(ss.str());
        }
    }
    m_contact_wildcards = wildcards;
}

void DEMForceModel::SetPerOwnerWildcards(const std::set<std::string>& wildcards) {
    for (const auto& a_str : wildcards) {
        if (match_pattern(a_str, " ")) {
            std::stringstream ss;
            ss << "Owner wildcard " << a_str << " is not valid: no spaces allowed in its name." << std::endl;
            throw std::runtime_error(ss.str());
        }
    }
    m_owner_wildcards = wildcards;
}

void DEMForceModel::SetPerGeometryWildcards(const std::set<std::string>& wildcards) {
    for (const auto& a_str : wildcards) {
        if (match_pattern(a_str, " ")) {
            std::stringstream ss;
            ss << "Geometry wildcard " << a_str << " is not valid: no spaces allowed in its name." << std::endl;
            throw std::runtime_error(ss.str());
        }
    }
    m_geo_wildcards = wildcards;
}

}  // END namespace deme
