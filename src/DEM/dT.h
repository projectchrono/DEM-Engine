//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_DT
#define DEME_DT

#include <array>
#include <mutex>
#include <vector>
#include <thread>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <functional>
#include <algorithm>
#include <cmath>

#include "../core/utils/CudaAllocator.hpp"
#include "../core/utils/ThreadManager.h"
#include "../core/utils/GpuManager.h"
#include "../core/utils/JitHelper.h"
#include "../core/utils/DataMigrationHelper.hpp"
#include "../kernel/DEMHelperKernels.cuh"
#include "BdrsAndObjs.h"
#include "Defines.h"
#include "Structs.h"
#include "AuxClasses.h"

namespace deme {

// Implementation-level classes
class DEMKinematicThread;
class DEMDynamicThread;
class DEMSolverScratchData;

// Internal estimator for tracking the drift/cost trade-off:
// J(d) ≈ a + b/d + c*(d/dmax) + e*(d/dmax)^2, with forgetting (non-stationary baseline).
struct DriftRLS {
    static constexpr int N = 4;  // a, b, c, e
    double theta[N] = {0.0, 0.0, 0.0, 0.0};
    double P[N][N] = {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}};
    double lambda = 0.999;     // forgetting factor (closer to 1 => less covariance blow-up)
    double sigma2_ema = 1e-6;  // residual variance estimate (EWMA)
    bool initialized = false;

    void reset(double p0 = 1e2) {
        for (int i = 0; i < N; i++) {
            theta[i] = 0.0;
            for (int j = 0; j < N; j++) {
                P[i][j] = (i == j) ? p0 : 0.0;
            }
        }
        sigma2_ema = 1e-6;
        initialized = true;
    }

    static double dot(const double* a, const double* b) {
        double s = 0.0;
        for (int i = 0; i < N; i++)
            s += a[i] * b[i];
        return s;
    }

    double huberWeight(double r) const {
        const double sigma = std::sqrt(std::max(1e-12, sigma2_ema));
        const double k = 2.5;
        const double t = k * sigma;
        const double ar = std::abs(r);
        if (ar <= t)
            return 1.0;
        return t / ar;
    }

    void update(unsigned int d, double d0, double y) {
        if (!initialized)
            reset();
        if (!std::isfinite(y))
            return;

        const double dd = static_cast<double>(std::max(1u, d));
        // Use a saturating normalization with an *external* scale d0 (typical drift),
        // so the hard safety cap (upperBoundFutureDrift) does not distort the model.
        d0 = std::max(1.0, d0);
        const double x = dd / (dd + d0);
        const double phi[N] = {1.0, d0 / dd, x, x * x};

        const double y_hat = dot(theta, phi);
        if (!std::isfinite(y_hat)) {
            reset();
            return;
        }
        const double r = y - y_hat;

        // Update residual variance estimate (EWMA)
        const double beta = 0.05;
        sigma2_ema = (1.0 - beta) * sigma2_ema + beta * (r * r);
        if (!std::isfinite(sigma2_ema) || sigma2_ema < 0.0)
            sigma2_ema = 1e-6;

        // Robust weight
        const double w = huberWeight(r);
        const double s = std::sqrt(w);

        // Weighted observation
        double phiw[N];
        for (int i = 0; i < N; i++)
            phiw[i] = s * phi[i];
        const double yw = s * y;

        // Compute P * phiw
        double Pphi[N] = {0.0, 0.0, 0.0, 0.0};
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++)
                Pphi[i] += P[i][j] * phiw[j];
        }

        const double denom = lambda + dot(phiw, Pphi);
        if (!std::isfinite(denom) || denom <= 1e-18) {
            reset();
            return;
        }

        // Gain K = Pphi / denom
        double K[N];
        for (int i = 0; i < N; i++)
            K[i] = Pphi[i] / denom;

        // Innovation
        const double errw = yw - dot(theta, phiw);
        if (!std::isfinite(errw)) {
            reset();
            return;
        }
        for (int i = 0; i < N; i++)
            theta[i] += K[i] * errw;

        // Light physical priors: prevent pathological fits under noise.
        if (theta[1] < 0.0)
            theta[1] = 0.0;  // b >= 0
        if (theta[2] < 0.0)
            theta[2] = 0.0;  // c >= 0
        if (theta[3] < 0.0)
            theta[3] = 0.0;  // e >= 0
        for (int i = 0; i < N; i++) {
            if (!std::isfinite(theta[i])) {
                reset();
                return;
            }
        }

        // P = (P - K * phiw^T * P) / lambda
        double newP[N][N];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                double ssum = P[i][j];
                for (int k = 0; k < N; k++) {
                    ssum -= (K[i] * phiw[k]) * P[k][j];
                }
                newP[i][j] = ssum / lambda;
            }
        }
        constexpr double P_ABS_MAX = 1e12;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (!std::isfinite(newP[i][j]) || std::abs(newP[i][j]) > P_ABS_MAX) {
                    reset();
                    return;
                }
                P[i][j] = newP[i][j];
            }
        }
    }

    double predict(unsigned int d, double d0) const {
        const double dd = static_cast<double>(std::max(1u, d));
        d0 = std::max(1.0, d0);
        const double x = dd / (dd + d0);
        const double phi[N] = {1.0, d0 / dd, x, x * x};
        return theta[0] * phi[0] + theta[1] * phi[1] + theta[2] * phi[2] + theta[3] * phi[3];
    }
};

/// DynamicThread class
class DEMDynamicThread {
  protected:
    WorkerReportChannel* pPagerToMain;
    ThreadManager* pSchedSupport;
    // GpuManager* pGpuDistributor;

    // dT verbosity
    verbosity_t verbosity = VERBOSITY_INFO;

    // Some behavior-related flags
    SolverFlags solverFlags;

    // The std::thread that binds to this instance
    std::thread th;

    // Friend system DEMKinematicThread
    DEMKinematicThread* kT;

    // Number of items in the buffer array for primitive contact info
    size_t primitiveBufferSize = 0;
    // Number of items in the buffer array for patch-enabled contact info
    size_t patchBufferSize = 0;

    // dT's one-element buffer of kT-supplied nPrimitiveContactPairs (as buffer, it's device-only, but I used DualStruct
    // just for convenience...)
    DualStruct<size_t> nPrimitiveContactPairs_buffer = DualStruct<size_t>(0);
    // Similarly, the patch contact-related buffer
    DualStruct<size_t> nPatchContactPairs_buffer = DualStruct<size_t>(0);

    // Array-used memory size in bytes
    size_t m_approxDeviceBytesUsed = 0;
    size_t m_approxHostBytesUsed = 0;

    // Object which stores the device and stream IDs for this thread
    GpuManager::StreamInfo streamInfo;

    // Reusable event for stream barriers
    cudaEvent_t streamSyncEvent = nullptr;
    // Signals that dT finished writing the dT->kT transfer buffers (same-device fast path).
    cudaEvent_t dT_to_kT_BufferReadyEvent = nullptr;
    // Signals that the kT->dT numContacts copy has completed (kT stream for same-device fast path).
    cudaEvent_t kT_numContactsReadyEvent = nullptr;
    bool kT_numContacts_copy_pending = false;
    bool contactMappingUsesBuffer = false;
    uint64_t last_kT_produce_stamp = 0;  // last seen kT->dT update count (same-device fast path)
    int64_t recv_stamp_override = -1;
    static constexpr int kProgressEventDepth = 8;
    static constexpr int kMaxInFlightProgress = 1;
    std::array<cudaEvent_t, kProgressEventDepth> progressEvents = {};
    std::array<int64_t, kProgressEventDepth> progressEventStamps = {};
    int progressEventHead = 0;
    int progressEventCount = 0;

    // A class that contains scratch pad and system status data (constructed with the number of temp arrays we need)
    DEMSolverScratchData solverScratchSpace = DEMSolverScratchData(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // The number of for iterations dT does for a specific user "run simulation" call
    double cycleDuration;

    // dT believes this amount of future drift is ideal
    DualStruct<unsigned int> perhapsIdealFutureDrift = DualStruct<unsigned int>(0);
    // Total number of force-relevant periodic skip candidates since last kT unpack.
    DualStruct<unsigned int> ownerCylSkipPotentialTotal = DualStruct<unsigned int>(0);

    // Buffer arrays for storing info from the dT side.
    // kT modifies these arrays; dT uses them only.

    // dT gets contact pair/location/history map info from kT (ping-pong buffers)
    DeviceArray<bodyID_t> idPrimitiveA_buffer[2] = {DeviceArray<bodyID_t>(&m_approxDeviceBytesUsed),
                                                    DeviceArray<bodyID_t>(&m_approxDeviceBytesUsed)};
    DeviceArray<bodyID_t> idPrimitiveB_buffer[2] = {DeviceArray<bodyID_t>(&m_approxDeviceBytesUsed),
                                                    DeviceArray<bodyID_t>(&m_approxDeviceBytesUsed)};
    DeviceArray<contact_t> contactTypePrimitive_buffer[2] = {DeviceArray<contact_t>(&m_approxDeviceBytesUsed),
                                                             DeviceArray<contact_t>(&m_approxDeviceBytesUsed)};

    // NEW: Buffer arrays for separate patch IDs and their mapping to geometry arrays (ping-pong)
    DeviceArray<bodyID_t> idPatchA_buffer[2] = {DeviceArray<bodyID_t>(&m_approxDeviceBytesUsed),
                                                DeviceArray<bodyID_t>(&m_approxDeviceBytesUsed)};
    DeviceArray<bodyID_t> idPatchB_buffer[2] = {DeviceArray<bodyID_t>(&m_approxDeviceBytesUsed),
                                                DeviceArray<bodyID_t>(&m_approxDeviceBytesUsed)};
    DeviceArray<contact_t> contactTypePatch_buffer[2] = {DeviceArray<contact_t>(&m_approxDeviceBytesUsed),
                                                         DeviceArray<contact_t>(&m_approxDeviceBytesUsed)};
    DeviceArray<bodyID_t> contactPatchIsland_buffer[2] = {DeviceArray<bodyID_t>(&m_approxDeviceBytesUsed),
                                                          DeviceArray<bodyID_t>(&m_approxDeviceBytesUsed)};
    DeviceArray<contactPairs_t> geomToPatchMap_buffer[2] = {DeviceArray<contactPairs_t>(&m_approxDeviceBytesUsed),
                                                            DeviceArray<contactPairs_t>(&m_approxDeviceBytesUsed)};
    DeviceArray<contactPairs_t> contactMapping_buffer[2] = {DeviceArray<contactPairs_t>(&m_approxDeviceBytesUsed),
                                                            DeviceArray<contactPairs_t>(&m_approxDeviceBytesUsed)};
    int kt_write_buf = 0;  // which buffer kT writes to next

    // Permanent array for patch contact penetrations (used to compute max tri-tri penetration)
    DeviceArray<double> finalPenetrations = DeviceArray<double>(&m_approxDeviceBytesUsed);

    // Max tri-tri penetration value to be sent to kT
    DualStruct<double> maxTriTriPenetration = DualStruct<double>(0.0);

    // Simulation params-related variables
    DualStruct<DEMSimParams> simParams = DualStruct<DEMSimParams>();

    // Pointers to those data arrays defined below, stored in a struct
    DualStruct<DEMDataDT> granData = DualStruct<DEMDataDT>();


    // For cylindrical periodicity: indices of per-contact wildcard triplets (x,y,z) that represent global vectors
    // and must be rotated when an owner wraps.
    DualArray<int3> cylPeriodicWCTriplets = DualArray<int3>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // Body-related arrays, for dT's personal use (not transfer buffer)

    // Those are the smaller ones, the unique, template ones
    // The mass values
    DualArray<float> massOwnerBody = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // The components of MOI values
    DualArray<float> mmiXX = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<float> mmiYY = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<float> mmiZZ = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // Volume values
    DualArray<float> volumeOwnerBody = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // The distinct sphere radii values
    DualArray<float> radiiSphere = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // The distinct sphere local position (wrt CoM) values
    DualArray<float> relPosSphereX = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<float> relPosSphereY = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<float> relPosSphereZ = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // Triangles (templates) are given a special place (unlike other analytical shapes), b/c we expect them to appear
    // frequently as meshes.
    DualArray<float3> relPosNode1 = DualArray<float3>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<float3> relPosNode2 = DualArray<float3>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<float3> relPosNode3 = DualArray<float3>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // Relative position (to mesh CoM) of each mesh patch
    DualArray<float3> relPosPatch = DualArray<float3>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // External object's components may need the following arrays to store some extra defining features of them. We
    // assume there are usually not too many of them in a simulation.
    // Relative position w.r.t. the owner. For example, the following 3 arrays may hold center points for plates, or tip
    // positions for cones.
    DualArray<float> relPosEntityX = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<float> relPosEntityY = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<float> relPosEntityZ = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    // Some orientation specifiers. For example, the following 3 arrays may hold normal vectors for planes, or center
    // axis vectors for cylinders.
    DualArray<float> oriEntityX = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<float> oriEntityY = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<float> oriEntityZ = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    // Some size specifiers. For example, the following 3 arrays may hold top, bottom and length information for finite
    // cylinders.
    DualArray<float> sizeEntity1 = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<float> sizeEntity2 = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<float> sizeEntity3 = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // What type is this owner? Clump? Analytical object? Meshed object?
    DualArray<ownerType_t> ownerTypes = DualArray<ownerType_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // Per-owner bounding radius (circumscribed sphere radius in the owner's local frame).
    // Used by cylindrical periodicity wrapping logic to keep per-owner ghost bands consistent with kT ghosting.
    DualArray<float> ownerBoundRadius = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);


    // Per-owner cylindrical periodic wrap count (filled every integration step; consumed to rotate contact history).
    DualArray<int> ownerCylWrapK = DualArray<int>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    // Cumulative cylindrical wrap offset since last kT update (used to interpret ghost IDs under async drift).
    DualArray<int> ownerCylWrapOffset = DualArray<int>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    // Per-owner cylindrical periodic flags shared by kT and dT.
    // Bit CYL_GHOST_HINT_START (0x1): start-side ghost hint (+span).
    // Bit CYL_GHOST_HINT_END (0x2): end-side ghost hint (-span).
    // Bit CYL_GHOST_HINT_MISMATCH (0x4): dT observed kT/dT branch mismatch.
    DualArray<unsigned int> ownerCylGhostActive =
        DualArray<unsigned int>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    // Per-owner count of periodic candidates skipped because kT/dT image branches differ in this dT step.
    DualArray<unsigned int> ownerCylSkipCount =
        DualArray<unsigned int>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    // Subset of ownerCylSkipCount where the skipped candidate would otherwise be a contact.
    DualArray<unsigned int> ownerCylSkipPotentialCount =
        DualArray<unsigned int>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // Those are the large ones, ones that have the same length as the number of clumps
    // The mass/MOI offsets
    DualArray<inertiaOffset_t> inertiaPropOffsets =
        DualArray<inertiaOffset_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // Clump's family identification code. Used in determining whether they can be contacts between two families, and
    // whether a family has prescribed motions.
    DualArray<family_t> familyID = DualArray<family_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // The (impl-level) family IDs whose entities should not be outputted to files
    std::unordered_set<family_t> familiesNoOutput;

    // The voxel ID (split into 3 parts, representing XYZ location)
    DualArray<voxelID_t> voxelID = DualArray<voxelID_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // The XYZ local location inside a voxel
    DualArray<subVoxelPos_t> locX = DualArray<subVoxelPos_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<subVoxelPos_t> locY = DualArray<subVoxelPos_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<subVoxelPos_t> locZ = DualArray<subVoxelPos_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // The clump quaternion
    DualArray<oriQ_t> oriQw = DualArray<oriQ_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<oriQ_t> oriQx = DualArray<oriQ_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<oriQ_t> oriQy = DualArray<oriQ_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<oriQ_t> oriQz = DualArray<oriQ_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // Linear velocity
    DualArray<float> vX = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<float> vY = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<float> vZ = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // Local angular velocity
    DualArray<float> omgBarX = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<float> omgBarY = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<float> omgBarZ = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // Linear acceleration
    DualArray<float> aX = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<float> aY = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<float> aZ = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // Local angular acceleration
    DualArray<float> alphaX = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<float> alphaY = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<float> alphaZ = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // If true, the acceleration is specified for this owner and the prep force kernel should not clear its value in the
    // next time step.
    DualArray<notStupidBool_t> accSpecified =
        DualArray<notStupidBool_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<notStupidBool_t> angAccSpecified =
        DualArray<notStupidBool_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // Contact pair/location, for dT's personal use!!
    DualArray<bodyID_t> idPrimitiveA = DualArray<bodyID_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<bodyID_t> idPrimitiveB = DualArray<bodyID_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<contact_t> contactTypePrimitive = DualArray<contact_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    // DualArray<contactPairs_t> contactMapping;

    // NEW: Separate patch IDs and mapping arrays (work arrays for dT)
    DualArray<bodyID_t> idPatchA = DualArray<bodyID_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<bodyID_t> idPatchB = DualArray<bodyID_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<contact_t> contactTypePatch = DualArray<contact_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<bodyID_t> contactPatchIsland = DualArray<bodyID_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<contactPairs_t> geomToPatchMap =
        DualArray<contactPairs_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // Some of dT's own work arrays
    // Force of each contact event. It is the force that bodyA feels. They are in global.
    DualArray<float3> contactForces = DualArray<float3>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    // An imaginary `force' in each contact event that produces torque only, and does not affect the linear motion. It
    // will rise in our default rolling resistance model, which is just a torque model; yet, our contact registration is
    // contact pair-based, meaning we do not know the specs of each contact body, so we can register force only, not
    // torque. Therefore, this vector arises. This force-like torque is in global.
    DualArray<float3> contactTorque_convToForce = DualArray<float3>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    // Local position of contact point of contact w.r.t. the reference frame of body A and B
    DualArray<float3> contactPointGeometryA = DualArray<float3>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<float3> contactPointGeometryB = DualArray<float3>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    // Wildcard (extra property) arrays associated with contacts and owners
    std::vector<std::unique_ptr<DualArray<float>>> contactWildcards;
    std::vector<std::unique_ptr<DualArray<float>>> ownerWildcards;
    // DualArray<float> contactWildcards[DEME_MAX_WILDCARD_NUM];
    // DualArray<float> ownerWildcards[DEME_MAX_WILDCARD_NUM];
    // An example of such wildcard arrays is contact history: how much did the contact point move on the geometry
    // surface compared to when the contact first emerged?
    // Geometric entities' wildcards
    std::vector<std::unique_ptr<DualArray<float>>> sphereWildcards;
    std::vector<std::unique_ptr<DualArray<float>>> analWildcards;
    std::vector<std::unique_ptr<DualArray<float>>> patchWildcards;

    // Storage for the names of the contact wildcards (whose order agrees with the impl-level wildcard numbering, from 1
    // to n)
    std::set<std::string> m_contact_wildcard_names;
    std::set<std::string> m_owner_wildcard_names;
    std::set<std::string> m_geo_wildcard_names;

    // DualArray<float3> contactHistory;
    // // Durations in time of persistent contact pairs
    // DualArray<float> contactDuration;
    // The velocity of the contact points in the global frame: can be useful in determining the time step size
    // DualArray<float3> contactPointVel;

    // dT's total steps run (since last time the collaboration stats cache is cleared)
    uint64_t nTotalSteps = 0;

    // If true, dT needs to re-process idA- and idB-related data arrays before collecting forces, as those arrays are
    // freshly obtained from kT.
    bool contactPairArr_isFresh = true;

    // If true, something critical (such as new clumps loaded, ts size changed...) just happened, and dT will need a kT
    // update to proceed.
    bool pendingCriticalUpdate = true;

    // Number of threads per block for dT force calculation kernels
    unsigned int DT_FORCE_CALC_NTHREADS_PER_BLOCK = 128;

    // Template-related arrays
    // Belonged-body ID
    DualArray<bodyID_t> ownerClumpBody = DualArray<bodyID_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<bodyID_t> ownerTriMesh = DualArray<bodyID_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<bodyID_t> ownerAnalBody = DualArray<bodyID_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    // Mesh owner flags (indexed by owner body ID)
    DualArray<notStupidBool_t> ownerMeshConvex =
        DualArray<notStupidBool_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<notStupidBool_t> ownerMeshNeverWinner =
        DualArray<notStupidBool_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    // Mesh patch information: each facet belongs to a patch, and each patch has material properties
    // Patch ID for each triangle facet (maps facet to patch)
    DualArray<bodyID_t> triPatchID = DualArray<bodyID_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    // Triangle edge neighbors (compact; index via triNeighborIndex)
    DualArray<bodyID_t> triNeighborIndex = DualArray<bodyID_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<bodyID_t> triNeighbor1 = DualArray<bodyID_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<bodyID_t> triNeighbor2 = DualArray<bodyID_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<bodyID_t> triNeighbor3 = DualArray<bodyID_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    // Mesh patch owner IDs (one per patch, flattened across all meshes)
    DualArray<bodyID_t> ownerPatchMesh = DualArray<bodyID_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // The ID that maps this sphere component's geometry-defining parameters, when this component is jitified
    DualArray<clumpComponentOffset_t> clumpComponentOffset =
        DualArray<clumpComponentOffset_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    // The ID that maps this sphere component's geometry-defining parameters, when this component is not jitified (too
    // many templates)
    DualArray<clumpComponentOffsetExt_t> clumpComponentOffsetExt =
        DualArray<clumpComponentOffsetExt_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    // The ID that maps this analytical entity component's geometry-defining parameters, when this component is jitified
    // DualArray<clumpComponentOffset_t> analComponentOffset;

    // The ID that maps this entity's material
    DualArray<materialsOffset_t> sphereMaterialOffset =
        DualArray<materialsOffset_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    // Material offset for each mesh patch (indexed by patch, can be looked up via triPatchID)
    DualArray<materialsOffset_t> patchMaterialOffset =
        DualArray<materialsOffset_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // dT's copy of family map
    // std::unordered_map<unsigned int, family_t> familyUserImplMap;
    // std::unordered_map<family_t, unsigned int> familyImplUserMap;

    // A long array (usually 32640 elements) registering whether between 2 families there should be contacts
    DualArray<notStupidBool_t> familyMaskMatrix =
        DualArray<notStupidBool_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // The amount of contact margin that each family should add to its associated contact geometries. Default is 0, and
    // that means geometries should be considered in contact when they are physically in contact.
    DualArray<float> familyExtraMarginSize = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // dT's copy of "clump template and their names" map
    std::unordered_map<unsigned int, std::string> templateNumNameMap;

    // dT's storage of how many contact pairs of each contact type are currently present
    DualArray<contact_t> existingContactTypes;
    DualArray<contactPairs_t> typeStartOffsetsPrimitive;
    DualArray<contactPairs_t> typeStartOffsetsPatch;
    size_t m_numExistingTypes = 0;
    // A map that records the contact <ID start, and count> for each contact type currently existing
    ContactTypeMap<std::pair<contactPairs_t, contactPairs_t>> typeStartCountPrimitiveMap;
    ContactTypeMap<std::pair<contactPairs_t, contactPairs_t>> typeStartCountPatchMap;
    // A map that records the corresponding jitify program bundle and kernel name for each contact type
    ContactTypeMap<std::vector<std::pair<std::shared_ptr<JitHelper::CachedProgram>, std::string>>>
        contactTypePrimitiveKernelMap;
    ContactTypeMap<std::vector<std::pair<std::shared_ptr<JitHelper::CachedProgram>, std::string>>>
        contactTypePatchKernelMap;

    // dT's timers
    std::vector<std::string> timer_names = {"Clear force array", "Calculate contact forces", "Optional force reduction",
                                            "Integration",       "Unpack updates from kT",   "Send to kT buffer"};
    SolverTimers timers = SolverTimers(timer_names);
    std::chrono::steady_clock::time_point cycle_stopwatch_start;
    bool cycle_stopwatch_started = false;
    void startCycleStopwatch();
    double getCycleElapsedSeconds() const;

  public:
    friend class DEMSolver;
    friend class DEMKinematicThread;

    DEMDynamicThread(WorkerReportChannel* pPager, ThreadManager* pSchedSup, const GpuManager::StreamInfo& sInfo)
        : pPagerToMain(pPager), pSchedSupport(pSchedSup), streamInfo(sInfo) {
        cycleDuration = 0;

        pPagerToMain->userCallDone = false;
        pSchedSupport->dynamicShouldJoin = false;
        pSchedSupport->dynamicStarted = false;

        // I found creating the stream here is needed (rather than creating it in the child thread).
        // This is because in smaller problems, the array data transfer portion (which needs the stream) could even be
        // reached before the stream is created in the child thread. So we have to create the stream here before
        // spawning the child thread.

        DEME_GPU_CALL(cudaStreamCreate(&streamInfo.stream));

        DEME_GPU_CALL(cudaEventCreateWithFlags(&streamSyncEvent, cudaEventDisableTiming));
        DEME_GPU_CALL(cudaEventCreateWithFlags(&dT_to_kT_BufferReadyEvent, cudaEventDisableTiming));
        DEME_GPU_CALL(cudaEventCreateWithFlags(&kT_numContactsReadyEvent, cudaEventDisableTiming));
        for (auto& evt : progressEvents) {
            evt = nullptr;
            DEME_GPU_CALL(cudaEventCreateWithFlags(&evt, cudaEventDisableTiming));
        }
        timers.InitGpuEvents();

        // Launch a worker thread bound to this instance
        th = std::move(std::thread([this]() { this->workerThread(); }));
    }
    ~DEMDynamicThread() {
        // std::cout << "Dynamic thread closing..." << std::endl;
        pSchedSupport->dynamicShouldJoin = true;
        startThread();
        th.join();
        timers.DestroyGpuEvents();
        cudaStreamDestroy(streamInfo.stream);
        if (streamSyncEvent) {
            cudaEventDestroy(streamSyncEvent);
            streamSyncEvent = nullptr;
        }
        if (dT_to_kT_BufferReadyEvent) {
            cudaEventDestroy(dT_to_kT_BufferReadyEvent);
            dT_to_kT_BufferReadyEvent = nullptr;
        }
        if (kT_numContactsReadyEvent) {
            cudaEventDestroy(kT_numContactsReadyEvent);
            kT_numContactsReadyEvent = nullptr;
        }
        for (auto& evt : progressEvents) {
            if (evt) {
                cudaEventDestroy(evt);
                evt = nullptr;
            }
        }

        deallocateEverything();
    }

    void setCycleDuration(double val) { cycleDuration = val; }

    // buffer exchange methods
    void setDestinationBufferPointers();

    /// Set SimParams items
    void setSimParams(unsigned char nvXp2,
                      unsigned char nvYp2,
                      unsigned char nvZp2,
                      float l,
                      double voxelSize,
                      double binSize,
                      binID_t nbX,
                      binID_t nbY,
                      binID_t nbZ,
                      float3 LBFPoint,
                      float3 user_box_min,
                      float3 user_box_max,
                      float3 G,
                      double ts_size,
                      float expand_factor,
                      float approx_max_vel,
                      double max_tritri_penetration,
                      float expand_safety_param,
                      float expand_safety_adder,
                      bool use_angvel_margin,
                      const std::set<std::string>& contact_wildcards,
                      const std::set<std::string>& owner_wildcards,
                      const std::set<std::string>& geo_wildcards);

    /// @brief Get total number of contacts.
    /// @return Number of contacts.
    size_t getNumContacts() const;
    /// Get this owner's position in user unit, for n consecutive items.
    std::vector<float3> getOwnerPos(bodyID_t ownerID, bodyID_t n = 1);
    /// Get this owner's angular velocity, for n consecutive items.
    std::vector<float3> getOwnerAngVel(bodyID_t ownerID, bodyID_t n = 1);
    /// Get this owner's quaternion, for n consecutive items.
    std::vector<float4> getOwnerOriQ(bodyID_t ownerID, bodyID_t n = 1);
    /// Get this owner's velocity, for n consecutive items.
    std::vector<float3> getOwnerVel(bodyID_t ownerID, bodyID_t n = 1);
    /// Get this owner's acceleration, for n consecutive items.
    std::vector<float3> getOwnerAcc(bodyID_t ownerID, bodyID_t n = 1);
    /// Get this owner's angular acceleration, for n consecutive items.
    std::vector<float3> getOwnerAngAcc(bodyID_t ownerID, bodyID_t n = 1);
    /// Get this owner's family number, for n consecutive items.
    std::vector<unsigned int> getOwnerFamily(bodyID_t ownerID, bodyID_t n = 1);
    /// Get this owner's cylindrical wrap count for n consecutive items.
    std::vector<int> getOwnerCylWrapK(bodyID_t ownerID, bodyID_t n = 1);
    /// Get this owner's cylindrical wrap offset since the last kT update for n consecutive items.
    std::vector<int> getOwnerCylWrapOffset(bodyID_t ownerID, bodyID_t n = 1);
    /// Get this owner's bound radius for n consecutive items.
    std::vector<float> getOwnerBoundRadius(bodyID_t ownerID, bodyID_t n = 1);
    /// Get this owner's cylindrical ghost-active flag for n consecutive items.
    std::vector<unsigned int> getOwnerCylGhostActive(bodyID_t ownerID, bodyID_t n = 1);
    /// Get this owner's count of periodic candidates skipped due to image-branch mismatch.
    std::vector<unsigned int> getOwnerCylSkipCount(bodyID_t ownerID, bodyID_t n = 1);
    /// Get this owner's count of force-relevant periodic skips (candidate would otherwise be a contact).
    std::vector<unsigned int> getOwnerCylSkipPotentialCount(bodyID_t ownerID, bodyID_t n = 1);
    /// Get per-owner contact counts split by real/ghost(+)/ghost(-).
    void getOwnerContactGhostCounts(std::vector<int>& real_cnt,
                                    std::vector<int>& ghost_pos_cnt,
                                    std::vector<int>& ghost_neg_cnt);
    // Get the current auto-adjusted update freq.
    float getUpdateFreq() const;

    /// Set consecutive owners' position in user unit.
    void setOwnerPos(bodyID_t ownerID, const std::vector<float3>& pos);
    /// Set consecutive owners's angular velocity.
    void setOwnerAngVel(bodyID_t ownerID, const std::vector<float3>& angVel);
    /// Set consecutive owners' quaternion.
    void setOwnerOriQ(bodyID_t ownerID, const std::vector<float4>& oriQ);
    /// Set consecutive owners' velocity.
    void setOwnerVel(bodyID_t ownerID, const std::vector<float3>& vel);
    /// Set consecutive owners' family number, for n consecutive items.
    void setOwnerFamily(bodyID_t ownerID, family_t fam, bodyID_t n = 1);

    /// @brief Add an extra acceleration to consecutive owners for the next time step.
    void addOwnerNextStepAcc(bodyID_t ownerID, const std::vector<float3>& acc);
    /// @brief Add an extra angular acceleration to consecutive owners for the next time step.
    void addOwnerNextStepAngAcc(bodyID_t ownerID, const std::vector<float3>& angAcc);

    /// Rewrite the relative positions of the flattened triangle soup, starting from `start', using triangle nodal
    /// positions in `triangles'.
    void setTriNodeRelPos(size_t start, const std::vector<DEMTriangle>& triangles);
    /// Rewrite the relative positions of the flattened triangle soup, starting from `start' by the amount stipulated in
    /// updates.
    void updateTriNodeRelPos(size_t start, const std::vector<DEMTriangle>& updates);

    /// @brief Globally modify a owner wildcard's value.
    void setOwnerWildcardValue(bodyID_t ownerID, unsigned int wc_num, const std::vector<float>& vals);
    /// @brief Modify the owner wildcard values of all entities in family family_num.
    void setFamilyOwnerWildcardValue(unsigned int family_num, unsigned int wc_num, const std::vector<float>& vals);

    /// @brief Set all clumps in this family to have this material.
    void setFamilyClumpMaterial(unsigned int N, unsigned int mat_id);
    /// @brief Set all meshes in this family to have this material.
    void setFamilyMeshMaterial(unsigned int N, unsigned int mat_id);

    /// @brief Set the geometry wildcards of mesh patches, starting from geoID, for the length of vals.
    void setPatchWildcardValue(bodyID_t geoID, unsigned int wc_num, const std::vector<float>& vals);
    /// @brief Set the geometry wildcards of spheres, starting from geoID, for the length of vals.
    void setSphWildcardValue(bodyID_t geoID, unsigned int wc_num, const std::vector<float>& vals);
    /// @brief Set the geometry wildcards of analytical components, starting from geoID, for the length of vals.
    void setAnalWildcardValue(bodyID_t geoID, unsigned int wc_num, const std::vector<float>& vals);

    /// @brief Returns the wildacard value of this owner, for n consecutive items.
    std::vector<float> getOwnerWildcardValue(bodyID_t ID, unsigned int wc_num, bodyID_t n = 1);
    /// @brief Fill res with the wc_num wildcard value.
    void getAllOwnerWildcardValue(std::vector<float>& res, unsigned int wc_num);
    /// @brief Fill res with the wc_num wildcard value for entities with family number family_num.
    void getFamilyOwnerWildcardValue(std::vector<float>& res, unsigned int family_num, unsigned int wc_num);

    /// @brief Fill res with the `wc_num' wildcard values, for n spheres starting from ID.
    void getSphereWildcardValue(std::vector<float>& res, bodyID_t ID, unsigned int wc_num, size_t n);
    /// @brief Fill res with the `wc_num' wildcard values, for n mesh patches starting from ID.
    void getPatchWildcardValue(std::vector<float>& res, bodyID_t ID, unsigned int wc_num, size_t n);
    /// @brief Fill res with the `wc_num' wildcard values, for n analytical entities starting from ID.
    void getAnalWildcardValue(std::vector<float>& res, bodyID_t ID, unsigned int wc_num, size_t n);

    /// @brief Change the value of contact wildcards no.wc_num to val if either of the contact geometries is in family
    /// N.
    void setFamilyContactWildcardValueEither(unsigned int N, unsigned int wc_num, float val);
    /// @brief Change the value of contact wildcards no.wc_num to val if both of the contact geometries are in family N.
    void setFamilyContactWildcardValueBoth(unsigned int N, unsigned int wc_num, float val);
    /// @brief Change the value of contact wildcards no.wc_num to val if the contacts are in family N1 and N2
    /// respectively.
    void setFamilyContactWildcardValue(unsigned int N1, unsigned int N2, unsigned int wc_num, float val);
    /// @brief Change the value of contact wildcards no.wc_num to val.
    void setContactWildcardValue(unsigned int wc_num, float val);

    /// @brief Get all forces concerning all provided owners.
    size_t getOwnerContactForces(const std::vector<bodyID_t>& ownerIDs,
                                 std::vector<float3>& points,
                                 std::vector<float3>& forces);
    /// @brief Get all forces concerning all provided owners.
    size_t getOwnerContactForces(const std::vector<bodyID_t>& ownerIDs,
                                 std::vector<float3>& points,
                                 std::vector<float3>& forces,
                                 std::vector<float3>& torques,
                                 bool torque_in_local = false);

    /// Get owner of contact geometry (sphere, triangle, analytical entity).
    bodyID_t getGeoOwnerID(const bodyID_t& geo, const geoType_t& type) const;

    /// Get the owner of a contact patch (triangle patch, sphere, analytical entity).
    bodyID_t getPatchOwnerID(const bodyID_t& patchID, const geoType_t& type) const;

    /// Let dT know that it needs a kT update, as something important may have changed, and old contact pair info is no
    /// longer valid.
    void announceCritical() { pendingCriticalUpdate = true; }

    /// @brief Change all entities with (user-level) family number ID_from to have a new number ID_to.
    void changeFamily(unsigned int ID_from, unsigned int ID_to);

    /// Resize arrays
    void allocateGPUArrays(size_t nOwnerBodies,
                           size_t nOwnerClumps,
                           unsigned int nExtObj,
                           size_t nTriMeshes,
                           size_t nSpheresGM,
                           size_t nTriGM,
                           size_t nTriNeighbors,
                           size_t nMeshPatches,
                           unsigned int nAnalGM,
                           size_t nExtraContacts,
                           unsigned int nMassProperties,
                           unsigned int nClumpTopo,
                           unsigned int nClumpComponents,
                           unsigned int nJitifiableClumpComponents,
                           unsigned int nMatTuples);

    // Components of initGPUArrays
    void buildTrackedObjs(const std::vector<std::shared_ptr<DEMClumpBatch>>& input_clump_batches,
                          const std::vector<unsigned int>& ext_obj_comp_num,
                          const std::vector<std::shared_ptr<DEMMesh>>& input_mesh_objs,
                          std::vector<std::shared_ptr<DEMTrackedObj>>& tracked_objs,
                          size_t nExistOwners,
                          size_t nExistSpheres,
                          size_t nExistingPatches,
                          unsigned int nExistingAnalGM);
    void populateEntityArrays(const std::vector<std::shared_ptr<DEMClumpBatch>>& input_clump_batches,
                              const std::vector<float3>& input_ext_obj_xyz,
                              const std::vector<float4>& input_ext_obj_rot,
                              const std::vector<unsigned int>& input_ext_obj_family,
                              const std::vector<std::shared_ptr<DEMMesh>>& input_mesh_objs,
                              const std::vector<float3>& input_mesh_obj_xyz,
                              const std::vector<float4>& input_mesh_obj_rot,
                              const std::vector<unsigned int>& input_mesh_obj_family,
                              const std::vector<notStupidBool_t>& input_mesh_obj_convex,
                              const std::vector<notStupidBool_t>& input_mesh_obj_never_winner,
                              const std::vector<unsigned int>& mesh_facet_owner,
                              const std::vector<bodyID_t>& mesh_facet_patch,
                              const std::vector<bodyID_t>& mesh_facet_neighbor1,
                              const std::vector<bodyID_t>& mesh_facet_neighbor2,
                              const std::vector<bodyID_t>& mesh_facet_neighbor3,
                              const std::vector<DEMTriangle>& mesh_facets,
                              const std::vector<bodyID_t>& mesh_patch_owner,
                              const std::vector<materialsOffset_t>& mesh_patch_materials,
                              const ClumpTemplateFlatten& clump_templates,
                              const std::vector<float>& ext_obj_mass_types,
                              const std::vector<float3>& ext_obj_moi_types,
                              const std::vector<unsigned int>& ext_obj_comp_num,
                              const std::vector<float>& mesh_obj_mass_types,
                              const std::vector<float3>& mesh_obj_moi_types,
                              const std::vector<inertiaOffset_t>& mesh_obj_mass_offsets,
                              size_t nExistOwners,
                              size_t nExistSpheres,
                              size_t nExistingFacets,
                              size_t nExistingPatches,
                              size_t nExistingTriNeighbors);
    void registerPolicies(const std::unordered_map<unsigned int, std::string>& template_number_name_map,
                          const ClumpTemplateFlatten& clump_templates,
                          const std::vector<float>& ext_obj_mass_types,
                          const std::vector<float3>& ext_obj_moi_types,
                          const std::vector<float>& mesh_obj_mass_types,
                          const std::vector<float3>& mesh_obj_moi_types,
                          const std::vector<std::shared_ptr<DEMMaterial>>& loaded_materials,
                          const std::vector<notStupidBool_t>& family_mask_matrix,
                          const std::set<unsigned int>& no_output_families);

    /// Initialized arrays
    void initGPUArrays(const std::vector<std::shared_ptr<DEMClumpBatch>>& input_clump_batches,
                       const std::vector<float3>& input_ext_obj_xyz,
                       const std::vector<float4>& input_ext_obj_rot,
                       const std::vector<unsigned int>& input_ext_obj_family,
                       const std::vector<std::shared_ptr<DEMMesh>>& input_mesh_objs,
                       const std::vector<float3>& input_mesh_obj_xyz,
                       const std::vector<float4>& input_mesh_obj_rot,
                       const std::vector<unsigned int>& input_mesh_obj_family,
                       const std::vector<notStupidBool_t>& input_mesh_obj_convex,
                       const std::vector<notStupidBool_t>& input_mesh_obj_never_winner,
                       const std::vector<unsigned int>& mesh_facet_owner,
                       const std::vector<bodyID_t>& mesh_facet_patch,
                       const std::vector<bodyID_t>& mesh_facet_neighbor1,
                       const std::vector<bodyID_t>& mesh_facet_neighbor2,
                       const std::vector<bodyID_t>& mesh_facet_neighbor3,
                       const std::vector<DEMTriangle>& mesh_facets,
                       const std::vector<bodyID_t>& mesh_patch_owner,
                       const std::vector<materialsOffset_t>& mesh_patch_materials,
                       const std::unordered_map<unsigned int, std::string>& template_number_name_map,
                       const ClumpTemplateFlatten& clump_templates,
                       const std::vector<float>& ext_obj_mass_types,
                       const std::vector<float3>& ext_obj_moi_types,
                       const std::vector<unsigned int>& ext_obj_comp_num,
                       const std::vector<float>& mesh_obj_mass_types,
                       const std::vector<float3>& mesh_obj_moi_types,
                       const std::vector<float>& mesh_obj_mass_jit_types,
                       const std::vector<float3>& mesh_obj_moi_jit_types,
                       const std::vector<inertiaOffset_t>& mesh_obj_mass_offsets,
                       const std::vector<std::shared_ptr<DEMMaterial>>& loaded_materials,
                       const std::vector<notStupidBool_t>& family_mask_matrix,
                       const std::set<unsigned int>& no_output_families,
                       std::vector<std::shared_ptr<DEMTrackedObj>>& tracked_objs);

    /// Add more clumps and/or meshes into the system, without re-initialization. It must be clump/mesh-addition only,
    /// no other changes to the system.
    void updateClumpMeshArrays(const std::vector<std::shared_ptr<DEMClumpBatch>>& input_clump_batches,
                               const std::vector<float3>& input_ext_obj_xyz,
                               const std::vector<float4>& input_ext_obj_rot,
                               const std::vector<unsigned int>& input_ext_obj_family,
                               const std::vector<std::shared_ptr<DEMMesh>>& input_mesh_objs,
                               const std::vector<float3>& input_mesh_obj_xyz,
                               const std::vector<float4>& input_mesh_obj_rot,
                               const std::vector<unsigned int>& input_mesh_obj_family,
                               const std::vector<notStupidBool_t>& input_mesh_obj_convex,
                               const std::vector<notStupidBool_t>& input_mesh_obj_never_winner,
                               const std::vector<unsigned int>& mesh_facet_owner,
                               const std::vector<bodyID_t>& mesh_facet_patch,
                               const std::vector<bodyID_t>& mesh_facet_neighbor1,
                               const std::vector<bodyID_t>& mesh_facet_neighbor2,
                               const std::vector<bodyID_t>& mesh_facet_neighbor3,
                               const std::vector<DEMTriangle>& mesh_facets,
                               const std::vector<bodyID_t>& mesh_patch_owner,
                               const std::vector<materialsOffset_t>& mesh_patch_materials,
                               const ClumpTemplateFlatten& clump_templates,
                               const std::vector<float>& ext_obj_mass_types,
                               const std::vector<float3>& ext_obj_moi_types,
                               const std::vector<unsigned int>& ext_obj_comp_num,
                               const std::vector<float>& mesh_obj_mass_types,
                               const std::vector<float3>& mesh_obj_moi_types,
                               const std::vector<float>& mesh_obj_mass_jit_types,
                               const std::vector<float3>& mesh_obj_moi_jit_types,
                               const std::vector<inertiaOffset_t>& mesh_obj_mass_offsets,
                               const std::vector<std::shared_ptr<DEMMaterial>>& loaded_materials,
                               const std::vector<notStupidBool_t>& family_mask_matrix,
                               const std::set<unsigned int>& no_output_families,
                               std::vector<std::shared_ptr<DEMTrackedObj>>& tracked_objs,
                               size_t nExistingOwners,
                               size_t nExistingClumps,
                               size_t nExistingSpheres,
                               size_t nExistingTriMesh,
                               size_t nExistingFacets,
                               size_t nExistingTriNeighbors,
                               size_t nExistingPatches,
                               unsigned int nExistingObj,
                               unsigned int nExistingAnalGM);

    /// Change radii and relPos info of these owners (if these owners are clumps)
    void changeOwnerSizes(const std::vector<bodyID_t>& IDs, const std::vector<float>& factors);

    /// Put sim data array pointers in place
    void packDataPointers();
    void packTransferPointers(DEMKinematicThread*& kT);

    // Move array data to or from device
    void migrateDataToDevice();
    // void migrateDataToHost();

    // Generate contact info container based on the current contact array, and return it.
    std::shared_ptr<ContactInfoContainer> generateContactInfo(float force_thres);
    std::shared_ptr<ContactInfoContainer> generateContactInfoFromHost(float force_thres);

#ifdef DEME_USE_CHPF
    void writeSpheresAsChpf(std::ofstream& ptFile);
    void writeClumpsAsChpf(std::ofstream& ptFile, unsigned int accuracy = 10);
    void writeSpheresAsChpfFromHost(std::ofstream& ptFile);
    void writeClumpsAsChpfFromHost(std::ofstream& ptFile, unsigned int accuracy = 10);
#endif
    void writeSpheresAsCsv(std::ofstream& ptFile);
    void writeClumpsAsCsv(std::ofstream& ptFile, unsigned int accuracy = 10);
    void writeContactsAsCsv(std::ofstream& ptFile, float force_thres = DEME_TINY_FLOAT);
    void writeMeshesAsVtk(std::ofstream& ptFile);
    void writeMeshesAsStl(std::ofstream& ptFile);
    void writeMeshesAsPly(std::ofstream& ptFile, bool patch_colors = false);
    void writeSpheresAsCsvFromHost(std::ofstream& ptFile);
    void writeClumpsAsCsvFromHost(std::ofstream& ptFile, unsigned int accuracy = 10);
    void writeContactsAsCsvFromHost(std::ofstream& ptFile, float force_thres = DEME_TINY_FLOAT);
    void writeMeshesAsVtkFromHost(std::ofstream& ptFile);
    void writeMeshesAsStlFromHost(std::ofstream& ptFile);
    void writeMeshesAsPlyFromHost(std::ofstream& ptFile, bool patch_colors = false);

    /// Called each time when the user calls DoDynamicsThenSync.
    void startThread();

    // The actual kernel things go here.
    // It is called upon construction.
    void workerThread();

    // Sync my stream
    void syncMemoryTransfer() {
        DEME_GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
    }

    // Record and sync a reusable event on the main stream
    void recordAndSyncEvent();
    // Record only; sync later
    void recordEventOnly();
    // Record a progress event for non-blocking completion tracking.
    void recordProgressEvent(int64_t stamp);
    // Drain completed progress events and update completion stamp.
    void drainProgressEvents();
    // Keep host enqueueing close to GPU progress by limiting in-flight steps.
    void throttleInFlightProgress();
    // Sync the previously recorded event
    void syncRecordedEvent();

    // Reset kT--dT interaction coordinator stats
    void resetUserCallStat();
    // Return the approximate RAM usage
    size_t estimateDeviceMemUsage() const;
    size_t estimateHostMemUsage() const;

    /// Return timing inforation for this current run
    void getTiming(std::vector<std::string>& names, std::vector<double>& vals);

    /// Reset the timers
    void resetTimers() {
        for (const auto& name : timer_names) {
            timers.GetTimer(name).reset();
        }
    }

    /// Get the simulation time passed since the start of simulation
    double getSimTime() const;
    /// Set the simulation time manually
    void setSimTime(double time);

    // Jitify dT kernels (at initialization) based on existing knowledge of this run
    void jitifyKernels(const std::unordered_map<std::string, std::string>& Subs,
                       const std::vector<std::string>& JitifyOptions);

    // Execute this kernel, then return the reduced value
    float* inspectCall(const std::shared_ptr<JitHelper::CachedProgram>& inspection_kernel,
                       const std::string& kernel_name,
                       INSPECT_ENTITY_TYPE thing_to_insp,
                       CUB_REDUCE_FLAVOR reduce_flavor,
                       bool all_domain,
                       DualArray<scratch_t>& reduceResArr,
                       DualArray<scratch_t>& reduceRes,
                       bool return_device_ptr = false);

    // Device-only inspection helper (no host sync); intended for non-reduce paths.
    float* inspectCallDeviceNoReduce(const std::shared_ptr<JitHelper::CachedProgram>& inspection_kernel,
                                     const std::string& kernel_name,
                                     INSPECT_ENTITY_TYPE thing_to_insp,
                                     CUB_REDUCE_FLAVOR reduce_flavor,
                                     bool all_domain,
                                     DualArray<scratch_t>& reduceResArr,
                                     DualArray<scratch_t>& reduceRes);

  private:
    // Name for this class
    const std::string Name = "dT";

    // If true, then the user manually loaded extra contacts to the system. In this case, not only we need to wait for
    // an initial update from kT, we also need to update kT's previous-step contact arrays, so it properly builds
    // contact map for dT.
    bool new_contacts_loaded = false;

    // Meshes cached on dT side that has corresponding owner number associated. Useful for outputting meshes.
    std::vector<std::shared_ptr<DEMMesh>> m_meshes;

    // Number of trackers I already processed before (if I see a tracked_obj array longer than this in initialization, I
    // know I have to process the new-comers)
    unsigned int nTrackersProcessed = 0;

    // A pointer that points to the location that holds the current max_vel info, which will soon be transferred to kT
    float* pCycleVel;
    // Pointer that points to the location that holds the current angular velocity magnitude info, which will soon be
    // transferred to kT
    float* pCycleAngVel;

    // The inspector for calculating max vel for this cycle
    std::shared_ptr<DEMInspector> approxVelFunc;
    // The inspector for calculating angular velocity magnitude for this cycle
    std::shared_ptr<DEMInspector> approxAngVelFunc;

    // Migrate contact history to fit the structure of the newly received contact array
    inline void migrateEnduringContacts();

    // Impl of calculateForces
    inline void dispatchPrimitiveForceKernels(
        const ContactTypeMap<std::pair<contactPairs_t, contactPairs_t>>& typeStartCountMap,
        const ContactTypeMap<std::vector<std::pair<std::shared_ptr<JitHelper::CachedProgram>, std::string>>>&
            typeKernelMap);
    inline void dispatchPatchBasedForceCorrections(
        const ContactTypeMap<std::pair<contactPairs_t, contactPairs_t>>& typeStartCountPrimitiveMap,
        const ContactTypeMap<std::pair<contactPairs_t, contactPairs_t>>& typeStartCountPatchMap,
        const ContactTypeMap<std::vector<std::pair<std::shared_ptr<JitHelper::CachedProgram>, std::string>>>&
            typeKernelMap);
    // Update clump-based acceleration array based on sphere-based force array
    void calculateForces();

    // Update clump pos/oriQ and vel/omega based on acceleration
    inline void integrateOwnerMotions();

    // If kT provides fresh CD results, we unpack and use it
    inline void ifProduceFreshThenUseItAndSendNewOrder();
    inline void ifProduceFreshThenUseIt(bool allow_blocking);
    bool tryConsumeKinematicProduce(bool allow_blocking, bool mark_receive, bool use_logical_stamp);
    inline void unpack_impl();

    // Change sim params based on dT's experience, if needed
    inline void calibrateParams();

    // Determine the vel info for this cycle, kT needs it
    inline void determineSysVel();

    // Some per-step checks/modification, done before integration, but after force calculation (thus sort of in the
    // mid-step stage)
    inline void routineChecks();

    // Bring dT buffer array data to its working arrays
    inline void unpackMyBuffer();
    // Send produced data to kT-owned biffers
    void sendToTheirBuffer();
    // Resize some work arrays based on the number of contact pairs provided by kT
    void contactPrimitivesArraysResize(size_t nContactPairs);
    // Resize mesh patch pair array based on the number of mesh-involved contact pairs
    void contactPatchArrayResize(size_t nMeshInvolvedContactPairs);

    // Deallocate everything
    void deallocateEverything();
    // The dT-side allocations that can be done at initialization time
    void initAllocation();

    // Wildcard setting impl function
    void setFamilyContactWildcardValue_impl(
        unsigned int N1,
        unsigned int N2,
        unsigned int wc_num,
        float val,
        const std::function<bool(unsigned int, unsigned int, unsigned int, unsigned int)>& condition);

    // Just-in-time compiled kernels
    std::shared_ptr<JitHelper::CachedProgram> cal_force_kernels;
    std::shared_ptr<JitHelper::CachedProgram> cal_patch_force_kernels;
    std::shared_ptr<JitHelper::CachedProgram> collect_force_kernels;
    std::shared_ptr<JitHelper::CachedProgram> integrator_kernels;
    // std::shared_ptr<JitHelper::CachedProgram> quarry_stats_kernels;
    std::shared_ptr<JitHelper::CachedProgram> mod_kernels;
    void prewarmKernels();

    // Curcial Drift optimizer
    struct FutureDriftRegulator {
        double last_total_time = 0.0;
        double debug_cum_time = 0.0;
        double last_debug_cum_time = 0.0;

        uint64_t last_step_sample = 0;
        bool has_last_step_sample = false;

        // calibrateParams may be called multiple times; only the first call after a kT update should advance the
        // timing baseline and update the tuner.
        bool receive_pending = false;
        uint64_t pending_recv_stamp = 0;
        double pending_total_time = 0.0;

        static constexpr int COST_WINDOW = 100;
        double cost_window[COST_WINDOW] = {0.0};
        unsigned int drift_window[COST_WINDOW] = {0u};
        int window_size = 0;
        int window_pos = 0;

        // Last chosen TOTAL drift target (de-headroomed).
        unsigned int last_proposed = 0;
        // The max drift command (with safety headroom) that was last sent to kT.
        unsigned int last_sent_proposed = 0;
        // The TRUE drift target used for the last work order (de-headroomed).
        unsigned int last_sent_true = 0;
        // WAIT (intentional delay in dT steps) used for the last work order.
        unsigned int last_sent_wait = 0;

        // Last WAIT we computed (actuator space).
        unsigned int last_wait_cmd = 0;

        // Smoothed estimate of kT lag (in dT steps, pipeline-corrected); used for scheduling only.
        double lag_ema = 0.0;
        bool lag_ema_initialized = false;

        // Next kT work order scheduling (in units of dT steps, i.e., nTotalSteps).
        uint64_t next_send_step = 0;
        unsigned int next_send_wait = 0;
        bool pending_send = false;

        unsigned int last_observed_kinematic_lag_steps = 0;

        DriftRLS drift_rls;
        uint64_t drift_rls_samples = 0;
        double drift_scale_ema = 0.0;
        bool drift_scale_initialized = false;
        double cost_scale_ema = 0.0;
        bool cost_scale_initialized = false;

        void Clear() { *this = FutureDriftRegulator{}; }
    };
    FutureDriftRegulator futureDriftRegulator;
    // Helpers for future drift regulator -
    static inline unsigned clamp_drift_u(unsigned v, unsigned maxv) {
        return (v < 1u) ? 1u : (v > maxv ? maxv : v);
    }
    static inline unsigned clamp_wait_i(int v, unsigned maxv) {
        if (v <= 0)
            return 0u;
        const unsigned u = (unsigned)v;
        return (u > maxv) ? maxv : u;
    }
    static inline unsigned apply_wait_policy_u(unsigned w,
                                               double lag_pred,
                                               double upper_ratio,
                                               double lower_ratio,
                                               unsigned maxv) {
        w = (w > maxv) ? maxv : w;
        double total = lag_pred + (double)w;
        if (total < 1.0)
            total = 1.0;
        if (upper_ratio <= 0.0)
            w = 0u;
        else if (upper_ratio < 1.0) {
            const unsigned uw = clamp_wait_i((int)std::ceil(total * upper_ratio), maxv);
            if (w > uw)
                w = uw;
        }
        if (lower_ratio > 0.0) {
            const unsigned lw = clamp_wait_i((int)std::floor(total * lower_ratio), maxv);
            if (w <= lw)
                w = 0u;
        }
        return w;
    }
    static inline void ema_asym(double& ema, bool& init, double x, double a_up, double a_dn, double minv) {
        if (!init) {
            init = true;
            ema = x;
        } else {
            const double a = (x > ema) ? a_up : a_dn;
            ema = (1.0 - a) * ema + a * x;
        }
        if (!std::isfinite(ema) || ema < minv)
            ema = std::max(minv, x);
    }
    static inline void ring_push(FutureDriftRegulator& r, double cost, unsigned obs) {
        const int W = FutureDriftRegulator::COST_WINDOW;
        const int i = r.window_pos;
        r.cost_window[i] = cost;
        r.drift_window[i] = obs;
        if (r.window_size < W)
            r.window_size++;
        r.window_pos = (i + 1) % W;
    }
    static inline double drift_ref_quantile(const FutureDriftRegulator& r, double floor_ref) {
        constexpr int WIN = 30;
        const int n = std::min(r.window_size, WIN);
        if (n <= 0)
            return floor_ref;
        std::array<unsigned, WIN> a;  // no zero-init
        int idx = (r.window_pos > 0) ? (r.window_pos - 1) : (FutureDriftRegulator::COST_WINDOW - 1);
        for (int i = 0; i < n; ++i) {
            a[i] = r.drift_window[idx];
            idx = (idx > 0) ? (idx - 1) : (FutureDriftRegulator::COST_WINDOW - 1);
        }
        const int q = n / 5;  // Quantile - IMPRORTANT
        std::nth_element(a.begin(), a.begin() + q, a.begin() + n);
        double qv = (double)a[q];
        if (r.drift_scale_initialized)
            qv = std::min(qv, r.drift_scale_ema);
        return std::max(floor_ref, qv);
    }
    static inline bool rls_is_bad(const DriftRLS& rls, unsigned obs, double drift_ref, double scale) {
        const double yhat = rls.predict(obs, drift_ref);
        if (!std::isfinite(yhat) || std::abs(yhat) > 1000.0 * scale)
            return true;
        const double clip = std::max(1e-3, 1000.0 * scale);
        for (int i = 0; i < DriftRLS::N; ++i) {
            const double t = rls.theta[i];
            if (!std::isfinite(t) || std::abs(t) > clip)
                return true;
        }
        return false;
    }

    // A collection of migrate-to-host methods. Bulk migrate-to-host is by nature on-demand only.
    void migrateFamilyToHost();
    void migrateClumpPosInfoToHost();
    void migrateClumpHighOrderInfoToHost();
    void migrateOwnerWildcardToHost();
    void migrateSphGeoWildcardToHost();
    void migratePatchGeoWildcardToHost();
    void migrateAnalGeoWildcardToHost();
    void migrateContactInfoToHost();

    void migrateDeviceModifiableInfoToHost();

};  // dT ends

}  // namespace deme

#endif
