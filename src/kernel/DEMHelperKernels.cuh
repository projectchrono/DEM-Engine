// DEM device-side helper kernel collection

#ifndef DEME_HELPER_KERNELS_CUH
#define DEME_HELPER_KERNELS_CUH

#include "CUDAMathHelpers.cuh"
// If being statically compiled using CMake, then include DEM/Defines.h using relative path.
// Otherwise, use the angle bracket form for jitify to pick it up from the build dir.
#if __has_include("../DEM/Defines.h")
    #include "../DEM/Defines.h"
#else
    #include <DEM/Defines.h>
#endif

//   __device__ voxelID_t position2VoxelID

////////////////////////////////////////////////////////////////////////////////
// A few helper functions specific to DEM module
////////////////////////////////////////////////////////////////////////////////

// Sign function
template <typename T1>
__host__ __device__ int sgn(const T1& val) {
    return (T1(0) < val) - (val < T1(0));
}

template <typename T1>
__host__ __device__ T1 dot3(const T1& x1, const T1& x2, const T1& x3, const T1& y1, const T1& y2, const T1& y3) {
    return x1 * y1 + x2 * y2 + x3 * y3;
}

// Integer division that rounds towards -infty
template <typename T1, typename T2>
__host__ __device__ T1 div_floor(const T1& a, const T2& b) {
    T1 res = a / b;
    T1 rem = a % b;
    // Correct division result downwards if up-rounding happened,
    // (for non-zero remainder of sign different than the divisor).
    T1 corr = (rem != 0 && ((rem < 0) != (b < 0)));
    return res - corr;
}

// Modulus that rounds towards -infty
template <typename T1, typename T2>
__host__ __device__ T1 mod_floor(const T1& a, const T2& b) {
    if (b < 0)  // you can check for b == 0 separately and do what you want
        return -mod_floor(-a, -b);
    T1 ret = a % b;
    if (ret < 0)
        ret += b;
    return ret;
}

// In an upper-triangular (including the diagonal part) matrix, given i and j, this function returns the index of the
// corresponding flatten-ed non-zero entries (col-major like in matlab). This function does not assume i <= j. It is
// used in locating masks that maps the contact between families.
template <typename T1>
__host__ __device__ T1 locateMaskPair(const T1& i, const T1& j) {
    if (i > j)
        return locateMaskPair(j, i);
    return (1 + j) * j / 2 + i;
}

// Magic function that converts an index of a flatten-ed upper-triangular matrix (EXCLUDING the diagonal) to its
// corresponding i and j. It is ROW-major. It is used to map contact pair numbers in a bin.
template <typename T1>
__host__ __device__ void recoverCntPair(T1& i, T1& j, const T1& ind, const T1& n) {
    i = n - 2 - (T1)(sqrt((float)(4 * n * (n - 1) - 7 - 8 * ind)) / 2.0 - 0.5);
    j = ind + i + 1 + (n - i) * ((n - i) - 1) / 2 - n * (n - 1) / 2;
}

// Cylindrical periodic ghost ID helpers
__host__ __device__ inline deme::bodyID_t cylPeriodicEncodeGhostID(deme::bodyID_t id, bool neg = false) {
    return id | deme::CYL_PERIODIC_GHOST_FLAG | (neg ? deme::CYL_PERIODIC_GHOST_NEG_FLAG : 0);
}

// Encode a +/-1 cylindrical-periodic ghost shift into an ID (shift=0 clears ghost flags).
__host__ __device__ inline deme::bodyID_t cylPeriodicEncodeGhostShift(deme::bodyID_t id, int shift) {
    id &= deme::CYL_PERIODIC_SPHERE_ID_MASK;
    if (shift == 0) {
        return id;
    }
    return cylPeriodicEncodeGhostID(id, shift < 0);
}

__host__ __device__ inline deme::bodyID_t cylPeriodicDecodeID(deme::bodyID_t id, bool& is_ghost, bool& neg) {
    is_ghost = (id & deme::CYL_PERIODIC_GHOST_FLAG) != 0;
    neg = (id & deme::CYL_PERIODIC_GHOST_NEG_FLAG) != 0;
    return id & deme::CYL_PERIODIC_SPHERE_ID_MASK;
}
__host__ __device__ inline deme::bodyID_t cylPeriodicDecodeID(deme::bodyID_t id, bool& is_ghost) {
    bool neg = false;
    return cylPeriodicDecodeID(id, is_ghost, neg);
}

// Rotate a position around the cylindrical axis by a given angle (cos/sin provided).
__host__ __device__ inline float3 cylPeriodicRotate(const float3& pos,
                                                    const float3& origin,
                                                    const float3& axis,
                                                    const float3& u,
                                                    const float3& v,
                                                    float cos_theta,
                                                    float sin_theta) {
    float3 p = pos - origin;
    const float axial = dot(p, axis);
    const float pu = dot(p, u);
    const float pv = dot(p, v);
    const float pu2 = cos_theta * pu - sin_theta * pv;
    const float pv2 = sin_theta * pu + cos_theta * pv;
    float3 rotated = axis * axial + u * pu2 + v * pv2;
    return rotated + origin;
}

// Rotate a global position (with LBF offset) around the cylindrical axis by a given angle.
__host__ __device__ inline float3 cylPeriodicRotatePosGlobal(const float3& pos,
                                                            const deme::DEMSimParams* simParams,
                                                            float cos_theta,
                                                            float sin_theta) {
    float3 pos_local = make_float3(pos.x - simParams->LBFX, pos.y - simParams->LBFY, pos.z - simParams->LBFZ);
    pos_local = cylPeriodicRotate(pos_local, simParams->cylPeriodicOrigin, simParams->cylPeriodicAxisVec,
                                  simParams->cylPeriodicU, simParams->cylPeriodicV, cos_theta, sin_theta);
    pos_local.x += simParams->LBFX;
    pos_local.y += simParams->LBFY;
    pos_local.z += simParams->LBFZ;
    return pos_local;
}

// Compute angle in [0, 2*pi) around the cylindrical axis.
__host__ __device__ inline float cylPeriodicAngle(const float3& pos,
                                                  const float3& origin,
                                                  const float3& u,
                                                  const float3& v) {
    float3 p = pos - origin;
    const float pu = dot(p, u);
    const float pv = dot(p, v);
    float angle = atan2f(pv, pu);
    if (angle < 0.f) {
        angle += 2.f * deme::PI;
    }
    return angle;
}

__host__ __device__ inline float cylPeriodicRelAngle(const float3& pos,
                                                     bool is_ghost,
                                                     bool ghost_neg,
                                                     const deme::DEMSimParams* simParams) {
    float angle = cylPeriodicAngle(pos, simParams->cylPeriodicOrigin, simParams->cylPeriodicU, simParams->cylPeriodicV);
    if (is_ghost) {
        angle += ghost_neg ? simParams->cylPeriodicSpan : -simParams->cylPeriodicSpan;
        if (angle < 0.f) {
            angle += 2.f * deme::PI;
        } else if (angle >= 2.f * deme::PI) {
            angle -= 2.f * deme::PI;
        }
    }
    float rel = angle - simParams->cylPeriodicStart;
    const float eps = 1e-6f;  // radians; numerical tolerance around the boundary
    if (rel < 0.f) {
        if (-rel <= eps) {
            rel = 0.f;
        } else {
            rel += 2.f * deme::PI;
        }
    }
    // NOTE: Do NOT wrap rel-angle values that are numerically at/just above the end plane
    // back to ~0.0. Doing so suppresses required cross-boundary (ghost) interactions.
    // Teleportation enforces the configuration to remain within the wedge; here we only
    // clamp tiny numerical overshoots.
    if (rel >= simParams->cylPeriodicSpan) {
        // NOTE: Do NOT modulo-wrap rel-angle values that are genuinely outside the wedge back into [0, span).
        // Values > span convey that the point is outside the wedge (allowed transiently before teleportation),
        // and folding them back causes min-image / ghost selection to flip-flop and inject energy.
        if (rel - simParams->cylPeriodicSpan <= eps) {
            // Clamp tiny numerical overshoot to the boundary.
            rel = simParams->cylPeriodicSpan;
        } else {
            // Keep rel as-is (no wrapping).
        }
    }
    return rel;
}

// Estimate whether an owner should generate periodic ghosts at the start/end planes.
__host__ __device__ inline void cylPeriodicGhostAvailability(const float3& ownerPosGlobal,
                                                             float boundRadius,
                                                             const deme::DEMSimParams* simParams,
                                                             bool& avail_pos,
                                                             bool& avail_neg) {
    const float eps = 1e-6f;
    const float max_other = fmaxf(simParams->maxSphereRadius, simParams->maxTriRadius);
    const float other_margin = simParams->dyn.beta + simParams->maxFamilyExtraMargin;
    const float ghost_dist = fmaxf(boundRadius, 0.f) + max_other + other_margin + eps;
    const float3 pos_local = make_float3(ownerPosGlobal.x - simParams->LBFX,
                                         ownerPosGlobal.y - simParams->LBFY,
                                         ownerPosGlobal.z - simParams->LBFZ);
    const float3 pos_global = pos_local - simParams->cylPeriodicOrigin;
    const float dist_start = dot(pos_global, simParams->cylPeriodicStartNormal);
    const float dist_end = dot(pos_global, simParams->cylPeriodicEndNormal);
    avail_pos = dist_start <= ghost_dist;
    avail_neg = dist_end >= -ghost_dist;
}

// Estimate whether a component should generate periodic ghosts at the start/end planes.
// pos_local is in the same coordinate frame as kT contact detection (no LBF offset).
__host__ __device__ inline void cylPeriodicComponentGhostAvailability(const float3& pos_local,
                                                                      float radius,
                                                                      const deme::DEMSimParams* simParams,
                                                                      bool& avail_pos,
                                                                      bool& avail_neg) {
    const float eps = 1e-6f;
    const float max_other = fmaxf(simParams->maxSphereRadius, simParams->maxTriRadius);
    const float other_margin = simParams->dyn.beta + simParams->maxFamilyExtraMargin;
    const float ghost_dist = fmaxf(radius, 0.f) + max_other + other_margin + eps;
    const float3 pos_global = pos_local - simParams->cylPeriodicOrigin;
    const float dist_start = dot(pos_global, simParams->cylPeriodicStartNormal);
    const float dist_end = dot(pos_global, simParams->cylPeriodicEndNormal);
    avail_pos = dist_start <= ghost_dist;
    avail_neg = dist_end >= -ghost_dist;
}

// Choose a single, deterministic periodic-image shift for a pair based on rel-angles and ghost availability.
// This avoids double-counting cross-boundary contacts while keeping one active representation.
__host__ __device__ inline void cylPeriodicSelectGhostShift(float relA,
                                                           float relB,
                                                           bool availA_pos,
                                                           bool availA_neg,
                                                           bool availB_pos,
                                                           bool availB_neg,
                                                           float span,
                                                           int& shiftA,
                                                           int& shiftB) {
    shiftA = 0;
    shiftB = 0;
    if (span <= 0.f) {
        return;
    }
    const float d = fabsf(relA - relB);
    const float half_span = 0.5f * span;
    const float eps = 1e-6f;
    if (d <= half_span + eps) {
        return;
    }
    int desA = 0, desB = 0;
    int altA = 0, altB = 0;
    if (relA < relB) {
        desA = 1;
        altB = -1;
    } else {
        desB = 1;
        altA = -1;
    }
    bool des_ok = true;
    if (desA == 1 && !availA_pos) des_ok = false;
    if (desA == -1 && !availA_neg) des_ok = false;
    if (desB == 1 && !availB_pos) des_ok = false;
    if (desB == -1 && !availB_neg) des_ok = false;
    if (des_ok) {
        shiftA = desA;
        shiftB = desB;
        return;
    }
    bool alt_ok = true;
    if (altA == 1 && !availA_pos) alt_ok = false;
    if (altA == -1 && !availA_neg) alt_ok = false;
    if (altB == 1 && !availB_pos) alt_ok = false;
    if (altB == -1 && !availB_neg) alt_ok = false;
    if (alt_ok) {
        shiftA = altA;
        shiftB = altB;
    }
}

// Decide whether to keep a ghost-involved contact pair, avoiding duplicate cross-boundary contacts.
__host__ __device__ inline float cylPeriodicDist2(const float3& a, const float3& b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    const float dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

// Apply a ±span shift (or none) to a global position.
__host__ __device__ inline float3
cylPeriodicShiftPosGlobal(const float3& pos, const deme::DEMSimParams* simParams, int shift) {
    if (shift == 0) {
        return pos;
    }
    const float cos_theta = simParams->cylPeriodicCosSpan;
    const float sin_theta = (shift > 0) ? simParams->cylPeriodicSinSpan : -simParams->cylPeriodicSinSpan;
    return cylPeriodicRotatePosGlobal(pos, simParams, cos_theta, sin_theta);
}

// Distance between two positions after applying integer span shifts (±1 or 0).
__host__ __device__ inline float cylPeriodicShiftDist2(const float3& posA_un,
                                                       const float3& posB_un,
                                                       const deme::DEMSimParams* simParams,
                                                       int shiftA,
                                                       int shiftB) {
    const float3 A = cylPeriodicShiftPosGlobal(posA_un, simParams, shiftA);
    const float3 B = cylPeriodicShiftPosGlobal(posB_un, simParams, shiftB);
    return cylPeriodicDist2(A, B);
}

// Select the canonical (minimum-image) representation for a body pair under cylindrical wedge periodicity.
// posA_un/posB_un are global positions (LBF-shifted); this helper applies LBF-aware rotations internally.
// We compare COM distances between:
//   (A0,B0), (A+1,B0), (A-1,B0), (A0,B+1), (A0,B-1)
// and keep exactly ONE image per pair. This decision is independent of primitive-level radii to
// avoid per-primitive image flips (critical for triangle meshes / patch aggregation).
__host__ __device__ inline void cylPeriodicSelectGhostShiftByDist(const float3& posA_un,
                                                                 const float3& posB_un,
                                                                 const deme::DEMSimParams* simParams,
                                                                 int& shiftA,
                                                                 int& shiftB) {
    shiftA = 0;
    shiftB = 0;
    float best = cylPeriodicDist2(posA_un, posB_un);
    const float eps = 1e-6f * (1.f + best);

    // Rotate A by +span / -span
    {
        const float3 Apos = cylPeriodicRotatePosGlobal(posA_un, simParams, simParams->cylPeriodicCosSpan,
                                                       simParams->cylPeriodicSinSpan);
        const float d = cylPeriodicDist2(Apos, posB_un);
        if (d + eps < best) {
            best = d;
            shiftA = 1;
            shiftB = 0;
        }
    }
    {
        const float3 Aneg = cylPeriodicRotatePosGlobal(posA_un, simParams, simParams->cylPeriodicCosSpan,
                                                       -simParams->cylPeriodicSinSpan);
        const float d = cylPeriodicDist2(Aneg, posB_un);
        if (d + eps < best) {
            best = d;
            shiftA = -1;
            shiftB = 0;
        }
    }

    // Rotate B by +span / -span
    {
        const float3 Bpos = cylPeriodicRotatePosGlobal(posB_un, simParams, simParams->cylPeriodicCosSpan,
                                                       simParams->cylPeriodicSinSpan);
        const float d = cylPeriodicDist2(posA_un, Bpos);
        if (d + eps < best) {
            best = d;
            shiftA = 0;
            shiftB = 1;
        }
    }
    {
        const float3 Bneg = cylPeriodicRotatePosGlobal(posB_un, simParams, simParams->cylPeriodicCosSpan,
                                                       -simParams->cylPeriodicSinSpan);
        const float d = cylPeriodicDist2(posA_un, Bneg);
        if (d + eps < best) {
            best = d;
            shiftA = 0;
            shiftB = -1;
        }
    }
}

// Select the canonical (minimum-image) representation for a body pair, honoring ghost availability.
// Only candidate images whose owner-side ghost is available are considered.
__host__ __device__ inline void cylPeriodicSelectGhostShiftByDistAvail(const float3& posA_un,
                                                                       const float3& posB_un,
                                                                       const deme::DEMSimParams* simParams,
                                                                       bool availA_pos,
                                                                       bool availA_neg,
                                                                       bool availB_pos,
                                                                       bool availB_neg,
                                                                       int& shiftA,
                                                                       int& shiftB,
                                                                       float& bestDist) {
    shiftA = 0;
    shiftB = 0;
    float best = cylPeriodicDist2(posA_un, posB_un);
    const float eps = 1e-6f * (1.f + best);

    if (availA_pos) {
        const float3 Apos = cylPeriodicRotatePosGlobal(posA_un, simParams, simParams->cylPeriodicCosSpan,
                                                       simParams->cylPeriodicSinSpan);
        const float d = cylPeriodicDist2(Apos, posB_un);
        if (d + eps < best) {
            best = d;
            shiftA = 1;
            shiftB = 0;
        }
    }
    if (availA_neg) {
        const float3 Aneg = cylPeriodicRotatePosGlobal(posA_un, simParams, simParams->cylPeriodicCosSpan,
                                                       -simParams->cylPeriodicSinSpan);
        const float d = cylPeriodicDist2(Aneg, posB_un);
        if (d + eps < best) {
            best = d;
            shiftA = -1;
            shiftB = 0;
        }
    }
    if (availB_pos) {
        const float3 Bpos = cylPeriodicRotatePosGlobal(posB_un, simParams, simParams->cylPeriodicCosSpan,
                                                       simParams->cylPeriodicSinSpan);
        const float d = cylPeriodicDist2(posA_un, Bpos);
        if (d + eps < best) {
            best = d;
            shiftA = 0;
            shiftB = 1;
        }
    }
    if (availB_neg) {
        const float3 Bneg = cylPeriodicRotatePosGlobal(posB_un, simParams, simParams->cylPeriodicCosSpan,
                                                       -simParams->cylPeriodicSinSpan);
        const float d = cylPeriodicDist2(posA_un, Bneg);
        if (d + eps < best) {
            best = d;
            shiftA = 0;
            shiftB = -1;
        }
    }

    bestDist = best;
}

// True if owner COM (global coordinates) is near either periodic seam plane within `band`.
__host__ __device__ inline bool cylPeriodicNearSeam(const float3& pos_global,
                                                    float band,
                                                    const deme::DEMSimParams* simParams) {
    const float dist_start = dot(pos_global, simParams->cylPeriodicStartNormal);
    const float dist_end = dot(pos_global, simParams->cylPeriodicEndNormal);
    return (fabsf(dist_start) <= band) || (fabsf(dist_end) <= band);
}

// Shared near-seam branch-hysteresis predicate used by dT/kT.
__host__ __device__ inline bool cylPeriodicShouldKeepGhostShiftNearSeamImpl(const float3& posA_un,
                                                                            float radA,
                                                                            const float3& posB_un,
                                                                            float radB,
                                                                            int ghostShiftA,
                                                                            int ghostShiftB,
                                                                            int desiredShiftA,
                                                                            int desiredShiftB,
                                                                            float bestDist2,
                                                                            float bandScale,
                                                                            float absTolScale,
                                                                            float relTolScale,
                                                                            float absTolFloor,
                                                                            const deme::DEMSimParams* simParams) {
    if (ghostShiftA == desiredShiftA && ghostShiftB == desiredShiftB) {
        return false;
    }
    // Never revive non-minimal ghost-ghost representation.
    if (ghostShiftA != 0 && ghostShiftB != 0) {
        return false;
    }

    const float bandA = fmaxf(bandScale * fmaxf(radA, 0.f), 1e-6f);
    const float bandB = fmaxf(bandScale * fmaxf(radB, 0.f), 1e-6f);
    const bool nearSeam =
        cylPeriodicNearSeam(posA_un, bandA, simParams) || cylPeriodicNearSeam(posB_un, bandB, simParams);
    if (!nearSeam) {
        return false;
    }

    const float currDist2 = cylPeriodicShiftDist2(posA_un, posB_un, simParams, ghostShiftA, ghostShiftB);
    if (currDist2 <= bestDist2) {
        return true;
    }

    const float bestDist = sqrtf(fmaxf(bestDist2, 0.f));
    const float currDist = sqrtf(fmaxf(currDist2, 0.f));
    const float absTol = fmaxf(absTolScale * fminf(fmaxf(radA, 0.f), fmaxf(radB, 0.f)), absTolFloor);
    const float relTol = relTolScale * (bestDist + 1e-6f);
    return (currDist - bestDist) <= fmaxf(absTol, relTol);
}

// Conservative near-seam keep rule for kT candidate acceptance.
__host__ __device__ inline bool cylPeriodicShouldKeepGhostShiftNearSeam(const float3& posA_un,
                                                                        float radA,
                                                                        const float3& posB_un,
                                                                        float radB,
                                                                        int ghostShiftA,
                                                                        int ghostShiftB,
                                                                        int desiredShiftA,
                                                                        int desiredShiftB,
                                                                        float bestDist2,
                                                                        const deme::DEMSimParams* simParams) {
    return cylPeriodicShouldKeepGhostShiftNearSeamImpl(posA_un, radA, posB_un, radB, ghostShiftA, ghostShiftB,
                                                       desiredShiftA, desiredShiftB, bestDist2,
                                                       0.35f, 0.05f, 0.02f, 2e-4f, simParams);
}

// dT branch decision is more tolerant than kT because kT/dT are asynchronous.
// This reduces seam-side skip/recreate churn while staying local to near-equivalent image branches.
__host__ __device__ inline bool cylPeriodicShouldKeepGhostShiftNearSeamDT(const float3& posA_un,
                                                                          float radA,
                                                                          const float3& posB_un,
                                                                          float radB,
                                                                          int ghostShiftA,
                                                                          int ghostShiftB,
                                                                          int desiredShiftA,
                                                                          int desiredShiftB,
                                                                          float bestDist2,
                                                                          const deme::DEMSimParams* simParams) {
    return cylPeriodicShouldKeepGhostShiftNearSeamImpl(posA_un, radA, posB_un, radB, ghostShiftA, ghostShiftB,
                                                       desiredShiftA, desiredShiftB, bestDist2,
                                                       0.45f, 0.08f, 0.03f, 2.5e-4f, simParams);
}

// Decide whether to keep a (possibly ghost-involved) contact pair.
// Canonical min-image is preferred, but near seam we may keep an almost-equivalent alternative image as well
// so dT can select a stable branch under kT/dT asynchrony.
__host__ __device__ inline bool cylPeriodicShouldUseGhostPair(const float3& posA,
                                                             float radA,
                                                             bool ghostA,
                                                             bool ghostA_neg,
                                                             deme::bodyID_t ownerA,
                                                             const float3& posB,
                                                             float radB,
                                                             bool ghostB,
                                                             bool ghostB_neg,
                                                             deme::bodyID_t ownerB,
                                                             const deme::DEMSimParams* simParams,
                                                             const unsigned int* ownerCylGhostActive = nullptr) {
    if (!simParams->useCylPeriodic || simParams->cylPeriodicSpan <= 0.f) {
        return true;
    }

    // Decode this candidate's periodic image shifts.
    const int shiftA = ghostA ? (ghostA_neg ? -1 : 1) : 0;
    const int shiftB = ghostB ? (ghostB_neg ? -1 : 1) : 0;

    // Never allow non-minimal ghost-ghost (±2 span) representations.
    if (shiftA != 0 && shiftB != 0) {
        return false;
    }

    // Recover base-wedge owner positions, then pick one canonical image for this pair.
    const float3 posA_base = (shiftA != 0) ? cylPeriodicShiftPosGlobal(posA, simParams, -shiftA) : posA;
    const float3 posB_base = (shiftB != 0) ? cylPeriodicShiftPosGlobal(posB, simParams, -shiftB) : posB;

    bool availA_pos = true, availA_neg = true, availB_pos = true, availB_neg = true;
    cylPeriodicGhostAvailability(posA_base, fmaxf(radA, 0.f), simParams, availA_pos, availA_neg);
    cylPeriodicGhostAvailability(posB_base, fmaxf(radB, 0.f), simParams, availB_pos, availB_neg);

    int desiredA = 0, desiredB = 0;
    float bestDist2 = 0.f;
    cylPeriodicSelectGhostShiftByDistAvail(posA_base, posB_base, simParams, availA_pos, availA_neg, availB_pos,
                                           availB_neg, desiredA, desiredB, bestDist2);
    // Hard availability guard: if this candidate uses an unavailable side, reject it.
    if ((shiftA > 0 && !availA_pos) || (shiftA < 0 && !availA_neg) || (shiftB > 0 && !availB_pos) ||
        (shiftB < 0 && !availB_neg)) {
        return false;
    }
    // Prefer a single canonical image in kT.
    if (shiftA == desiredA && shiftB == desiredB) {
        return true;
    }

    // kT should emit one canonical image per owner pair; dT handles async consumption.
    // Returning false for all non-canonical images prevents duplicate seam branches.
    (void)ownerA;
    (void)ownerB;
    (void)ownerCylGhostActive;
    return false;
}

__host__ __device__ inline void cylPeriodicShiftTrig(int shift,
                                                     const deme::DEMSimParams* simParams,
                                                     float& cos_theta,
                                                     float& sin_theta,
                                                     float& cos_half,
                                                     float& sin_half) {
    if (shift == 0) {
        cos_theta = 1.f;
        sin_theta = 0.f;
        cos_half = 1.f;
        sin_half = 0.f;
        return;
    }
    if (shift == 1) {
        cos_theta = simParams->cylPeriodicCosSpan;
        sin_theta = simParams->cylPeriodicSinSpan;
        cos_half = simParams->cylPeriodicCosHalfSpan;
        sin_half = simParams->cylPeriodicSinHalfSpan;
        return;
    }
    if (shift == -1) {
        cos_theta = simParams->cylPeriodicCosSpan;
        sin_theta = -simParams->cylPeriodicSinSpan;
        cos_half = simParams->cylPeriodicCosHalfSpan;
        sin_half = -simParams->cylPeriodicSinHalfSpan;
        return;
    }
    const float angle = shift * simParams->cylPeriodicSpan;
    sin_theta = sinf(angle);
    cos_theta = cosf(angle);
    const float half = 0.5f * angle;
    sin_half = sinf(half);
    cos_half = cosf(half);
}

__host__ __device__ inline bool cylPeriodicShouldUseGhost(const float3& posA,
                                                          bool ghostA,
                                                          bool ghostA_neg,
                                                          const float3& posB,
                                                          bool ghostB,
                                                          bool ghostB_neg,
                                                          const deme::DEMSimParams* simParams) {
    (void)posA;
    (void)ghostA_neg;
    (void)posB;
    (void)ghostB_neg;
    (void)simParams;
    // Allow both original and ghost representations so forces can be applied on both sides.
    // Reject ghost-ghost only when they are on the same side (same sign), which would double-count.
    if (ghostA && ghostB) {
        return ghostA_neg != ghostB_neg;
    }
    return true;
}

__host__ __device__ inline bool cylPeriodicPointInWedge(const float3& pos, const deme::DEMSimParams* simParams) {
    float angle = cylPeriodicAngle(pos, simParams->cylPeriodicOrigin, simParams->cylPeriodicU, simParams->cylPeriodicV);
    float rel = angle - simParams->cylPeriodicStart;
    if (rel < 0.f) {
        rel += 2.f * deme::PI;
    }
    return rel < simParams->cylPeriodicSpan;
}

// Make sure a T1 type triplet falls in a range, then output as T2 type
template <typename T1, typename T2>
__host__ __device__ T2 clampBetween3Comp(const T1& data, const T2& low, const T2& high) {
    T2 res;
    res.x = DEME_MIN(DEME_MAX(data.x, low.x), high.x);
    res.y = DEME_MIN(DEME_MAX(data.y, low.y), high.y);
    res.z = DEME_MIN(DEME_MAX(data.z, low.z), high.z);
    return res;
}

// Make sure a T1 type triplet falls in a range, then output as T2 type
template <typename T1, typename T2>
__host__ __device__ T2 clampBetween(const T1& data, const T2& low, const T2& high) {
    T2 res;
    res = DEME_MIN(DEME_MAX(data, low), high);
    return res;
}

// Chops a long ID (typically voxelID) into XYZ components
template <typename T1, typename T2>
__host__ __device__ void IDChopper(T1& X,
                                   T1& Y,
                                   T1& Z,
                                   const T2& ID,
                                   const unsigned char& nvXp2,
                                   const unsigned char& nvYp2) {
    X = ID & (((T1)1 << nvXp2) - 1);  // & operation here equals modulo
    Y = (ID >> nvXp2) & (((T1)1 << nvYp2) - 1);
    Z = (ID) >> (nvXp2 + nvYp2);
}

// Packs XYZ components back to a long ID (typically voxelID)
template <typename T1, typename T2>
__host__ __device__ void IDPacker(T1& ID,
                                  const T2& X,
                                  const T2& Y,
                                  const T2& Z,
                                  const unsigned char& nvXp2,
                                  const unsigned char& nvYp2) {
    ID = X;
    ID += Y << nvXp2;
    ID += Z << (nvXp2 + nvYp2);
}

// From a voxelID to (usually double-precision) xyz coordinate
template <typename T1, typename T2, typename T3>
__host__ __device__ void voxelIDToPosition(T1& X,
                                           T1& Y,
                                           T1& Z,
                                           const T2& ID,
                                           const T3& subPosX,
                                           const T3& subPosY,
                                           const T3& subPosZ,
                                           const unsigned char& nvXp2,
                                           const unsigned char& nvYp2,
                                           const T1& voxelSize,
                                           const T1& l) {
    T2 voxelIDX, voxelIDY, voxelIDZ;
    IDChopper<T2, T2>(voxelIDX, voxelIDY, voxelIDZ, ID, nvXp2, nvYp2);
    X = (T1)voxelIDX * voxelSize + (T1)subPosX * l;
    Y = (T1)voxelIDY * voxelSize + (T1)subPosY * l;
    Z = (T1)voxelIDZ * voxelSize + (T1)subPosZ * l;
}

// From xyz coordinate (usually double-precision) to voxelID
template <typename T1, typename T2, typename T3>
__host__ __device__ void positionToVoxelID(T1& ID,
                                           T2& subPosX,
                                           T2& subPosY,
                                           T2& subPosZ,
                                           const T3& X,
                                           const T3& Y,
                                           const T3& Z,
                                           const unsigned char& nvXp2,
                                           const unsigned char& nvYp2,
                                           const T3& voxelSize,
                                           const T3& l) {
    deme::voxelID_t voxelNumX = X / voxelSize;
    deme::voxelID_t voxelNumY = Y / voxelSize;
    deme::voxelID_t voxelNumZ = Z / voxelSize;
    subPosX = (X - (T3)voxelNumX * voxelSize) / l;
    subPosY = (Y - (T3)voxelNumY * voxelSize) / l;
    subPosZ = (Z - (T3)voxelNumZ * voxelSize) / l;

    ID = voxelNumX;
    ID += voxelNumY << nvXp2;
    ID += voxelNumZ << (nvXp2 + nvYp2);
}

template <typename T1, typename T2>
__host__ __device__ void
applyOriQToVector3(T1& X, T1& Y, T1& Z, const T2& Qw, const T2& Qx, const T2& Qy, const T2& Qz) {
    T1 oldX = X;
    T1 oldY = Y;
    T1 oldZ = Z;
    X = ((T2)2.0 * (Qw * Qw + Qx * Qx) - (T2)1.0) * oldX + ((T2)2.0 * (Qx * Qy - Qw * Qz)) * oldY +
        ((T2)2.0 * (Qx * Qz + Qw * Qy)) * oldZ;
    Y = ((T2)2.0 * (Qx * Qy + Qw * Qz)) * oldX + ((T2)2.0 * (Qw * Qw + Qy * Qy) - (T2)1.0) * oldY +
        ((T2)2.0 * (Qy * Qz - Qw * Qx)) * oldZ;
    Z = ((T2)2.0 * (Qx * Qz - Qw * Qy)) * oldX + ((T2)2.0 * (Qy * Qz + Qw * Qx)) * oldY +
        ((T2)2.0 * (Qw * Qw + Qz * Qz) - (T2)1.0) * oldZ;
}

template <typename T1, typename T2, typename T3>
__host__ __device__ void applyFrameTransformLocalToGlobal(T1& pos, const T2& vec, const T3& rot_Q) {
    applyOriQToVector3(pos.x, pos.y, pos.z, rot_Q.w, rot_Q.x, rot_Q.y, rot_Q.z);
    pos.x += vec.x;
    pos.y += vec.y;
    pos.z += vec.z;
}

template <typename T1>
__host__ __device__ T1 distSquared(const T1& x1, const T1& y1, const T1& z1, const T1& x2, const T1& y2, const T1& z2) {
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
}

// Get the maginitude of a 3-component vector
template <typename T1>
__host__ __device__ T1 magVector3(T1& x, T1& y, T1& z) {
    return sqrt(x * x + y * y + z * z);
}

// Normalize a 3-component vector
template <typename T1>
inline __host__ __device__ void normalizeVector3(T1& x, T1& y, T1& z) {
    T1 r2 = x * x + y * y + z * z;
    T1 invMag = rsqrtf(r2);
    invMag = invMag * (1.5f - 0.5f * r2 * invMag * invMag);  // 1x Newton for acurrency
    // TODO: Think about whether this is safe
    // if (magnitude < DEME_TINY_FLOAT) {
    //     printf("Caution!\n");
    // }
    x *= invMag;
    y *= invMag;
    z *= invMag;
}
// Normalize a 3-component vector safe (not used and maybe not needed)
template <typename T1>
inline __host__ __device__ void normalizeVector3safe(T1& x, T1& y, T1& z) {
    T1 r2 = x * x + y * y + z * z;
    const T1 tiny = 1e-30f;
    T1 safe_r2 = fmaxf(r2, tiny);
    T1 invMag = rsqrtf(safe_r2);
    invMag = invMag * (1.5f - 0.5f * safe_r2 * invMag * invMag);
    T1 m = r2 > tiny ? 1.0f : 0.0f;
    x = x * invMag * m;
    y = y * invMag * m;
    z = z * invMag * m;
}

// Calculate the centroid of a triangle
template <typename T1>
inline __host__ __device__ T1 triangleCentroid(const T1& p1, const T1& p2, const T1& p3) {
    return (p1 + p2 + p3) * deme::ONE_OVER_THREE;
}

// Calculate the incenter of a triangle
template <typename T1>
__host__ __device__ T1 triangleIncenter(const T1& p1, const T1& p2, const T1& p3) {
    float a = length(p2 - p3);
    float b = length(p1 - p3);
    float c = length(p1 - p2);
    T1 res;
    res.x = (a * p1.x + b * p2.x + c * p3.x) / (a + b + c);
    res.y = (a * p1.y + b * p2.y + c * p3.y) / (a + b + c);
    res.z = (a * p1.z + b * p2.z + c * p3.z) / (a + b + c);
    return res;
}

// Hamilton product of 2 quaternions
template <typename T1, typename T2, typename T3>
__host__ __device__ void HamiltonProduct(T1& A,
                                         T1& B,
                                         T1& C,
                                         T1& D,
                                         const T2 a1,
                                         const T2 b1,
                                         const T2 c1,
                                         const T2 d1,
                                         const T3 a2,
                                         const T3 b2,
                                         const T3 c2,
                                         const T3 d2) {
    A = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2;
    B = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2;
    C = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2;
    D = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2;
}

// Compute the binID for a point in space (mixed precision, fp32 with fp64 fallback)
template <typename T1>
inline __host__ __device__ T1 getPointBinID(const double& X,
                                            const double& Y,
                                            const double& Z,
                                            const double& inv_binSize,
                                            const T1& nbX,
                                            const T1& nbY) {
    T1 binIDX = X * inv_binSize;
    T1 binIDY = Y * inv_binSize;
    T1 binIDZ = Z * inv_binSize;
    return binIDX + binIDY * nbX + binIDZ * nbX * nbY;
}

// Compute the binID using its indices in X, Y and Z directions
template <typename T1>
__host__ __device__ T1
binIDFrom3Indices(const T1& X, const T1& Y, const T1& Z, const T1& nbX, const T1& nbY, const T1& nbZ) {
    if ((X < nbX) && (Y < nbY) && (Z < nbZ)) {
        return X + Y * nbX + Z * nbX * nbY;
    } else {
        return deme::NULL_BINID;
    }
}

// Do we have an edge case where higher precision is required?--> Yes/No
inline __host__ __device__ bool near_edge(float b) {
    const float k = floorf(b);
    const float d = fminf(b - k, (k + 1.0f) - b);
    const float tau = 8.0f * fmaxf(fabsf(b) * 1.1920929e-7f, 1.1920929e-7f);
    return d <= tau;
}
// Computes the inclusive [imin, imax] bin range (clamped to [0, nb-1])
// along one axis for a sphere at position p with radius r,
// using a fast FP32 path with near-edge detection and a precise FP64 fallback.
inline __host__ __device__ deme::AxisBounds axis_bounds(double p, double r, int nb, double invBinSize) {
    // FP32 fast path
    const float invBinF = (float)invBinSize;
    const float pF = (float)p, rF = (float)r;
    const float bPlusF = fmaf(rF, invBinF, pF * invBinF);  // (p+r)/B
    const float bMinusF = (pF - rF) * invBinF;             // (p-r)/B
    if (!(near_edge(bPlusF) || near_edge(bMinusF))) {
        int imax = (int)fminf(bPlusF, (float)(nb - 1));
        int imin = (int)fmaxf(bMinusF, 0.0f);
        if (imax < imin)
            return {0, -1};
        return {imin, imax};
    }
    // Double fallback for edge cases
    const double bPlusD = fma(r, invBinSize, p * invBinSize);
    const double bMinusD = (p - r) * invBinSize;
    const int imax = (int)fmin(bPlusD, (double)(nb - 1));
    const int imin = (int)fmax(bMinusD, 0.0);
    if (imax < imin)
        return {0, -1};
    return {imin, imax};
}

// Is there a contact? And which binID does the contact point belong to?
// Returns contact type, overlapDepth and binID.
inline __host__ __device__ deme::contact_t checkSphereContactOverlapAndBin(const double& XA,
                                                                           const double& YA,
                                                                           const double& ZA,
                                                                           const double& radA,
                                                                           const double& XB,
                                                                           const double& YB,
                                                                           const double& ZB,
                                                                           const double& radB,
                                                                           const double& invBinSize,
                                                                           const deme::binID_t& nbX,
                                                                           const deme::binID_t& nbY,
                                                                           double& overlapDepth,
                                                                           deme::binID_t& binID) {
    // ---- FP32 fast path for distance test and contact point ----
    const float xA = (float)XA, yA = (float)YA, zA = (float)ZA;
    const float xB = (float)XB, yB = (float)YB, zB = (float)ZB;
    const float dx = xA - xB;
    const float dy = yA - yB;
    const float dz = zA - zB;
    // Distance^2 with FMAs (fast, stable in fp32)
    const float dist2f = fmaf(dz, dz, fmaf(dy, dy, dx * dx));
    const float rsumf = (float)radA + (float)radB;
    const float r2f = rsumf * rsumf;
    {
        if (dist2f > r2f) {
            overlapDepth = 0.0;
            binID = (deme::binID_t)0;
            return deme::NOT_A_CONTACT;  // early exit! (safe, since we have a margin)
        }
    }
    // Contact! - in fp32: compute normal and contact point
    float nx, ny, nz, distf = 0.0f;
    if (dist2f > 0.0f) {
        distf = sqrtf(dist2f);
        const float invd = 1.0f / distf;
        nx = dx * invd;
        ny = dy * invd;
        nz = dz * invd;
    } else {  // Should not occur....
        nx = 1.0f;
        ny = 0.0f;
        nz = 0.0f;
    }  // now contact point
    const float overlapF = rsumf - distf;
    const float stepF = (float)radB - 0.5f * overlapF;
    const float CPXf = fmaf(stepF, nx, xB);
    const float CPYf = fmaf(stepF, ny, yB);
    const float CPZf = fmaf(stepF, nz, zB);
    // Map contact point to binID using fp32, but guard against bin-edge cases.
    const float invF = (float)invBinSize;
    const float bxF = CPXf * invF;
    const float byF = CPYf * invF;
    const float bzF = CPZf * invF;
    // If not near any integer boundary in fp32, we can safely truncate in fp32.
    if (!(near_edge(bxF) || near_edge(byF) || near_edge(bzF))) {
        const deme::binID_t ix = (deme::binID_t)bxF;  // truncate toward zero (matches original)
        const deme::binID_t iy = (deme::binID_t)byF;
        const deme::binID_t iz = (deme::binID_t)bzF;
        binID = ix + iy * nbX + iz * nbX * nbY;
        overlapDepth = (double)overlapF;
        return deme::SPHERE_SPHERE_CONTACT;
    }
    // ---- FP64 fallback ONLY when the fp32 CP is near any bin boundary ----
    // overlap, and contact point in double for a robust bin map.
    {
        const double centerDist2 = distSquared<double>(XA, YA, ZA, XB, YB, ZB);
        overlapDepth = radA + radB - sqrt(centerDist2);
        const double CPXd = XB + (radB - overlapDepth * 0.5) * nx;
        const double CPYd = YB + (radB - overlapDepth * 0.5) * ny;
        const double CPZd = ZB + (radB - overlapDepth * 0.5) * nz;
        // Bin mapping in full double, using truncate-to-zero cast (compat with original)
        const deme::binID_t ix = (deme::binID_t)(CPXd * invBinSize);
        const deme::binID_t iy = (deme::binID_t)(CPYd * invBinSize);
        const deme::binID_t iz = (deme::binID_t)(CPZd * invBinSize);
        binID = ix + iy * nbX + iz * nbX * nbY;
        return deme::SPHERE_SPHERE_CONTACT;
    }
}

// This utility function returns the normal to the triangular face defined by
// the vertices A, B, and C. The face is assumed to be non-degenerate.
// Note that order of vertices is important!
template <typename T1>
__host__ __device__ T1 face_normal(const T1& A, const T1& B, const T1& C) {
    return normalize(cross(B - A, C - A));
}

// Binary search on GPU, which is probably quite divergent... use if absolutely needs to
// T2 must be a singed type
template <typename T1, typename T2>
__host__ __device__ bool cuda_binary_search(T1* A, const T1& val, T2 imin, T2 imax, T2& res) {
    while (imax >= imin) {
        T2 imid = imin + (imax - imin) / 2;
        if (val == A[imid]) {
            res = imid;
            return true;
        } else if (val > A[imid]) {
            imin = imid + 1;
        } else {
            imax = imid - 1;
        }
    }

    return false;
}
template <typename T1, typename T2>
__host__ __device__ bool cuda_binary_search(T1* A, const T1& val, T2 imin, T2 imax) {
    while (imax >= imin) {
        T2 imid = imin + (imax - imin) / 2;
        if (val == A[imid]) {
            return true;
        } else if (val > A[imid]) {
            imin = imid + 1;
        } else {
            imax = imid - 1;
        }
    }

    return false;
}

/**
 * Template arguments:
 *   - T1: the floating point accuracy level for the point coordinates
 *
 * Basic idea: calculate a vector that goes from B to A, and pack it as a float3
 *
 */
template <typename T1>
__host__ __device__ float3
vectorAB(const T1& AX, const T1& AY, const T1& AZ, const T1& BX, const T1& BY, const T1& BZ) {
    return make_float3(AX - BX, AY - BY, AZ - BZ);
}

/**
 * Template arguments:
 *   - T1: the floating point accuracy level for the point coordinate and the frame O coordinate
 *
 * Basic idea: calculate a point's (typically contact point) local coordinate in a specific frame, then return as a
 * float3
 *
 */
template <typename T1>
__host__ __device__ float3 findLocalCoord(const T1& X,
                                          const T1& Y,
                                          const T1& Z,
                                          const T1& Ox,
                                          const T1& Oy,
                                          const T1& Oz,
                                          const deme::oriQ_t& oriQw,
                                          const deme::oriQ_t& oriQx,
                                          const deme::oriQ_t& oriQy,
                                          const deme::oriQ_t& oriQz) {
    float locX, locY, locZ;
    locX = X - Ox;
    locY = Y - Oy;
    locZ = Z - Oz;
    // To find the contact point in the local (body) frame, just apply inverse quaternion to OP vector in global frame
    applyOriQToVector3<float, deme::oriQ_t>(locX, locY, locZ, oriQw, -oriQx, -oriQy, -oriQz);
    return make_float3(locX, locY, locZ);
}

/// Calculate the contact params based on the 2 contact material types given
template <typename T1>
__host__ __device__ void matProxy2ContactParam(T1& E_eff,
                                               T1& G_eff,
                                               const T1& Y1,
                                               const T1& nu1,
                                               const T1& Y2,
                                               const T1& nu2) {
    T1 invE = ((T1)1 - nu1 * nu1) / Y1 + ((T1)1 - nu2 * nu2) / Y2;
    E_eff = (T1)1 / invE;
    T1 invG = (T1)2 * ((T1)2 - nu1) * ((T1)1 + nu1) / Y1 + (T1)2 * ((T1)2 - nu2) * ((T1)1 + nu2) / Y2;
    G_eff = (T1)1 / invG;
}

/// Calculate the contact params based on the 2 contact material types given (no-tangent version)
template <typename T1>
__host__ __device__ void matProxy2ContactParam(T1& E_eff, const T1& Y1, const T1& nu1, const T1& Y2, const T1& nu2) {
    T1 invE = ((T1)1 - nu1 * nu1) / Y1 + ((T1)1 - nu2 * nu2) / Y2;
    E_eff = (T1)1 / invE;
}

// Singed distance from a point to a plane defined by a point and a normal vector
template <typename T1, typename T2>
__host__ __device__ T1 planeSignedDistance(const T2& point, const T2& planePoint, const float3& planeNormal) {
    // The distance is positive if the point is above the plane, and negative if below
    // The distance is 0 if the point is on the plane
    return (T1)(dot(point - planePoint, planeNormal));
}

// Radial distance vector from cylinder axis to a point
template <typename T1>
__host__ __device__ T1 cylRadialDistanceVec(const T1& point, const T1& cylPoint, const float3& cylDir) {
    // Projection of the point onto the cylinder axis
    float proj_dist = dot(point - cylPoint, cylDir);
    // Radial vector from cylinder center to point
    T1 radial_vec = point - (cylPoint + to_real3<float3, T1>(proj_dist * cylDir));
    return radial_vec;
}

/// Takes in a triangle ID and figures out an SD AABB for broadphase use
inline __device__ void boundingBoxIntersectBin(deme::binID_t* L,
                                               deme::binID_t* U,
                                               const float3& vA,
                                               const float3& vB,
                                               const float3& vC,
                                               deme::DEMSimParams* simParams) {
    float3 min_pt;
    min_pt.x = DEME_MIN(vA.x, DEME_MIN(vB.x, vC.x));
    min_pt.y = DEME_MIN(vA.y, DEME_MIN(vB.y, vC.y));
    min_pt.z = DEME_MIN(vA.z, DEME_MIN(vB.z, vC.z));

    // Enlarge bounding box, so that no triangle lies right between 2 layers of bins
    min_pt -= (float)DEME_BIN_ENLARGE_RATIO_FOR_FACETS * (float)simParams->dyn.binSize;
    // A point on a mesh can be out of the simulation world. In this case, becasue we only need to detect their contact
    // with spheres, and spheres are all in the simulation world, so we just clamp out the bins that are outside the
    // simulation world.
    int3 min_bin =
        clampBetween3Comp<float3, int3>(min_pt * (float)simParams->dyn.inv_binSize, make_int3(0, 0, 0),
                                        make_int3(simParams->nbX - 1, simParams->nbY - 1, simParams->nbZ - 1));

    float3 max_pt;
    max_pt.x = DEME_MAX(vA.x, DEME_MAX(vB.x, vC.x));
    max_pt.y = DEME_MAX(vA.y, DEME_MAX(vB.y, vC.y));
    max_pt.z = DEME_MAX(vA.z, DEME_MAX(vB.z, vC.z));

    max_pt += (float)DEME_BIN_ENLARGE_RATIO_FOR_FACETS * (float)simParams->dyn.binSize;
    int3 max_bin =
        clampBetween3Comp<float3, int3>(max_pt * (float)simParams->dyn.inv_binSize, make_int3(0, 0, 0),
                                        make_int3(simParams->nbX - 1, simParams->nbY - 1, simParams->nbZ - 1));

    L[0] = min_bin.x;
    L[1] = min_bin.y;
    L[2] = min_bin.z;

    U[0] = max_bin.x;
    U[1] = max_bin.y;
    U[2] = max_bin.z;
}

// Helper function to convert double to float3 for storage
// Note: This conversion depends on platform endianness. On little-endian systems (standard for x86/x64 and NVIDIA
// GPUs), the least significant bytes of the double are stored in f[0] and most significant in f[1]. This is consistent
// across the same architecture, ensuring proper round-trip conversion when reading back the data in follow-up kernels.
// This assumes sizeof(double) == 2 * sizeof(float), which is verified by static_assert.
inline __device__ float3 doubleToFloat3Storage(double value) {
    static_assert(
        sizeof(double) == 2 * sizeof(float),
        "Double must be exactly twice the size of float on this system for this conversion to work correctly.");
    union {
        double d;
        float f[2];
    } converter;
    converter.d = value;
    return make_float3(converter.f[0], converter.f[1], 0.0f);
}

// Helper function to extract double from float3 storage
// This is the reverse operation of doubleToFloat3Storage
inline __device__ double float3StorageToDouble(const float3& storage) {
    static_assert(
        sizeof(double) == 2 * sizeof(float),
        "Double must be exactly twice the size of float on this system for this conversion to work correctly.");
    union {
        double d;
        float f[2];
    } converter;
    converter.f[0] = storage.x;
    converter.f[1] = storage.y;
    return converter.d;
}

#endif
