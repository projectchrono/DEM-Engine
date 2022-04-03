#include <granular/DataStructs.h>
#include <granular/GranularDefines.h>

// Apply presecibed velocity and report whether the ``true'' physics should be skipped, rather than added on top of that
template <typename T1, typename T2>
inline __device__ void applyPrescribedVel(bool& LinPrescribed, bool& RotPrescribed, T1& hvX, T1& hvY, T1& hvZ, T2& hOmgBarX, T2& hOmgBarY, T2& hOmgBarZ, sgps::family_t family) {
    // TODO: implement real prescription cases via JITC
    switch (family) {
        case 0:
            LinPrescribed = false;
            RotPrescribed = false;
            break;
        case 1:
            hvX=0;
            hvY=0;
            hvZ=0;
            hOmgBarX=0;
            hOmgBarY=0;
            hOmgBarZ=0;
            LinPrescribed = true;
            RotPrescribed = true;
            break;
        default:
            LinPrescribed = false;
            RotPrescribed = false;
    }
}

// Apply presecibed location and report whether the ``true'' physics should be skipped, rather than added on top of that
template <typename T1, typename T2>
inline __device__ void applyPrescribedPos(bool& LinPrescribed, bool& RotPrescribed, T1& X, T1& Y, T1& Z, T2& ori0, T2& ori1, T2& ori2,  T2& ori3,sgps::family_t family) {
    // TODO: implement real prescription cases via JITC
    switch (family) {
        case 0:
            LinPrescribed = false;
            RotPrescribed = false;
            break;
        case 1:
            LinPrescribed = true;
            RotPrescribed = true;
            break;
        default:
            LinPrescribed = false;
            RotPrescribed = false;
    }
}

