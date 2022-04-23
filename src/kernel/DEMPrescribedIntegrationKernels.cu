#include <granular/DataStructs.h>
#include <granular/GranularDefines.h>

// Apply presecibed velocity and report whether the ``true'' physics should be skipped, rather than added on top of that
template <typename T1, typename T2>
inline __device__ void applyPrescribedVel(bool& LinPrescribed,
                                          bool& RotPrescribed,
                                          T1& vX,
                                          T1& vY,
                                          T1& vZ,
                                          T2& omgBarX,
                                          T2& omgBarY,
                                          T2& omgBarZ,
                                          sgps::family_t family) {
    // TODO: implement real prescription cases via JITC
    switch (family) {
        case 0:
            LinPrescribed = false;
            RotPrescribed = false;
            break;
        case 1:
            vX = 0;
            vY = 0;
            vZ = 0;
            omgBarX = 0;
            omgBarY = 0;
            omgBarZ = 0;
            LinPrescribed = true;
            RotPrescribed = true;
            break;
        default:
            vX = 0;
            vY = 0;
            vZ = 0;
            omgBarX = 0;
            omgBarY = 0;
            omgBarZ = 0;
            LinPrescribed = true;
            RotPrescribed = true;
    }
}

// Apply presecibed location and report whether the ``true'' physics should be skipped, rather than added on top of that
template <typename T1, typename T2>
inline __device__ void applyPrescribedPos(bool& LinPrescribed,
                                          bool& RotPrescribed,
                                          T1& X,
                                          T1& Y,
                                          T1& Z,
                                          T2& ori0,
                                          T2& ori1,
                                          T2& ori2,
                                          T2& ori3,
                                          sgps::family_t family) {
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
            LinPrescribed = true;
            RotPrescribed = true;
    }
}
