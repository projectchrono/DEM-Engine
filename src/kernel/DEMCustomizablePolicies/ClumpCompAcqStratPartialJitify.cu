deme::clumpComponentOffset_t myCompOffset = granData->clumpComponentOffset[sphereID];
if (myCompOffset != deme::RESERVED_CLUMP_COMPONENT_OFFSET) {
    myRelPos.x = CDRelPosX[myCompOffset];
    myRelPos.y = CDRelPosY[myCompOffset];
    myRelPos.z = CDRelPosZ[myCompOffset];
    myRadius = Radii[myCompOffset];
} else {
    // Look for my components in global memory
    deme::clumpComponentOffsetExt_t myCompOffsetExt = granData->clumpComponentOffsetExt[sphereID];
    myRelPos.x = granData->relPosSphereX[myCompOffsetExt];
    myRelPos.y = granData->relPosSphereY[myCompOffsetExt];
    myRelPos.z = granData->relPosSphereZ[myCompOffsetExt];
    myRadius = granData->radiiSphere[myCompOffsetExt];
}
