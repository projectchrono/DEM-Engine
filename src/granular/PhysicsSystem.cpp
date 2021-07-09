//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <core/ApiVersion.h>
#include <cstring>
#include <granular/GranularDefines.h>
#include <granular/PhysicsSystem.h>
#include <iostream>
#include <thread>

namespace sgps {

int kinematicThread::costlyProductionStep(int val) const {
    std::this_thread::sleep_for(std::chrono::milliseconds(kinematicAverageTime));
    return 2 * val + 1;
}

}  // namespace sgps
