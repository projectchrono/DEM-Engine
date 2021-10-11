//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#pragma once
#include <atomic>
#include <condition_variable>
#include <mutex>

// class holds on to statistics related to the scheduling process
class ManagerStatistics {
  public:
    std::atomic<unsigned int> nTimesDynamicHeldBack;
    std::atomic<unsigned int> nTimesKinematicHeldBack;
    std::atomic<unsigned int> nDynamicUpdates;
    std::atomic<unsigned int> nKinematicUpdates;

    ManagerStatistics() noexcept {
        nTimesDynamicHeldBack = 0;
        nTimesKinematicHeldBack = 0;
        nDynamicUpdates = 0;
        nKinematicUpdates = 0;
    }

    ~ManagerStatistics() {}
};

// class that will be used via an atomic object to coordinate the
// production-consumption interplay
class ThreadManager {
  public:
    std::atomic<int> stampLastUpdateOfDynamic;
    std::atomic<int> currentStampOfDynamic;
    std::atomic<int> dynamicRequestedUpdateFrequency;
    std::atomic<bool> dynamicDone;

    std::atomic<bool> dynamicOwned_Prod2ConsBuffer_isFresh;
    std::atomic<bool> kinematicOwned_Cons2ProdBuffer_isFresh;

    std::mutex dynamicOwnedBuffer_AccessCoordination;
    std::mutex kinematicOwnedBuffer_AccessCoordination;
    std::mutex kinematicCanProceed;
    std::mutex dynamicCanProceed;
    std::condition_variable cv_KinematicCanProceed;
    std::condition_variable cv_DynamicCanProceed;
    ManagerStatistics schedulingStats;
    
    // The following variables are used to ensure that when an instance of d or k thread is created, a while loop that spins in place is created. It does actual work only when we tell it all preparations are done and it can proceed to do the next AdvanceSimulation call.
    std::atomic<bool> dynamicStarted;
    std::atomic<bool> dynamicShouldJoin;
    std::atomic<bool> kinematicStarted;
    std::atomic<bool> kinematicShouldJoin;
    std::mutex dynamicStartLock;
    std::mutex kinematicStartLock;
    std::condition_variable cv_DynamicStartLock;
    std::condition_variable cv_KinematicStartLock;

    ThreadManager() noexcept {
        // that is, let dynamic advance into future as much as it wants
        dynamicRequestedUpdateFrequency = -1;
        stampLastUpdateOfDynamic = -1;
        currentStampOfDynamic = 0;
        dynamicDone = false;
        dynamicOwned_Prod2ConsBuffer_isFresh = false;
        kinematicOwned_Cons2ProdBuffer_isFresh = false;
    }

    ~ThreadManager() {}

    inline bool dynamicShouldWait() const {
        // do not hold dynamic back under the following circustances:
        // * the update frequency is negative, dynamic can drift into future
        // * the kinematic is done
        if (dynamicRequestedUpdateFrequency < 0)
            return false;

        // the dynamic should wait if it moved too far into the future
        bool shouldWait =
            (currentStampOfDynamic > stampLastUpdateOfDynamic + dynamicRequestedUpdateFrequency ? true : false);
        return shouldWait;
    }
};
