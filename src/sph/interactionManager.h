#ifndef SPH_INTERACTION_MANAGER
#define SPH_INTERACTION_MANAGER

#include "datastruct.cuh"

class InteractionManager {
  public:
    contactData* contact_pair;
    int contact_pair_n;

    int* offset;
    int offset_n;

    vector3* pos;
    int pos_n;
};

#endif