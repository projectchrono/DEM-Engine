// data structure utility class
#pragma once
// 3d vector data structure
struct vector3 {
  float x;
  float y;
  float z;
};

// 2d vector data structure
struct vector2 {
  float x;
  float y;
};

struct intVector3 {
  int x;
  int y;
  int z;
};

// 2d vector data structure
struct intVetor2 {
  int x;
  int y;
};

// contact pair/force structure
struct contactData{
  intVetor2 contact_pair;
  vector3 contact_force;
};

struct ExchangeData {
    contactData* contact_pair;
    int contact_pair_n;

    int* offset;
    int offset_n;

    vector3* pos;
    int pos_n;
};
