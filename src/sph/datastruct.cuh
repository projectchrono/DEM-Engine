// data structure utility class

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