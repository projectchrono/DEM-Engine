// data structure utility class
#pragma once
// 3d vector data structure
struct vector3 {
    float x;
    float y;
    float z;
    vector3(float x_in, float y_in, float z_in) {
        x = x_in;
        y = y_in;
        z = z_in;
    }
    vector3() {
        x = 0.f;
        y = 0.f;
        z = 0.f;
    }
};

// 2d vector data structure
struct vector2 {
    float x;
    float y;
    vector2(float x_in, float y_in) {
        x = x_in;
        y = y_in;
    }
    vector2() {
        x = 0.f;
        y = 0.f;
    }
};

struct intVector3 {
    int x;
    int y;
    int z;
    intVector3(int x_in, int y_in, int z_in) {
        x = x_in;
        y = y_in;
        z = z_in;
    }
    intVector3() {
        x = 0;
        y = 0;
        z = 0;
    }
};

// 2d vector data structure
struct intVector2 {
    int x;
    int y;
    intVector2(int x_in, int y_in) {
        x = x_in;
        y = y_in;
    }
    intVector2() {
        x = 0;
        y = 0;
    }
};

// contact pair/force structure
struct contactData {
    intVector2 contact_pair;
    vector3 contact_force;
    contactData() {
        contact_pair = intVector2(0, 0);
        contact_force = vector3(0.f, 0.f, 0.f);
    }
    contactData(intVector2 ij, vector3 frc) {
        contact_pair = ij;
        contact_force = frc;
    }
};