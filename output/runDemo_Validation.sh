#!/usr/bin/bash

make -C ../build/ -j 16
#  ../build/src/demo/DEMdemo_Granular_PlasticCylinderHopper
#  ../build/src/demo/DEMdemo_Granular_PlasticSphereHopper

#  ../build/src/demo/DEMdemo_Granular_WoodenSphereHopper

../build/src/demo/DEMdemo_PlasticCylinder_SphereHopper
../build/src/demo/DEMdemo_PlasticSphere_CylinderHopper
# ../build/src/demo/DEMdemo_Granular_PlasticCylinderDrum 1 1 0.50 0.05
# ../build/src/demo/DEMdemo_Granular_WoodenCylinderHopper