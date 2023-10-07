#!/usr/bin/bash

make -C ../build/ -j 12
../build/src/demo/DEMdemo_Granular_PlasticCylinderHopper
../build/src/demo/DEMdemo_Granular_PlasticSphereHopper
../build/src/demo/DEMdemo_Granular_WooderCylinderHopper
../build/src/demo/DEMdemo_Granular_WoodenSphereHopper

