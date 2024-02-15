#!/usr/bin/bash

# counterFolder=0

# for rolling_value in 0.00 0.01 0.02 0.04 0.08
# do
#     counterCase=0
#     for contact_friction in 0.00 0.01 0.025 0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90
#     do
#         echo "Iteration: $counterCase in folder $counterFolder"
#         echo "Parameters: mu=$contact_friction and Cr $rolling_value"
#         ../build/src/demo/DEMdemo_Granular_WoodenCylinderDrum $counterFolder $counterCase $contact_friction $rolling_value
#         ((counterCase++))
#     done
#     ((counterFolder++))
# done

make -C ../build/ -j 16

counterFolder=0
counterCase=1
for multiplierMass in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100 200 400 800 1600
do
    for contact_friction in 0.0001
    do
        tagFolder="4_$contact_friction"
        echo "Iteration: $counterCase in folder $tagFolder"
        echo "Parameters: mu=$contact_friction and Cr $rolling_value"
        echo "folder tag is:" 
        ../build/bin/DEMdemo_Force3D $tagFolder $counterCase $contact_friction $multiplierMass
    done
    ((counterCase++))
done

