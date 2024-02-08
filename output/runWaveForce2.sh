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

# for multiplierMass in 1 2 3 4 5 10 20 30
 for multiplierMass in 6.0 # for friction 000
#for multiplierMass in 1.5 2 3 5 6 11 # for friction 010 and 020
do
    counterCase=0
    for contact_friction in 0.0
    do
        echo "Iteration: $counterCase in folder $counterFolder"
        echo "Parameters: mu=$contact_friction and Cr $rolling_value"
        ../build/bin/DEMdemo_Force3D $counterFolder $counterCase "4_000_" $contact_friction $multiplierMass
        ((counterCase++))
    done
    ((counterFolder++))
done
