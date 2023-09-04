#!/usr/bin/bash

counter=0
for rolling_value in [0.00 0.01 0.02 0.04 0.08]; do
    ((counter++))
    echo "Iteration: $counter"
    ../build/src/demo/DEMdemo_Granular_PlasticSphereDrum $counter $rolling_value
    
done
