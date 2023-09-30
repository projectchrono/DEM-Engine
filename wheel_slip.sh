#!/usr/bin/env bash
#SBATCH --partition=sbel
#SBATCH --time=10-20:33:00
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40000
##SBATCH --gres=gpu:rtx2080ti:2
#SBATCH --gres=gpu:a100:1
##SBATCH --gres=gpu:v100:2

#SBATCH --output=wheel_auto_explore_200.out

module load nvidia/cuda/12.0.0 gcc

generate_random_real_range() {
    sleep 0.01
    local min=$1
    local max=$2
    local seed_value=$(date +%s%N | cut -b1-13)
    local range=$(awk "BEGIN { srand($seed_value); print $max - $min }")
    local random_real=$(awk "BEGIN { srand($seed_value); print $min + rand() * $range }")
    echo "$random_real"
}

generate_random_integer_range() {
    sleep 0.01
    local min=$1
    local max=$2
    local random_integer=$(( min + RANDOM % (max - min + 1) ))
    echo "$random_integer"
}

echo "Run id is " $SLURM_JOB_ID
echo "=================================="

sim_num=1

for i in {1..1000}; do
    height=$(generate_random_real_range 0.005 0.25) # 0.08 0.15
    thickness=$(generate_random_real_range 0.01 0.025) #0.01
    ampl=$(generate_random_real_range 0.0 0.03) #0.0
    wave_num=$(generate_random_integer_range 1 4) #1
    cp_dev=$(generate_random_real_range -0.3 0.3)
    num_g=$(generate_random_integer_range 8 32)
    width=$(generate_random_real_range 0.1 0.5)
    rad=$(generate_random_real_range 0.1 0.5) 
    slope=$(generate_random_real_range 5.0 20.0)
    mass=$(generate_random_real_range 22.0 132.0)
    wr=$(generate_random_real_range 0.2 2.0)
    G=$(generate_random_real_range 1.0 10.0)
    mu=$(generate_random_real_range 0.1 1.2)

    # if awk "BEGIN {exit !($height >= 0.08 && $height <= 0.15)}"; then
    # if awk "BEGIN {exit !($height <= 0.15)}"; then
    #     if awk "BEGIN {exit !($rad >= 0.15)}"; then
    #         continue
    #     fi
    # fi

    echo "Amplitude: " $ampl
    echo "Wave number: " $wave_num
    echo "CP derivation: " $cp_dev
    echo "Height: " $height
    echo "Thickness: " $thickness
    echo "Grouser num: " $num_g
    echo "Wheel width: " $width
    echo "Wheel rad: " $rad
    echo "Slope: " $slope
    echo "Wheel mass: " $mass
    echo "Angular vel: " $wr
    echo "G: " $G
    echo "Friction: " $mu

    python3 -B gen_wheel.py -a $ampl -c $cp_dev -n $wave_num -t $thickness -g $height -d $num_g -r $rad -w $width -s 1 -f wheel_$SLURM_JOB_ID.obj >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        ./src/demo/DEMdemo_Meshed_Steering $sim_num $rad $mass $height $width $slope $cp_dev DEMdemo_Wheel_Steer_$SLURM_JOB_ID wheel_$SLURM_JOB_ID.obj $wr $G $mu
        if [ $? -ne 0 ]; then
            echo "Steer simulation did not succeed."
        else 
            ./src/demo/DEMdemo_Meshed_WheelDP_SlopeSlip $sim_num $rad $mass $height $width $slope $cp_dev DEMdemo_Wheel_Tests_$SLURM_JOB_ID wheel_$SLURM_JOB_ID.obj $wr $G $mu
            if [ $? -ne 0 ]; then
                echo "Slip simulation did not succeed."
            else 
                echo "Success: 1"
            fi
        fi
    else
        echo "Wheel generation did not succeed."
    fi
    echo "=================================="

    ((sim_num++))
done




