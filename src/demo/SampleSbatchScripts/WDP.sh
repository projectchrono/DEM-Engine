#!/usr/bin/env bash
##SBATCH --partition=anything
#SBATCH --partition=sbel
#SBATCH --time=01-12:33:00
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=20000
#SBATCH --gres=gpu:rtx2080ti:2
##SBATCH --gres=gpu:a100:2
##SBATCH --gres=gpu:gtx1080:2

#SBATCH --output=WDP_RTX2080.out

./src/demo/DEMdemo_WheelDPSimplified

