#!/usr/bin/env bash
##SBATCH --partition=anything
#SBATCH --partition=sbel
#SBATCH --time=06-12:33:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40000
#SBATCH --gres=gpu:rtx2080ti:2
##SBATCH --gres=gpu:a100:2
##SBATCH --gres=gpu:gtx1080:2

#SBATCH --output=WSS.out

./src/demo/DEMdemo_WheelSlopeSlip

