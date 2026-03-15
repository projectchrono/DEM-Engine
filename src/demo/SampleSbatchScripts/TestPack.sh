#!/usr/bin/env bash
##SBATCH --partition=anything
#SBATCH --partition=research
#SBATCH --time=02-12:33:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=20000
#SBATCH --gres=gpu:gtx1080:2
##SBATCH --gres=gpu:a100:2
##SBATCH --gres=gpu:gtx1080:2

#SBATCH --output=RollUpIncline_mu=0.25.out

./src/demo/DEMdemo_TestPack

