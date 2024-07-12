#!/bin/bash
#SBATCH --job-name=base1
#SBATCH --account=project_2006419 
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=16000
#SBATCH --gres=gpu:v100:1

module load tykky
export PATH="/scratch/project_2006419/envs/base/base11/bin:$PATH"

srun python train_valid.py 