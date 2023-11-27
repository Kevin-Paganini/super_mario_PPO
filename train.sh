#!/bin/bash
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
command="python train.py"
singularity exec --nv /data/cs3450/pytorch20.11.3.sif bash -c '
    pip install rllib ray gymnasium dm_tree pyboy pysdl2 pysdl2-dll tensorflow-probability
    python train.py
'

