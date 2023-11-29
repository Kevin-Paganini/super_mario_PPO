#!/bin/bash
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
command="python train.py"
singularity exec --nv /data/sdp/senior_design_llm/containers/llm-sd.sif bash -c '
    pip install rllib ray gymnasium dm_tree pyboy pysdl2 pysdl2-dll tensorflow-probability
    python train.py
'

