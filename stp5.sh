#!/bin/bash
#SBATCH --job-name=stp3
#SBATCH --account=def-hadi87
#SBATCH --time=01:05:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=h100_3g.40gb:1
#step3,30G: C: 15', T: 20'
#set -euo pipefail d

# Replace this path with your actual Conda installation path
#source "/scratch/banafsh7/miniforge3/etc/profile.d/conda.sh"

# Activate by environment name or full path
#conda activate sasha
# conda activate /home/your_user/path/to/myenv
export CUBLAS_WORKSPACE_CONFIG=:4096:8
set -euo pipefail
# Activate by environment name or full path
source "/home/banafsh7/anaconda3/etc/profile.d/conda.sh"
set +u  # Disable strict mode for unbound variables
conda activate VISTA
set -u 
export LD_LIBRARY_PATH=$HOME/miniconda3/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/scratch/banafsh7/conda/VISTA/lib/libopenslide.so.1.0.0:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/scratch/banafsh7/conda/VISTA/lib:$LD_LIBRARY_PATH
sd=$1
python step5_tsu_training.py --dataset camelyon16 --config config/tcga_tsu_config.yml --seed ${sd} --arch hafed  --log_dir /scratch/banafsh7/SASHA_CAM17/stp5_${sd} \
                        --ckpt_path /scratch/banafsh7/SASHA_CAM17/stp3_${sd}/models/DEBUG/checkpoint-best.pt --level1_path /scratch/banafsh7/SASHA_CAM17/stp4_${sd} \
                        --level3_path /scratch/banafsh7/CAM17/SASHAv/lr/h5_files