#!/bin/bash
#SBATCH --job-name=stp2
#SBATCH --account=def-egranger
#SBATCH --time=01:30:00
#SBATCH --mem=50G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=h100_2g.20gb:1

set -euo pipefail
# Activate by environment name or full path
source "/home/banafsh7/anaconda3/etc/profile.d/conda.sh"
set +u  # Disable strict mode for unbound variables
conda activate VISTA
set -u 
# conda activate /home/your_user/path/to/myenv

echo "Python being used:"
which python
python --version



python step2_extract_features.py --dataset_name camelyon16 --data_h5_dir /scratch/banafsh7/CAM17/SASHA \
                        --index0 0 --indexn 1000 --data_slide_dir /scratch/banafsh7/CAM17/CAMELYON17/images --slide_ext .tif \
                        --csv_path dataset_csv/camelyon16/camelyon16.csv --feat_dir /scratch/banafsh7/CAM17/SASHAv --batch_size 32 \
                        --extract_high_res_features True --patch_level_low_res 3 --patch_level_high_res 1
