#!/bin/bash
#SBATCH --job-name=CT_to_nifti_HN1_try_2
#SBATCH --partition=superhimem
#SBATCH --mem=256G
#SBATCH --cpus-per-task=30
#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH --output=%j-%x.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mogtabaawadatamohamed.alim@uhn.ca

USERNAME="t126036uhn"
source /cluster/home/$USERNAME/.bashrc
conda activate readii_env

cd /cluster/home/t126036uhn/code/Ct_RTSTRUCT_to_nifti/negative-control-to-nifti

python3 code/all_CT_RTSTRUCT_to_nifti.py --config_file "/cluster/home/t126036uhn/code/Ct_RTSTRUCT_to_nifti/negative-control-to-nifti/config_HN1.yaml"
