#!/bin/bash
#SBATCH --job-name=prep_tokens
#SBATCH --account=torch_pr_62_tandon_priority
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/am15577/slurm_prep_tokens_%j.out

cd "/scratch/am15577/projects/ML Music Gen"
python3 -u prepare_tokens.py
