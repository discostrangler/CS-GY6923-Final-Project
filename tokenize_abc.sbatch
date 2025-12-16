#!/bin/bash
#SBATCH --job-name=tokenize
#SBATCH --account=torch_pr_62_tandon_priority
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/am15577/slurm_tokenize_%j.out

cd "/scratch/am15577/projects/ML Music Gen"
python3 -u build_tokenizer.py
