#!/bin/bash
#SBATCH --partition=shared
#SBATCH --time=60:0:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive

time Rscript pca.R

