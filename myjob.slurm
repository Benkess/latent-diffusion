#!/bin/bash

#Submit this script with: sbatch myjob.slurm

#SBATCH --time=1:00:00   # job time limit
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --cpus-per-task=1   # number of CPU cores per task
#SBATCH --gres=gpu:1   # gpu devices per node
#SBATCH --partition=gpu   # partition
#SBATCH --mem=200G   # memory
#SBATCH -J "mlia_ldm"   # job name
#SBATCH --mail-user=mxk3rz@virginia.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=mlia   # allocation name


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
