#!/bin/sh
# Below, is the queue
#PBS -q normal
#PBS -j oe
# Number of cores
#PBS -l select=1:ncpus=127:mem=991G
#PBS -l walltime=11:58:00
#PBS -M yang0886@e.ntu.edu.sg
#PBS -m abe
#PBS -N finML_job
# Start of commands
cd $PBS_O_WORKDIR
echo thisssssssssssssssssssssssssssssssss is the start of PBS script

module load miniforge3
conda activate deep
cd ~/proj/finML

python linear_bayesian.py
python linear_lasso.py
python linear_ridge.py
python neural_MLP.py
python tree_RF.py
python XGBoost.py

python DeepLearning/tabresnet.py
