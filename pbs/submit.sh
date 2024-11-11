#!/bin/sh
# Below, is the queue
#PBS -q normal
#PBS -j oe
# Number of cores
#PBS -l select=1:ncpus=127:mem=500G
#PBS -l walltime=2:00:00
#PBS -M yang0886@e.ntu.edu.sg
#PBS -m abe
#PBS -N finML_job_2hrs
# Start of commands
cd $PBS_O_WORKDIR
echo thisss is the start of PBS script
