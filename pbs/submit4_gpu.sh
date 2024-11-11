#!/bin/sh
# Below, is the queue
#PBS -q normal
#PBS -j oe
# Number of cores
#PBS -l select=1:ngpus=4
#PBS -l walltime=01:59:59
#PBS -M yang0886@e.ntu.edu.sg
#PBS -m abe
#PBS -N Bi444-Senti-EE6483
# Start of commands
cd $PBS_O_WORKDIR
echo thisss is the start of PBS script
