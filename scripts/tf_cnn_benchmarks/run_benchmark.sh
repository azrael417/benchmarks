#!/bin/bash

#SBATCH -N 5
#SBATCH -A nstaff
#SBATCH -t 00:30:00
#SBATCH -C knl,quad,cache
#SBATCH -p regular

#load intel tf
module load tensorflow/intel-head

#openmp
export OMP_NUM_THREADS=64
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY="granularity=fine,compact,1,0"

#some other parameters
num_ps=1
model='resnet50' #'vgg16'

#run the stuff
srun -u -n 5 python tf_cnn_benchmarks.py --num_ps=${num_ps} --model=${model} --num_inter_threads=2 --num_intra_threads=32
