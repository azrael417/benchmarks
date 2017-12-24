#!/bin/bash
#SBATCH -N 1
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
if [ $SLURM_NNODES -ge 2 ]; then
    num_ps=1
else
    num_ps=0
fi
num_workers=$(( ${SLURM_NNODES} - ${num_ps} ))

#model selection
model='resnet152' #'vgg16'

#run the stuff
srun -u -n ${SLURM_NNODES} python tf_cnn_benchmarks.py --num_ps=${num_ps} --model=${model} --num_inter_threads=2 --num_intra_threads=32 --host_prefix="nid" --num_host_digits=5 #--data_dir='/global/project/projectdirs/dasrepo/amathuri/Datasets/ImagenetFromMahmoud/imagenet-data'
