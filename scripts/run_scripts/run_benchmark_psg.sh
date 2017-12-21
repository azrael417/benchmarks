#!/bin/bash
#SBATCH -p hsw_v100
#SBATCH -t 1:00:00


#load intel tf
module load tensorflow/intel-head

#openmp
export OMP_NUM_THREADS=32
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY="granularity=fine,compact,1,0"

#some other parameters
if [ $SLURM_NNODES -ge 2 ]; then
    num_ps=1
else
    num_ps=0
fi

#model
model='resnet50' #'vgg16'

#run the stuff
srun -u -n 1 python ../tf_cnn_benchmarks/tf_cnn_benchmarks.py --num_ps=${num_ps} --model=${model} #--num_inter_threads=2 --num_intra_threads=32
