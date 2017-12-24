#!/bin/bash
#SBATCH -p hsw_v100
#SBATCH -t 1:00:00


#load intel tf


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
num_workers=$(( ${SLURM_NNODES} - ${num_ps} ))

#model
model='resnet152' #'vgg16' 'resnet50'

#run the stuff
echo $SLURM_NODELIST

for n in 0 0,1 0,1,2 0,1,2,3; do
    #enable GPUs
    export CUDA_VISIBLE_DEVICES=${n}

    echo "Running job on GPU(s) ${n}"
    num_gpus=$(echo ${n} | awk -F"," '{print NF}')

    #run the job
    srun -u -n ${SLURM_NNODES} python ../tf_cnn_benchmarks/tf_cnn_benchmarks.py --num_ps=${num_ps} --model=${model} --num_gpus=${num_gpus} --host_prefix="hsw" --num_host_digits=3 #--num_inter_threads=2 --num_intra_threads=32
done
