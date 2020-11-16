#!/bin/bash
#SBATCH --output output04.err                                                   # output log file
#SBATCH -e error04.err                                                   # error log file
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH -p collinslab                                     # request 1 gpu for this job
#SBATCH --exclude=dcc-collinslab-gpu-[03,02,01]
module load Anaconda3/3.5.2                                            # load conda to make sure use GPU version of tf
# add cuda and cudnn path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/apps/rhel7/cudnn/lib64:$LD_LIBRARY_PATH
# add my library path
export PYTHONPATH=$PYTHONPATH:/hpc/home/sr365/NIPS/idlm_Pytorch/
# execute my file
#python hyperswipe04.py
python predict.py
# python train.py
# python test.py
# python evaluate04.py
