"""
Hyper-parameters of the Tandem model
"""
# Model Architecture parameters
LINEAR_F = [8, 150, 150, 150, 150, 150]
CONV_OUT_CHANNEL_F = [4, 4, 4]
CONV_KERNEL_SIZE_F = [8, 5, 5]
CONV_STRIDE_F = [2, 1, 1]

LINEAR_B = [150, 150, 150, 8]
CONV_OUT_CHANNEL_B = [4, 4, 1]
CONV_KERNEL_SIZE_B = [5, 5, 8]
CONV_STRIDE_B = [1, 1, 2]

# Optimizer parameters
OPTIM = "Adam"
REG_SCALE = 1e-3
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 4096
EVAL_STEP = 10
TRAIN_STEP = 300
VERB_STEP = 10
LEARN_RATE = 1e-2
# DECAY_STEP = 25000 # This is for step decay, however we are using dynamic decaying
LR_DECAY_RATE = 0.5
STOP_THRESHOLD = 1e-4

# Running specific parameter
USE_CPU_ONLY = False
DETAIL_TRAIN_LOSS_FORWARD = True
EVAL_MODEL = "20191204_211327"

# Data-specific parameters
X_RANGE = [i for i in range(2, 10 )]
Y_RANGE = [i for i in range(10 , 2011 )]
MODEL_NAME = None

# DATA_DIR = '../'
DATA_DIR = '/work/sr365/'
# DATA_DIR = '/home/omar/PycharmProjects/github/idlm_Pytorch-master/forward/'
GEOBOUNDARY = [30, 52, 42, 52]
NORMALIZE_INPUT = True
