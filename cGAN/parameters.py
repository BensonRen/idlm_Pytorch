"""
Hyper-parameters of the Tandem model
"""
# Model Architecture parameters
LINEAR_F = [8, 50, 500, 1000, 1000, 1000, 500, 150]
CONV_OUT_CHANNEL_F = [4, 4, 4]
CONV_KERNEL_SIZE_F = [8, 5, 5]
CONV_STRIDE_F = [2, 1, 1]

LINEAR_B = [150, 500, 500, 1000, 500, 100, 8]
CONV_OUT_CHANNEL_B = [4, 4, 1]
CONV_KERNEL_SIZE_B = [51, 35, 30]
CONV_STRIDE_B = [1, 1, 2]

# Optimizer parameters
OPTIM = "Adam"
REG_SCALE = 5e-5
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 4096
EVAL_STEP = 1
TRAIN_STEP = 10
VERB_STEP = 1
LEARN_RATE = 1e-2
# DECAY_STEP = 25000 # This is for step decay, however we are using dynamic decaying
LR_DECAY_RATE = 0
STOP_THRESHOLD = 1e-3

# Running specific parameter
USE_CPU_ONLY = False
DETAIL_TRAIN_LOSS_FORWARD = True
EVAL_MODEL = "20191204_211327"

# Data-specific parameters
X_RANGE = [i for i in range(2, 10 )]
Y_RANGE = [i for i in range(10 , 2011 )]
MODEL_NAME = ''
DATA_DIR = '../'
GEOBOUNDARY = [30, 52, 42, 52]
NORMALIZE_INPUT = True
