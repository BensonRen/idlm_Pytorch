"""
Hyper-parameters of the Tandem model
"""
# Define which data set you are using
# DATA_SET = 'meta_material'
DATA_SET = 'gaussian_mixture'
# DATA_SET = 'sine_wave'
# DATA_SET = 'naval_propulsion'
# DATA_SET = 'robotic_arm'

# Model Architecture parameters
# LOAD_FORWARD_CKPT_DIR = 'pre_trained_forward/'
LOAD_FORWARD_CKPT_DIR = None

#LOAD_FORWARD_CKPT_DIR = None
# LINEAR_F = [8, 150, 150, 150, 150, 150]
# CONV_OUT_CHANNEL_F = [4, 4, 4]
# CONV_KERNEL_SIZE_F = [8, 5, 5]
# CONV_STRIDE_F = [2, 1, 1]

# LINEAR_B = [150, 150, 150, 150, 150,  8]
# CONV_OUT_CHANNEL_B = [4, 4, 4]
# CONV_KERNEL_SIZE_B = [5, 5, 8]
# CONV_STRIDE_B = [1, 1, 2]

# Model Architectural Params for gaussian mixture dataset
LINEAR_F = [2, 60,60,60,60,60,60,60, 4]
CONV_OUT_CHANNEL_F = []
CONV_KERNEL_SIZE_F = []
CONV_STRIDE_F = []

LINEAR_B = [4, 10, 10, 10, 10, 2]
CONV_OUT_CHANNEL_B = []
CONV_KERNEL_SIZE_B = []
CONV_STRIDE_B = []

# Optimizer parameters
OPTIM = "Adam"
REG_SCALE = 1e-3
BATCH_SIZE = 1024
EVAL_BATCH_SIZE = 4096
EVAL_STEP = 10
TRAIN_STEP = 200
VERB_STEP = 10
LEARN_RATE = 1e-3
# DECAY_STEP = 25000 # This is for step decay, however we are using dynamic decaying
LR_DECAY_RATE = 0.5
STOP_THRESHOLD = 8e-4

# Running specific parameter
USE_CPU_ONLY = False
DETAIL_TRAIN_LOSS_FORWARD = True
EVAL_MODEL = "sine_wave"

# Data-specific parameters
X_RANGE = [i for i in range(2, 10 )]
Y_RANGE = [i for i in range(10 , 2011 )]
MODEL_NAME = None

# DATA_DIR = '../'
DATA_DIR = '../'
# DATA_DIR = '/work/sr365/'
# DATA_DIR = '/home/omar/PycharmProjects/github/idlm_Pytorch-master/forward/'
GEOBOUNDARY = [30, 52, 42, 52]
NORMALIZE_INPUT = True
