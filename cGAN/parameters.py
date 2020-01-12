"""
Hyper-parameters of the Tandem model
"""
# Model Architecture parameters

# Params for the Random Noise
DIM_Z = 20
DIM_SPEC = 20

# Params for Forward Model, conv is actually upconv module
LINEAR = [8, 50, 500, 500, 500, 150]
CONV_OUT_CHANNEL = [4, 4, 4]
CONV_KERNEL_SIZE = [8, 5, 5]
CONV_STRIDE = [2, 1, 1]

# Params for Spec_encoder Model
LINEAR_SE = [150, 100, 100, 100, DIM_SPEC]
CONV_OUT_CHANNEL_SE = [4, 4, 1]
CONV_KERNEL_SIZE_SE = [51, 35, 30]
CONV_STRIDE_SE = [1, 1, 2]

# Params for discriminator
LINEAR_D = [8 + DIM_SPEC, 50, 500, 500, 500, 150]

# Params for Generator
LINEAR_G = [DIM_SPEC + DIM_Z, 100, 500, 500, 500, 100, 8]

# Optimizer parameters
OPTIM = "Adam"
REG_SCALE = 5e-4
BATCH_SIZE = 128
<<<<<<< HEAD
EVAL_BATCH_SIZE = 128
EVAL_STEP = 5
TRAIN_STEP = 100
VERB_STEP = 10
=======
EVAL_BATCH_SIZE = 4096
EVAL_STEP = 1
TRAIN_STEP = 10
VERB_STEP = 1
>>>>>>> 7748195ddf4709013658bba920510f1b7d9f150f
LEARN_RATE = 1e-2
# DECAY_STEP = 25000 # This is for step decay, however we are using dynamic decaying
LR_DECAY_RATE = 0.5
STOP_THRESHOLD = 1e-5

# Running specific parameter
USE_CPU_ONLY = False
DETAIL_TRAIN_LOSS_FORWARD = True
EVAL_MODEL = "20191204_211327"

# Data-specific parameters
X_RANGE = [i for i in range(2, 10 )]
Y_RANGE = [i for i in range(10 , 2011 )]
MODEL_NAME = ''
DATA_DIR = '/work/sr365/'
GEOBOUNDARY = [30, 52, 42, 52]
NORMALIZE_INPUT = True
