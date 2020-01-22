"""
The parameter file storing the parameters for VAE Model
"""

# Architectural Params
DIM_Z = 20
DIM_SPEC = 15
LINEAR_D = [DIM_SPEC + DIM_Z, 50, 500, 500, 100, 8]           # Linear units for Decoder
LINEAR_E = [8 + DIM_SPEC, 50, 500, 500, 150]                   # Linear units for Encoder
LINEAR_SE = [150, 150, 150, 150, DIM_SPEC]                      # Linear units for spectra encoder
CONV_OUT_CHANNEL_SE = [4, 4, 4]
CONV_KERNEL_SIZE_SE = [5, 5, 8]
CONV_STRIDE_SE = [1, 1, 2]

# Optimization params
OPTIM = "Adam"
REG_SCALE = 5e-5
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 4096
EVAL_STEP = 2
TRAIN_STEP = 100
VERB_STEP = 1
LEARN_RATE = 1e-2
# DECAY_STEP = 25000 # This is for step decay, however we are using dynamic decaying
LR_DECAY_RATE = 0.5
STOP_THRESHOLD = 1e-4

# Data specific params
X_RANGE = [i for i in range(2, 10 )]
Y_RANGE = [i for i in range(10 , 2011 )]
FORCE_RUN = True
MODEL_NAME  = None
#DATA_DIR = '/work/sr365/'      # For server usage
#DATA_DIR = '/home/omar/PycharmProjects/github/idlm_Pytorch-master/forward/'                # For Omar useage
DATA_DIR = '../'                # For local useage
GEOBOUNDARY =[30, 52, 42, 52]
NORMALIZE_INPUT = True

# Running specific params
USE_CPU_ONLY = False
EVAL_MODEL = "20191204_211327"
