"""
The parameter file storing the parameters for VAE Model
"""

# Define which data set you are using
# DATA_SET = 'meta_material'
# DATA_SET = 'gaussian_mixture'
DATA_SET = 'sine_wave'
# DATA_SET = 'naval_propulsion'
# DATA_SET = 'robotic_arm'

# Architectural Params
DIM_Z = 5
DIM_X = 3
DIM_Y = 2
DIM_SPEC = None
LINEAR_D = [DIM_Y + DIM_Z, 50, 50, 50, 50, DIM_X]           # Linear units for Decoder
LINEAR_E = [DIM_X + DIM_Y, 50, 50, 50]                   # Linear units for Encoder
LINEAR_SE = []                      # Linear units for spectra encoder
CONV_OUT_CHANNEL_SE = []
CONV_KERNEL_SIZE_SE = []
CONV_STRIDE_SE = []

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
