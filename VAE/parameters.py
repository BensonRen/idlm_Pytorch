"""
The parameter file storing the parameters for VAE Model
"""

# Define which data set you are using
# DATA_SET = 'meta_material'
# DATA_SET = 'gaussian_mixture'
# DATA_SET = 'sine_wave'
# DATA_SET = 'naval_propulsion'
# DATA_SET = 'robotic_arm'
DATA_SET = 'ballistics'

# Architectural Params
DIM_Z = 5 
DIM_X = 4 
DIM_Y = 1
DIM_SPEC = None
LINEAR_D = [DIM_Y + DIM_Z, 100, 100, 100,    DIM_X]           # Linear units for Decoder
LINEAR_E = [DIM_Y + DIM_X, 100, 100, 100,  2*DIM_Z]                   # Linear units for Encoder
LINEAR_SE = []                      # Linear units for spectra encoder
CONV_OUT_CHANNEL_SE = []
CONV_KERNEL_SIZE_SE = []
CONV_STRIDE_SE = []
#LINEAR_SE = [150, 500, 500, 500, 500, DIM_Y]                      # Linear units for spectra encoder
#CONV_OUT_CHANNEL_SE = [4, 4, 4]
#CONV_KERNEL_SIZE_SE = [5, 5, 8]
#CONV_STRIDE_SE = [1, 1, 2]

# Optimization params
KL_COEFF = 1
OPTIM = "Adam"
REG_SCALE = 5e-3
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 4096
EVAL_STEP = 20
TRAIN_STEP = 300
VERB_STEP = 1
LEARN_RATE = 1e-3
# DECAY_STEP = 25000 # This is for step decay, however we are using dynamic decaying
LR_DECAY_RATE = 0.9
STOP_THRESHOLD = 1e-4

# Data specific params
X_RANGE = [i for i in range(2, 10 )]
Y_RANGE = [i for i in range(10 , 2011 )]
FORCE_RUN = True
MODEL_NAME  = None
# MODEL_NAME  = 'dim_z_2 + wBN + 100 + lr1e-3 + reg5e-3'
#DATA_DIR = '/work/sr365/'      # For server usage
#DATA_DIR = '/home/omar/PycharmProjects/github/idlm_Pytorch-master/forward/'                # For Omar useage
DATA_DIR = '../'                # For local useage
GEOBOUNDARY =[30, 52, 42, 52]
NORMALIZE_INPUT = True

# Running specific params
USE_CPU_ONLY = False
EVAL_MODEL = "gaussian_mixturekl_coeff0.04lr0.01reg0.005"
