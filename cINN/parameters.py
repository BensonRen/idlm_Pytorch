"""
The parameter file storing the parameters for INN Model
"""

# Define which data set you are using
# DATA_SET = 'meta_material'
# DATA_SET = 'gaussian_mixture'
DATA_SET = 'sine_wave'
# DATA_SET = 'naval_propulsion'
# DATA_SET = 'robotic_arm'
# DATA_SET = 'ballistics'
TEST_RATIO = 0.2

# Architectural Params
DIM_Z = 3
DIM_X = 2
DIM_Y = 1
COUPLE_LAYER_NUM = 5
DIM_SPEC = None
# The below definitions are useless now since we are using the package
SUBNET_LINEAR = []                                          # Linear units for Subnet FC layer

##########################################################
# Originally for Spectra encoder, now for hybrid cINN+NA #
##########################################################
LINEAR = [2, 500, 5000, 5000, 500, 1]                      # Linear units for spectra encoder
CONV_OUT_CHANNEL = []
CONV_KERNEL_SIZE = []
CONV_STRIDE = []

# Loss ratio
LAMBDA_MSE = 3.             # The Loss factor of the MSE loss (reconstruction loss)
LAMBDA_Z = 300.             # The Loss factor of the latent dimension (converging to normal distribution)
LAMBDA_REV = 400.           # The Loss factor of the reverse transformation (let x converge to input distribution)
ZEROS_NOISE_SCALE = 5e-2          # The noise scale to add to
Y_NOISE_SCALE = 1e-1


# Optimization params
OPTIM = "Adam"
REG_SCALE = 2e-5
BATCH_SIZE = 1024
EVAL_BATCH_SIZE = 4096
EVAL_STEP = 20
GRAD_CLAMP = 15
TRAIN_STEP = 500
VERB_STEP = 20
LEARN_RATE = 1e-3
# DECAY_STEP = 25000 # This is for step decay, however we are using dynamic decaying
LR_DECAY_RATE = 0.5
STOP_THRESHOLD = -float('inf')

# Data specific params
X_RANGE = [i for i in range(2, 10 )]
#Y_RANGE = [i for i in range(10 , 2011 )]                       # Real Meta-material dataset range
Y_RANGE = [i for i in range(10 , 310 )]                         # Artificial Meta-material dataset
FORCE_RUN = True
MODEL_NAME  = None
# MODEL_NAME  = 'dim_z_2 + wBN + 100 + lr1e-3 + reg5e-3'
DATA_DIR = '../'                                               # All simulated simple dataset
#DATA_DIR = '/work/sr365/'                                      # real Meta-material dataset
#DATA_DIR = '/work/sr365/NN_based_MM_data/'                      # Artificial Meta-material dataset
GEOBOUNDARY =[30, 52, 42, 52]
NORMALIZE_INPUT = True

# Running specific params
USE_CPU_ONLY = False
#EVAL_MODEL = "ballistics_Ben_version"
EVAL_MODEL = "meta_material"
#EVAL_MODEL = "sine_wave"
#EVAL_MODEL = "ballistics_Jakob_version"
#EVAL_MODEL = "robotic_armcouple_layer_num6trail_0"
#EVAL_MODEL = "retrain_time_evalsine_wavecouple_layer_num10trail_0"
#EVAL_MODEL = "meta_materialcouple_layer_num14trail_1"
#EVAL_MODEL = "retrain_time_evalballistics_Jakob_version"
