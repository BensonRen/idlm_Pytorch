"""
Parameter file for specifying the running parameters for forward model
"""
DATA_SET = 'meta_material'
# DATA_SET = 'gaussian_mixture'
# DATA_SET = 'sine_wave'
# DATA_SET = 'naval_propulsion'
# DATA_SET = 'robotic_arm'
# Model Architectural Parameters

USE_LORENTZ = False
USE_CONV = False                         # Whether use upconv layer when not using lorentz @Omar
LINEAR = [8, 300, 300, 300, 300, 300, 12]
FIX_W0 = False
# If the Lorentzian is Flase
CONV_OUT_CHANNEL = [4, 4, 4]
CONV_KERNEL_SIZE = [8, 5, 5]
CONV_STRIDE = [2, 1, 1]

# Optimization parameters
OPTIM = "Adam"
REG_SCALE = 5e-4
BATCH_SIZE = 128
EVAL_STEP = 5
TRAIN_STEP = 100
LEARN_RATE = 1e-1
# DECAY_STEP = 25000 # This is for step decay, however we are using dynamic decaying
LR_DECAY_RATE = 0.5
STOP_THRESHOLD = 1e-5

# Data Specific params
#X_RANGE = [i for i in range(2, 10 )]
#Y_RANGE = [i for i in range(10 , 2011 )]
X_RANGE = [i for i in range(0, 8)]
Y_RANGE = [i for i in range(8, 20)]
FORCE_RUN = True
#DATA_DIR = '/work/sr365/'      # For server usage
DATA_DIR = '../dataIn/pretraining/'                # For local useage
#DATA_DIR = '/home/omar/PycharmProjects/github/idlm_Pytorch-master/forward/'                # For Omar useage
GEOBOUNDARY =[30, 52, 42, 52]
NORMALIZE_INPUT = True
TEST_RATIO = 0.2

# Running specific
USE_CPU_ONLY = False
MODEL_NAME  = None 
EVAL_MODEL = "20191202_161923"
NUM_COM_PLOT_TENSORBOARD = 1
