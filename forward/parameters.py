"""
Parameter file for specifying the running parameters for forward model
"""
# Model Architectural Parameters
USE_LORENTZ = True
LINEAR = [8, 50, 500, 1000, 1000, 1000, 500, 150]
CONV_OUT_CHANNEL = [4, 4, 4]
CONV_KERNEL_SIZE = [8, 5, 5]
CONV_STRIDE = [2, 1, 1]

# Optimization parameters
OPTIM = "Adam"
REG_SCALE = 5e-7
BATCH_SIZE = 128
EVAL_STEP = 1
TRAIN_STEP = 50
LEARN_RATE = 1e-1
# DECAY_STEP = 25000 # This is for step decay, however we are using dynamic decaying
LR_DECAY_RATE = 0.5
STOP_THRESHOLD = 1e-5

# Data Specific params
X_RANGE = [i for i in range(2, 10 )]
Y_RANGE = [i for i in range(10 , 2011 )]
FORCE_RUN = True
MODEL_NAME  = ''
DATA_DIR = '../'
GEOBOUNDARY =[30, 52, 42, 52]
NORMALIZE_INPUT = True

# Running specific
USE_CPU_ONLY = False
EVAL_MODEL = "20191202_161923"
NUM_COM_PLOT_TENSORBOARD = 10