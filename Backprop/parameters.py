"""
Params for Back propagation model
"""
# Model Architectural Params
USE_LORENTZ = False
LINEAR = [8,  500, 150]
CONV_OUT_CHANNEL = [4, 4, 4]
CONV_KERNEL_SIZE = [8, 5, 5]
CONV_STRIDE = [2, 1, 1]

# Optimizer Params
OPTIM = "Adam"
REG_SCALE = 1e-3
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 128
EVAL_STEP = 2
TRAIN_STEP = 300
VERB_STEP = 1
LEARN_RATE = 1e-2
# DECAY_STEP = 25000 # This is for step decay, however we are using dynamic decaying
LR_DECAY_RATE = 0.5
STOP_THRESHOLD = 1e-4

# Data specific Params
X_RANGE = [i for i in range(2, 10 )]
Y_RANGE = [i for i in range(10 , 2011 )]
FORCE_RUN = True
MODEL_NAME  = ''
DATA_DIR = '../'
GEOBOUNDARY =[30, 52, 42, 52]
NORMALIZE_INPUT = True

# Running specific
USE_CPU_ONLY = False
EVAL_MODEL = "20191204_211327"
