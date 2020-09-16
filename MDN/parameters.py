"""
Params for Back propagation model
"""
# Define which data set you are using
# DATA_SET = 'meta_material'
# DATA_SET = 'sine_wave'
# DATA_SET = 'robotic_arm'
DATA_SET = 'ballistics'
TEST_RATIO = 0.2

# Model Architectural Params for meta_material data Set
NUM_GAUSSIAN = 20
LINEAR = [1,  1000, 1000, 1000, 1000, 1000, 1000, 1000, 4]

# Optimizer Params
OPTIM = "Adam"
REG_SCALE = 1e-3
BATCH_SIZE = 512
EVAL_BATCH_SIZE = 4096
EVAL_STEP = 20
TRAIN_STEP = 300
LEARN_RATE = 1e-3
# DECAY_STEP = 25000 # This is for step decay, however we are using dynamic decaying
LR_DECAY_RATE = 0.9
STOP_THRESHOLD = -float('inf')

# Data specific Params
X_RANGE = [i for i in range(2, 10 )]
Y_RANGE = [i for i in range(10 , 2011 )]
FORCE_RUN = True
MODEL_NAME = None
DATA_DIR = '../'
#DATA_DIR = '/work/sr365/Christian_data'
#DATA_DIR = 'D:\AML\idlm_Ben'
#CKPT_DIR = 'models/'
CKPT_DIR = '/work/sr365/MDN_results/' + DATA_SET
GEOBOUNDARY =[30, 52, 42, 52]
NORMALIZE_INPUT = True

# Running specific
USE_CPU_ONLY = False
EVAL_MODEL = '/work/sr365/MDN_results/ballistics/20200915_232515'
