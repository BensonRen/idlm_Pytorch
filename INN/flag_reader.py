"""
This file serves to hold helper functions that is related to the "Flag" object which contains
all the parameters during training and inference
"""
# Built-in
import argparse
import pickle
import os
# Libs

# Own module
from parameters import *

# Torch

def read_flag():
    """
    This function is to write the read the flags from a parameter file and put them in formats
    :return: flags: a struct where all the input params are stored
    """
    parser = argparse.ArgumentParser()
    # INN Model Architectural Params
    parser.add_argument('--dim-z', type=int, default=DIM_Z, help='The dimension of the latent variable to supplement the dimension of spectra')
    parser.add_argument('--num-hidden-unit', type=int, default=NUM_HIDDEN_UNIT, help='the number of hidden unit in each coupling layers')
    parser.add_argument('--num-hidden-layers', type=int, default=NUM_HIDDEN_LAYERS, help='the number of layers in each coupling layers')
    parser.add_argument('--num-coupling-modules', type=int, default=NUM_COUPLING_MODULES, help='the number of coupling modules there are')

    # Auto Encoder Architectural Params
    parser.add_argument('--encoder-linear', type=list, default=LINEAR_ENCODER, help='The fc layers units for encoder part')
    parser.add_argument('--encoder-conv-out-channel', type=list, default=CONV_OUT_CHANNEL_ENCODER,
                        help='The output channel of your 1d conv for encoder model')
    parser.add_argument('--encoder-conv-kernel-size', type=list, default=CONV_KERNEL_SIZE_ENCODER,
                        help='The kernel size of your 1d conv for encoder model')
    parser.add_argument('--encoder-conv-stride', type=list, default=CONV_STRIDE_ENCODER,
                        help='The strides of your 1d conv fro encoder model')

    parser.add_argument('--decoder-linear', type=list, default=LINEAR_DECODER, help='The fc layers units for decoder part')
    parser.add_argument('--decoder-conv-out-channel', type=list, default=CONV_OUT_CHANNEL_DECODER,
                        help='The output channel of your 1d conv for decoder model')
    parser.add_argument('--decoder-conv-kernel-size', type=list, default=CONV_KERNEL_SIZE_DECODER,
                        help='The kernel size of your 1d conv for decoder model')
    parser.add_argument('--decoder-conv-stride', type=list, default=CONV_STRIDE_DECODER,
                        help='The strides of your 1d conv fro decoder model')

    # Optimizer Params
    parser.add_argument('--optim', default=OPTIM, type=str, help='the type of optimizer that you want to use')
    parser.add_argument('--reg-scale', type=float, default=REG_SCALE, help='#scale for regularization of dense layers')
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int, help='batch size (100)')
    parser.add_argument('--eval-batch-size', default=EVAL_BATCH_SIZE, type=int,
                        help='The Batch size for back propagation')
    parser.add_argument('--eval-step', default=EVAL_STEP, type=int, help='# steps between evaluations')
    parser.add_argument('--train-step', default=TRAIN_STEP, type=int, help='# steps to train on the dataSet')
    parser.add_argument('--verb-step', default=VERB_STEP, type=int, help='# steps to print and check best performance')
    parser.add_argument('--lr', default=LEARN_RATE, type=float, help='learning rate')
    parser.add_argument('--lr-decay-rate', default=LR_DECAY_RATE, type=float,
                        help='decay learn rate by multiplying this factor')
    parser.add_argument('--stop_threshold', default=STOP_THRESHOLD, type=float,
                        help='The threshold below which training should stop')
    #    parser.add_argument('--decay-step', default=DECAY_STEP, type=int,
    #                        help='decay learning rate at this number of steps')

    # Data Specific Params
    parser.add_argument('--x-range', type=list, default=X_RANGE, help='columns of input parameters')
    parser.add_argument('--y-range', type=list, default=Y_RANGE, help='columns of output parameters')
    parser.add_argument('--geoboundary', default=GEOBOUNDARY, type=tuple, help='the boundary of the geometric data')
    parser.add_argument('--model-name', default=MODEL_NAME, type=str, help='name of the model')
    parser.add_argument('--data-dir', default=DATA_DIR, type=str, help='data directory')
    parser.add_argument('--normalize-input', default=NORMALIZE_INPUT, type=bool,
                        help='whether we should normalize the input or not')

    # Running specific
    parser.add_argument('--eval-model', default=EVAL_MODEL, type=str, help='the folder name of the model that you want to evaluate')
    parser.add_argument('--use-cpu-only', type=bool, default=USE_CPU_ONLY, help='The boolean flag that indicate use CPU only')
    flags = parser.parse_args()  # This is for command line version of the code
    # flags = parser.parse_args(args = [])#This is for jupyter notebook version of the code
    # flagsVar = vars(flags)
    return flags


def save_flags(flags, save_file="flags.obj"):
    """
    This function serialize the flag object and save it for further retrieval during inference time
    :param flags: The flags object to save
    :param save_file: The place to save the file
    :return: None
    """
    with open(save_file,'wb') as f:          # Open the file
        pickle.dump(flags, f)               # Use Pickle to serialize the object


def load_flags(save_dir, save_file="flags.obj"):
    """
    This function inflate the pickled object to flags object for reuse, typically during evaluation (after training)
    :param save_dir: The place where the obj is located
    :param save_file: The file name of the file, usually flags.obj
    :return: flags
    """
    with open(os.path.join(save_dir, save_file), 'rb') as f:     # Open the file
        flags = pickle.load(f)                                  # Use pickle to inflate the obj back to RAM
    return flags

def write_flags_and_BVE(flags, best_validation_loss):
    """
    The function that is usually executed at the end of the training where the flags and the best validation loss are recorded
    They are put in the folder that called this function and save as "parameters.txt"
    This parameter.txt is also attached to the generated email
    :param flags: The flags struct containing all the parameters
    :param best_validation_loss: The best_validation_loss recorded in a training
    :return: None
    """
    #To avoid terrible looking shape of y_range
    yrange = flags.y_range
    # yrange_str = str(yrange[0]) + ' to ' + str(yrange[-1])
    yrange_str = [yrange[0], yrange[-1]]
    flags_dict = vars(flags)
    flags_dict_copy = flags_dict.copy()                 # in order to not corrupt the original data strucutre
    flags_dict_copy['y_range'] = yrange_str             # Change the y range to be acceptable long string
    flags_dict_copy['best_validation_loss'] = best_validation_loss #Append the bvl
    # Convert the dictionary into pandas data frame which is easier to handle with and write read
    print(flags_dict_copy)
    with open('parameters.txt','w') as f:
        print(flags_dict_copy, file = f )
    # Pickle the obj
    save_flags(flags)

