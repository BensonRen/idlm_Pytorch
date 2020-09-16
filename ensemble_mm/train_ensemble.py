"""
This file serves as a training interface for training the network
"""
# Built in
import glob
import os
import shutil
import sys
sys.path.append('../utils/')

# Torch

# Own
import flag_reader_ensemble
from utils import data_reader
from class_wrapper_ensemble import Network
from model_maker_ensemble import Backprop
from utils.helper_functions import put_param_into_folder, write_flags_and_BVE

def training_from_flag(flags):
    """
    Training interface. 1. Read data 2. initialize network 3. train network 4. record flags
    :param flag: The training flags read from command line or parameter.py
    :return: None
    """
    # Get the data
    train_loader, test_loader = data_reader.read_data(flags)
    print("Making network now")

    # Make Network
    ntwk = Network(Backprop, flags, train_loader, test_loader)

    # Training process
    print("Start training now...")
    ntwk.train()

    # Do the house keeping, write the parameters and put into folder, also use pickle to save the flags obejct
    write_flags_and_BVE(flags, ntwk.best_validation_loss, ntwk.ckpt_dir)
    #put_param_into_folder(ntwk.ckpt_dir)


def retrain_different_dataset():
     """
     This function is to evaluate all different datasets in the model with one function call
     """
     from utils.helper_functions import load_flags
     data_set_list = ["robotic_armreg0.0005trail_0_backward_complexity_swipe_layer500_num6",
                        "sine_wavereg0.005trail_1_complexity_swipe_layer1000_num8",
                        "ballisticsreg0.0005trail_0_complexity_swipe_layer500_num5",
                        "meta_materialreg0.0005trail_2_complexity_swipe_layer1000_num6"]
     for eval_model in data_set_list:
        flags = load_flags(os.path.join("models", eval_model))
        flags.model_name = "retrain_time_eval" + flags.model_name
        flags.train_step = 500
        flags.test_ratio = 0.2
        training_from_flag(flags)


if __name__ == '__main__':
    # Read the parameters to be set
    flags = flag_reader_ensemble.read_flag()

    # Call the train from flag function
    for i in range(3):
        training_from_flag(flags)
    print(type(flags))
    # Do the retraining for all the data set to get the training 
    #for i in range(10):
    #retrain_different_dataset()
