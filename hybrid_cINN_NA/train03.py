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
import cINN.flag_reader as flag_reader
from utils import data_reader
from class_wrapper import Network
from model_maker import make_cINN_and_NA
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
    ntwk = Network(make_cINN_and_NA, flags, train_loader, test_loader)

    # Training process
    print("Start training cINN now...")
    ntwk.train_cINN()
    print("Start training NA now..")
    ntwk.train_NA()

    # Do the house keeping, write the parameters and put into folder, also use pickle to save the flags obejct
    write_flags_and_BVE(flags, ntwk.best_validation_loss, ntwk.ckpt_dir)
    #put_param_into_folder(ntwk.ckpt_dir)


def retrain_different_dataset(ind=None):
     """
     This function is to evaluate all different datasets in the model with one function call
     """
     from utils.helper_functions import load_flags
     data_set_list = ['robotic_arm']
     #data_set_list = ['meta_material']
     for eval_model in data_set_list:
        flags = load_flags(os.path.join("models", eval_model))
        flags.model_name = "retrain_" + flags.model_name
        if ind is not None:
            flags.model_name += 'trail_{}'.format(ind)
        #flags.geoboundary = [-1,1,-1,1]
        #flags.data_dir = '/work/sr365/MM_ensemble/'
        flags.train_step = 500
        flags.test_ratio = 0.2
        training_from_flag(flags)


if __name__ == '__main__':
    # Read the parameters to be set
    flags = flag_reader.read_flag()
    print(flags)
    # Call the train from flag function
    #training_from_flag(flags)
    # Do the retraining for all the data set to get the training
    for i in range(10):
        retrain_different_dataset(ind=i)

