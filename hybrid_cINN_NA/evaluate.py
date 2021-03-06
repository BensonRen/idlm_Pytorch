"""
This file serves as a evaluation interface for the network
"""
# Built in
import os
# Torch

# Own
import cINN.flag_reader as flag_reader
from model_maker import make_cINN_and_NA
from class_wrapper import Network
from utils import data_reader
from utils import helper_functions
from utils.evaluation_helper import plotMSELossDistrib
from utils.evaluation_helper import get_test_ratio_helper

# Libs
import numpy as np
import matplotlib.pyplot as plt

def evaluate_from_model(model_dir, multi_flag=False, eval_data_all=False, test_ratio=None):
    """
    Evaluating interface. 1. Retreive the flags 2. get data 3. initialize network 4. eval
    :param model_dir: The folder to retrieve the model
    :param eval_data_all: The switch to turn on if you want to put all data in evaluation data
    :return: None
    """
    # Retrieve the flag object
    print("Retrieving flag object for parameters")
    if (model_dir.startswith("models")):
        model_dir = model_dir[7:]
        print("after removing prefix models/, now model_dir is:", model_dir)
    flags = helper_functions.load_flags(os.path.join("models", model_dir))
    flags.eval_model = model_dir                    # Reset the eval mode
    flags.batch_size = 1
    flags.backprop_step=50
    flags.eval_batch_size=2048

    if test_ratio is None:
        flags.test_ratio = get_test_ratio_helper(flags)
    else:
        # To make the test ratio swipe with respect to inference time
        # also making the batch size large enough
        flags.test_ratio = test_ratio
    # Get the data
    train_loader, test_loader = data_reader.read_data(flags, eval_data_all=eval_data_all)
    print("Making network now")

    # Make Network
    ntwk = Network(make_cINN_and_NA, flags, train_loader, test_loader, inference_mode=True, saved_model=flags.eval_model)
    #print(model_dir)
    print("number of trainable parameters is :")
    pytorch_total_params = sum(p.numel() for p in ntwk.model_cINN.parameters() if p.requires_grad)
    print(pytorch_total_params)
    pytorch_total_params = sum(p.numel() for p in ntwk.model_NA.parameters() if p.requires_grad)
    print(pytorch_total_params)
    # Evaluation process
    print("Start eval now:")
    if multi_flag:
        pred_file, truth_file = ntwk.evaluate(save_dir='/work/sr365/NIPS_multi_eval_backup/multi_eval/compare/hybrid_cINN_NA_no_BDY_50bp/'+flags.data_set, save_all=True)
    else:
        pred_file, truth_file = ntwk.evaluate()

    # Plot the MSE distribution
    if flags.data_set != 'meta_material' and not multi_flag: 
        plotMSELossDistrib(pred_file, truth_file, flags)
    print("Evaluation finished")
    
def evaluate_all(models_dir="models"):
    """
    This function evaluate all the models in the models/. directory
    :return: None
    """
    for file in os.listdir(models_dir):
        if os.path.isfile(os.path.join(models_dir, file, 'flags.obj')):
            evaluate_from_model(os.path.join(models_dir, file))
    return None


def evaluate_different_dataset(multi_flag, eval_data_all):
     """
     This function is to evaluate all different datasets in the model with one function call
     """
     data_set_list = ["robotic_arm","ballistics","sine_wave"]
     for eval_model in data_set_list:
        useless_flags = flag_reader.read_flag()
        useless_flags.eval_model = eval_model
        evaluate_from_model(useless_flags.eval_model, multi_flag=multi_flag, eval_data_all=eval_data_all)
    #evaluate_different_dataset(multi_flag=False, eval_data_all=True)

if __name__ == '__main__':
    # Read the flag, however only the flags.eval_model is used and others are not used
    useless_flags = flag_reader.read_flag()

    print(useless_flags.eval_model)
    # Call the evaluate function from model
    #evaluate_from_model(useless_flags.eval_model, multi_flag=True, eval_data_all=False)
    #evaluate_from_model(useless_flags.eval_model, multi_flag=False, eval_data_all=False)
    #for i in range(10,1000,10):
    #test_ratio = float(1/10000*200)
    evaluate_different_dataset(multi_flag=True, eval_data_all=False)
    #evaluate_all("models/")

