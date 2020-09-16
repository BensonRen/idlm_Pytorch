"""
This file serves as a evaluation interface for the network
"""
# Built in
import os
# Torch

# Own
import flag_reader
from class_wrapper import Network
from model_maker import MDN
from utils import data_reader
from utils import helper_functions
from utils.evaluation_helper import plotMSELossDistrib

# Libs
import numpy as np
import matplotlib.pyplot as plt

def evaluate_from_model(model_dir, multi_flag=False, eval_data_all=False):
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
    if model_dir.startswith('/'): # It is a absolute path
        flags = helper_functions.load_flags(model_dir)
    else:
        flags = helper_functions.load_flags(os.path.join("models", model_dir))
    flags.eval_model = model_dir                    # Reset the eval mode

    # Set up the test_ratio
    if flags.data_set == 'ballistics':
        flags.test_ratio = 0.078                        # 12800 in total
    elif flags.data_set == 'sine_wave':
        flags.test_ratio = 0.1                        # 8000 in total
    elif flags.data_set == 'robotic_arm':
        flags.test_ratio = 0.1                          # 10000 in total
    else:
        flags.test_ratio = 0.0476                         # 20000 in total for Meta material
    # Get the data
    train_loader, test_loader = data_reader.read_data(flags, eval_data_all=eval_data_all)
    print("Making network now")

    # Make Network
    ntwk = Network(MDN, flags, train_loader, test_loader, inference_mode=True, saved_model=flags.eval_model)
    print(model_dir)
    print("number of trainable parameters is :")
    pytorch_total_params = sum(p.numel() for p in ntwk.model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    # Evaluation process
    print("Start eval now:")
    if multi_flag:
        ntwk.evaluate_multiple_time()
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
     #data_set_list = ["ballistics_Jakob_version"] 
     data_set_list = ["sine_wavecouple_layer_num8trail_0",
                      "robotic_armcouple_layer_num6trail_0"]
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
    #for i in range(3,15):
    #    evaluate_all('/work/sr365/MDN_results/robotic_arm/Gaussian_'+str(i))
    evaluate_from_model(useless_flags.eval_model)
    #evaluate_from_model(useless_flags.eval_model, multi_flag=True)
    #evaluate_from_model(useless_flags.eval_model, multi_flag=False, eval_data_all=True)
    #evaluate_different_dataset(multi_flag=False, eval_data_all=False)
    #evaluate_all("models/sine_wave")

