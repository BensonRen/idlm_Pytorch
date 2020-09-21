"""
This file serves as a evaluation interface for the network
"""
# Built in
import os
import sys
sys.path.append('../utils/')
# Torch

# Own
import flag_reader
from class_wrapper import Network
from model_maker import Forward, Backward
from utils import data_reader
from utils.helper_functions import load_flags
from utils.evaluation_helper import plotMSELossDistrib

# Libs


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
    flags = load_flags(os.path.join("models", model_dir))
    flags.eval_model = model_dir                    # Reset the eval mode
    if flags.data_set == 'ballistics':
        flags.test_ratio = 0.078                        # 12800 in total
    elif flags.data_set == 'sine_wave':
        flags.test_ratio = 0.1                        # 8000 in total
    elif flags.data_set == 'robotic_arm':
        flags.test_ratio = 0.1                          # 10000 in total
    elif flags.data_set == 'meta_material':             # Test the eval part
        flags.test_ratio = 0.0476                         # 20000 in total for Meta material
        flags.geoboundary = [30, 55, 42, 52] 

    # Get the data
    train_loader, test_loader = data_reader.read_data(flags, eval_data_all=eval_data_all)
    print("Making network now")

    # Make Network
    ntwk = Network(Forward, Backward, flags, train_loader, test_loader, inference_mode=True, saved_model=flags.eval_model)
    print("number of trainable parameters is :")
    pytorch_total_params = sum(p.numel() for p in ntwk.model_f.parameters() if p.requires_grad) +\
                           sum(p.numel() for p in ntwk.model_b.parameters() if p.requires_grad)
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
     data_set_list = ["robotic_arm","sine_wave","ballistics"]
     for eval_model in data_set_list:
        useless_flags = flag_reader.read_flag()
        useless_flags.eval_model = eval_model
        evaluate_from_model(useless_flags.eval_model, multi_flag=multi_flag, eval_data_all=eval_data_all)

if __name__ == '__main__':
    # Read the flag, however only the flags.eval_model is used and others are not used
    useless_flags = flag_reader.read_flag()

    print(useless_flags.eval_model)
    # Call the evaluate function from model
    #evaluate_from_model(useless_flags.eval_model, multi_flag=False)
    #evaluate_from_model(useless_flags.eval_model, multi_flag=True)
    #evaluate_from_model(useless_flags.eval_model, multi_flag=False, eval_data_all=True)
    #evaluate_different_dataset(multi_flag=False, eval_data_all=False)
    #evaluate_from_model(useless_flags.eval_model)
    evaluate_all("models/ball_new_swipe")
