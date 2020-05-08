"""
This file serves as a prediction interface for the network
"""
# Built in
import os
import sys
sys.path.append('../utils/')
# Torch

# Own
import flag_reader
from class_wrapper import Network
from model_maker import Backprop
from utils import data_reader
from utils.helper_functions import load_flags
from utils.evaluation_helper import plotMSELossDistrib
# Libs
import numpy as np
import matplotlib.pyplot as plt


def predict_from_model(pre_trained_model, Xpred_file, shrink_factor=1, save_name=''):
    """
    Predicting interface. 1. Retreive the flags 2. get data 3. initialize network 4. eval
    :param model_dir: The folder to retrieve the model
    :return: None
    """
    # Retrieve the flag object
    print("This is doing the prediction for file", Xpred_file)
    print("Retrieving flag object for parameters")
    if (pre_trained_model.startswith("models")):
        eval_model = pre_trained_model[7:]
        print("after removing prefix models/, now model_dir is:", eval_model)
    
    flags = load_flags(pre_trained_model)                       # Get the pre-trained model
    flags.eval_model = eval_model                    # Reset the eval mode

    # Get the data, this part is useless in prediction but just for simplicity
    train_loader, test_loader = data_reader.read_data(flags)
    print("Making network now")

    # Make Network
    ntwk = Network(Backprop, flags, train_loader, test_loader, inference_mode=True, saved_model=flags.eval_model)
    print("number of trainable parameters is :")
    pytorch_total_params = sum(p.numel() for p in ntwk.model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    
    # Evaluation process
    print("Start eval now:")
    pred_file, truth_file = ntwk.predict(Xpred_file, save_prefix=save_name + 'shrink_factor' + str(shrink_factor), shrink_factor=shrink_factor)

    # Plot the MSE distribution
    #flags.eval_model = pred_file.replace('.','_') # To make the plot name different
    #plotMSELossDistrib(pred_file, truth_file, flags)
    #print("Evaluation finished")


def predict_all(models_dir="data"):
    """
    This function predict all the models in the models/. directory
    :return: None
    """
    for file in os.listdir(models_dir):
        if 'Xpred' in file and 'meta_material' in file:                     # Only meta material has this need currently
            print("predicting for file", file)
            predict_from_model("models/meta_materialreg0.0005trail_2_complexity_swipe_layer1000_num6", 
            os.path.join(models_dir,file))
    return None


if __name__ == '__main__':
    #shrink_list = np.arange(0, 1, 0.01)
    #reg_scale_list = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1, 5, 10]
    reg_scale_list = [20, 50, 100, 200, 1000]
    for reg_scale in reg_scale_list:
    #for sf in shrink_list:
        #predict_from_model('models/20200419_114504','data/range_3_full_Xpred.csv', shrink_factor=sf) 
        predict_from_model('models/sine_test_1dreg{}trail_0_complexity_swipe_layer500_num5'.format(reg_scale),'data/range_3_full_Xpred.csv', save_name='reg' + str(reg_scale)) 
    #predict_all('/work/sr365/multi_eval/Random/meta_material')
    #predict_from_model('models/20200419_114504','data/range_3_full_Xpred.csv', shrink_factor=0.5) 
