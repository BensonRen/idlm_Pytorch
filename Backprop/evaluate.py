"""
This file serves as a evaluation interface for the network
"""
# Built in
import os
# Torch

# Own
import flag_reader
from class_wrapper import Network
from model_maker import Backprop
import data_reader

# Libs
import numpy as np
import matplotlib.pyplot as plt


def compare_truth_pred(pred_file, truth_file):
    """
    Read truth and pred from csv files, compute their mean-absolute-error and the mean-squared-error
    :param pred_file: full path to pred file
    :param truth_file: full path to truth file
    :return: mae and mse
    """
    pred = np.loadtxt(pred_file, delimiter=' ')
    truth = np.loadtxt(truth_file, delimiter=' ')

    mae = np.mean(np.abs(pred-truth), axis=1)
    mse = np.mean(np.square(pred-truth), axis=1)

    return mae, mse


def plotMSELossDistrib(pred_file, truth_file, flags):
    mae, mse = compare_truth_pred(pred_file, truth_file)
    plt.figure(figsize=(12, 6))
    plt.hist(mse, bins=100)
    plt.xlabel('Mean Squared Error')
    plt.ylabel('cnt')
    plt.suptitle('Backprop (Avg MSE={:.4e})'.format(np.mean(mse)))
    plt.savefig(os.path.join(os.path.abspath(''), 'data',
                             'Backprop_{}.png'.format(flags.eval_model)))
    plt.show()
    print('Backprop (Avg MSE={:.4e})'.format(np.mean(mse)))

def evaluate_from_model(model_dir):
    """
    Evaluating interface. 1. Retreive the flags 2. get data 3. initialize network 4. eval
    :param model_dir: The folder to retrieve the model
    :return: None
    """
    # Retrieve the flag object
    print("Retrieving flag object for parameters")
    flags = flag_reader.load_flags(os.path.join("models", model_dir))
    flags.eval_model = model_dir                    # Reset the eval mode
    flags.batch_size = 1                            # For backprop eval mode, batchsize is always 1

    # Get the data
    train_loader, test_loader = data_reader.read_data(x_range=flags.x_range,
                                                      y_range=flags.y_range,
                                                      geoboundary=flags.geoboundary,
                                                      batch_size=flags.batch_size,
                                                      normalize_input=flags.normalize_input,
                                                      data_dir=flags.data_dir)
    print("Making network now")

    # Make Network
    ntwk = Network(Backprop, flags, train_loader, test_loader, inference_mode=True, saved_model=flags.eval_model)

    # Evaluation process
    print("Start eval now:")
    pred_file, truth_file = ntwk.evaluate()

    # Plot the MSE distribution
    plotMSELossDistrib(pred_file, truth_file, flags)
    print("Evaluation finished")



if __name__ == '__main__':
    # Read the flag, however only the flags.eval_model is used and others are not used
    useless_flags = flag_reader.read_flag()

    print(useless_flags.eval_model)
    # Call the evaluate function from model
    evaluate_from_model(useless_flags.eval_model)

