"""
This file serves as a evaluation interface for the network
"""
# Built in
import os
# Torch

# Own
import flag_reader
from class_wrapper import Network
from model_maker import cINN
from utils import data_reader
from utils import helper_functions
from Simulated_DataSets.Gaussian_Mixture import generate_Gaussian

# Libs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
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
    if (flags.data_set == 'gaussian_mixture'):
        # get the prediction and truth array
        pred = np.loadtxt(pred_file, delimiter=' ')
        truth = np.loadtxt(truth_file, delimiter=' ')
        # get confusion matrix
        cm = confusion_matrix(truth, pred)
        cm = cm / np.sum(cm)
        # Calculate the accuracy 
        accuracy = 0
        for i in range(len(cm)):
            accuracy += cm[i,i]
        print("confusion matrix is", cm)
        # Plotting the confusion heatmap
        f = plt.figure(figsize=[15,15])
        plt.title('accuracy = {}'.format(accuracy))
        sns.set(font_scale=1.4)
        sns.heatmap(cm, annot=True)
        eval_model_str = flags.eval_model.replace('/','_')
        f.savefig('data/{}.png'.format(eval_model_str),annot_kws={"size": 16})

    else:
        mae, mse = compare_truth_pred(pred_file, truth_file)
        plt.figure(figsize=(12, 6))
        plt.hist(mse, bins=100)
        plt.xlabel('Mean Squared Error')
        plt.ylabel('cnt')
        plt.suptitle('(Avg MSE={:.4e})'.format(np.mean(mse)))
        plt.savefig(os.path.join(os.path.abspath(''), 'data',
                             '{}.png'.format(flags.eval_model)))
        print('(Avg MSE={:.4e})'.format(np.mean(mse)))

def evaluate_from_model(model_dir):
    """
    Evaluating interface. 1. Retreive the flags 2. get data 3. initialize network 4. eval
    :param model_dir: The folder to retrieve the model
    :return: None
    """
    # Retrieve the flag object
    print("Retrieving flag object for parameters")
    if (model_dir.startswith("models")):
        model_dir = model_dir[7:]
        print("after removing prefix models/, now model_dir is:", model_dir)
    flags = helper_functions.load_flags(os.path.join("models", model_dir))
    flags.eval_model = model_dir                    # Reset the eval mode

    # Get the data
    train_loader, test_loader = data_reader.read_data(flags)
    print("Making network now")

    # Make Network
    ntwk = Network(cINN, flags, train_loader, test_loader, inference_mode=True, saved_model=flags.eval_model)

    # Evaluation process
    print("Start eval now:")
    pred_file, truth_file = ntwk.evaluate()

    # Plot the MSE distribution
    plotMSELossDistrib(pred_file, truth_file, flags)
    print("Evaluation finished")
    
    # If gaussian, plot the scatter plot
    if flags.data_set == 'gaussian_mixture':
        Xpred = helper_functions.get_Xpred(path='data/', name=flags.eval_model) 
        Ypred = helper_functions.get_Ypred(path='data/', name=flags.eval_model) 

        # Plot the points scatter
        generate_Gaussian.plotData(Xpred, Ypred, save_dir='data/' + flags.eval_model.replace('/','_') + 'generation plot.png', eval_mode=True)

def evaluate_all(models_dir="models"):
    """
    This function evaluate all the models in the models/. directory
    :return: None
    """
    for file in os.listdir(models_dir):
        if os.path.isfile(os.path.join(models_dir, file, 'flags.obj')):
            evaluate_from_model(os.path.join(models_dir, file))
    return None


if __name__ == '__main__':
    # Read the flag, however only the flags.eval_model is used and others are not used
    useless_flags = flag_reader.read_flag()

    print(useless_flags.eval_model)
    # Call the evaluate function from model
    evaluate_from_model(useless_flags.eval_model)

