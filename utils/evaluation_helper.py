"""
This is the helper functions for evaluation purposes

"""
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

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
