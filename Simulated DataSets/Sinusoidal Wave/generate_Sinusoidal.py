"""
This is the function (script) which generates the simulated data for doing inverse model comparison.
This function (script) generates sinusoidal waves of y as a function of input x
"""
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos

# Define some hyper-params
x_dimension = 3
y_dimension = 2
x_low = -1
x_high = 1
num_sample_dimension = 20

def plotData(data_x, data_y, save_dir='generated_sinusoidal_scatter.png'):
    """
    Plot the scatter plot of the simulated sinusoidal wave
    :param data_x: The simulated data x
    :param data_y: The simulated data y
    :param save_dir: The save name of the plot
    :return: None
    """


if __name__ == '__main__':
    x = []
    for i in range(x_dimension):
        x.append()