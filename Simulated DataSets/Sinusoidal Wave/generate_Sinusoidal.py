"""
This is the function (script) which generates the simulated data for doing inverse model comparison.
This function (script) generates sinusoidal waves of y as a function of input x
"""
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos

# Define some hyper-params
x_dimension = 2
y_dimension = 2
x_low = -1
x_high = 1
num_sample_dimension = 20
f = 5

def plotData(data_x, data_y, save_dir='generated_sinusoidal_scatter.png'):
    """
    Plot the scatter plot of the simulated sinusoidal wave
    :param data_x: The simulated data x
    :param data_y: The simulated data y
    :param save_dir: The save name of the plot
    :return: None
    """


if __name__ == '__main__':
    xx = []
    for i in range(x_dimension):
        xx.append(np.linspace(x_low, x_high, num=num_sample_dimension))         # append each linspace into the list
    x = np.meshgrid(*xx)                                # shape(x_dim, #point, #point, ...) of data points
    # Initialize the y
    y_shape = [num_sample_dimension for i in range(x_dimension + 1)]
    y_shape[0] = x_dimension
    data_y = np.zeros(y_shape)
    print(len(x))
    print('shape x', np.shape(x))
    print('shape y', np.shape(data_y))
    data_x = np.concatenate([np.reshape(a, [-1, 1]) for a in x], axis=1)
    data_y = np.reshape(data_y, [-1, 2])
    print('shape x', np.shape(data_x))
    print('shape y', np.shape(data_y))
    #print(data_x[:, 0])
    #print(np.shape(data_x[:, 0]))

    for i in range(np.shape(data_x)[-1]):
        data_y[:, 0] += sin(f*data_x[:, i])
        data_y[:, 1] += cos(f*data_x[:, i])
    # Reshape the data into one long list
    data_x = np.concatenate([np.reshape(a, [-1, 1]) for a in x], axis=1)
    data_y = np.reshape(data_y, [-1, 1])
    print(data_x)
    print(data_y)
    # Save the data into txt files
    np.savetxt('data_sin_x.csv', data_x, delimiter=',')
    np.savetxt('data_sin_y.csv', data_y, delimiter=',')

