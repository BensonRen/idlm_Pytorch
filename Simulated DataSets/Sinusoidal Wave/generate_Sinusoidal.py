"""
This is the function (script) which generates the simulated data for doing inverse model comparison.
This function (script) generates sinusoidal waves of y as a function of input x
"""
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos
from mpl_toolkits.mplot3d import Axes3D

# Define some hyper-params
x_dimension = 3         # Current version only support 2 dimension due to visualization issue
y_dimension = 2         # Current version only support 2 dimension due to visualization issue
x_low = -1
x_high = 1
num_sample_dimension = 40
f = 2

def plotData(data_x, data_y, save_dir='generated_sinusoidal_scatter.png'):
    """
    Plot the scatter plot of the simulated sinusoidal wave
    :param data_x: The simulated data x
    :param data_y: The simulated data y
    :param save_dir: The save name of the plot
    :return: None
    """
    # Plot one graph for each dimension of y
    for i in range(len(data_y)):
        f = plt.figure()
        ax = Axes3D(f)
        plt.title('scattering plot for dimension {} for sinusoidal data'.format(i+1))
        print(np.shape(data_x[0, :]))
        ax.scatter(data_x[0, :], data_x[1, :], data_y[i, :], s=2)
        f.savefig('dimension_{}'.format(i+1) + save_dir)

if __name__ == '__main__':
    xx = []
    for i in range(x_dimension):
        xx.append(np.linspace(x_low, x_high, num=num_sample_dimension))         # append each linspace into the list
    x = np.array(np.meshgrid(*xx))                                # shape(x_dim, #point, #point, ...) of data points
    # Initialize the y
    y_shape = [num_sample_dimension for i in range(x_dimension + 1)]
    y_shape[0] = y_dimension
    data_y = np.zeros(y_shape)
    print(len(x))
    print('shape x', np.shape(x))
    print('shape y', np.shape(data_y))
    #data_x = np.concatenate([np.reshape(a, [-1, 1]) for a in x], axis=1)
    #data_y = np.reshape(data_y, [-1, 2])
    #print(data_x[:, 0])
    #print(np.shape(data_x[:, 0]))
    print("shape x[0,::]", np.shape(x[0, :]))
    for i in range(len(x)):
        data_y[0, :] += sin(f*np.pi*x[i, ::])
        data_y[1, :] += cos(f*np.pi*x[i, ::])
        # data_y[0, :] += x[i, ::]              # Easy case for validation of architecture
        # data_y[1, :] -= x[i, ::]              # Easy case for validation of architecture
    print('shape x', np.shape(x))
    print('shape y', np.shape(data_y))
    # Plot the data
    plotData(x, data_y)
    # Reshape the data into one long list
    data_x = np.concatenate([np.reshape(np.ravel(x[i, :]), [-1, 1] ) for i in range(x_dimension)], axis=1)
    data_y = np.reshape(np.ravel(data_y), [-1, y_dimension])
    # Save the data into txt files
    np.savetxt('data_x.csv', data_x, delimiter=',')
    np.savetxt('data_y.csv', data_y, delimiter=',')
