"""
This is the function that generates the gaussian mixture for artificial model
The simulated cluster would be similar to the artifical data set from the INN Benchmarking paper
"""
# Define some
import numpy as np
import matplotlib.pyplot as plt
import math


# Define some constants
k = 0.9
g = 1
m = 0.5
num_samples = 20000

def determine_final_position(x, final_pos_return=False):
    """
    The function to determine the final y position of a ballistic movement which starts at (x1, x2) and throw at angle
    of x3 and initial velocity of x4, y is the horizontal position of the ball when hitting ground
    :param x: (N,4) numpy array
    :param final_pos_return: The flag to return the final position list for each point in each time steps, for debuggin
    purpose and should be turned off during generation
    :return: (N, 1) numpy array of y value
    """
    # Get the shape N
    N = np.shape(x)[0]

    # Initialize the output y
    output = np.zeros([N, 1])

    # Initialize the guess for time steps
    time_list = np.arange(0, 30, 0.0001)

    if final_pos_return:
        final_pos_list = []

    for i in range(N):
        # Separate the x into subsets for simplicity
        x1, x2, x3, x4 = x[i, 0], x[i, 1], x[i, 2], x[i, 3]

        final_pos = Position_at_time_T(x1, x2, x3, x4, time_list)

        # Final time step
        time = np.argmin(np.abs(final_pos[:, 1]))
        y = final_pos[time, 0]

        err = np.abs(final_pos[time, 1])
        assert err < 0.001, 'Your time solution is not accurate enough, current accuracy is {} at time step {}'.format(err, time)
        output[i] = y
        if final_pos_return:
            final_pos_list.append(final_pos)
    if final_pos_return:
        return output, final_pos_list
    else:
        return output


def Position_at_time_T(x1, x2, x3, x4, t):
    """
    infer the position of the trajectory at time x given input
    :param x1: x initial position, single number
    :param x2: y initial positio, single numbern
    :param x3: angle of thro, single numberw
    :param x4: velocity of thro, single numberw
    :param t: (N x 1) numpy array of time steps
    :return: (N X 2) numpy array of postions
    """
    # Get the initial velocity information
    v1 = x4 * np.cos(x3)
    v2 = x4 * np.sin(x3)
    N = np.shape(t)[0]      # get the shape of input
    output = np.zeros([N, 2])   # Initialize the output
    exponential_part = np.exp(-k*t/m) - 1
    output[:, 0] = x1 - v1 * m / k * exponential_part
    output[:, 1] = x2 - m / k / k * ((g*m + v2 * k)  * exponential_part + g*t*k)
    return output


def generate_random_x():
    """
    Generate random X array according to the description of the Benchmarking paper
    :return: (N, 4) random samples
    """
    output = np.zeros([num_samples, 4])
    output[:, 0] = np.random.normal(0, 0.25, size=num_samples)
    output[:, 1] = np.random.normal(1.5, 0.25, size=num_samples)
    output[:, 2] = np.radians(np.random.uniform(9, 72, size=num_samples))
    output[:, 3] = np.random.poisson(15, size=num_samples)
    return output


def plot_trajectory(x):
    """
    Plot the trajectory of a ballistic movement, in order to varify the correctness of this simulation
    :param x: (N, 4) the input x1~x4 param
    :return: one single plot of all the N trajectories
    """
    # Get the trajectory first
    y, final_postitions = determine_final_position(x, final_pos_return=True)
    f = plt.figure()
    plt.plot([0, 20], [0, 0],'r--', label="horitonal line")
    for i in range(len(y)):
        plt.plot(final_postitions[i][:, 0], final_postitions[i][:, 1], label=str(i))
    plt.legend()
    plt.ylim(bottom=0)
    plt.xlim([-1, 10])
    plt.title("trajectory plot for ballistic data set")
    plt.savefig('k={} m={} g={} Trajectory_plot.png'.format(k,m,g))


if __name__ == '__main__':
    X = generate_random_x()
    y = determine_final_position(X)
    # plot_trajectory(X)
    np.savetxt('data_x.csv', X, delimiter=',')
    np.savetxt('data_y.csv', y, delimiter=',')
