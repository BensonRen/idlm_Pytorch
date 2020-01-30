"""
This is the function (script) which generates the simulated data for doing inverse model comparison.
This function (script) generates robotic arms positions of y as a function of input x
"""

# Import libraries
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from os.path import join
from numpy import sin, cos

# Define some hyper-parameters
arm_num = 3
arm_lengths = [1, 1, 2]


def determine_final_position(origin_pos, arm_angles, arm_lengths=arm_lengths):
    """
    The function that computes the final position of the angle. One number input implementation now
    :param origin_pos: [N, 1] array of float , The origin position on the y axis
    :param arm_angles: [N, arm_num] an array of angles [-pi, pi] that states the angle of the arm w.r.t
                        horizontal surface. Horizon to the right is 0, up is positive and below is negative
    :param arm_lengths: [N, arm_num] an array of float that states the lengths of the robotic arms
    :return: current_pos: [N, 2] The final position of the robotic arm, a array of 2 numbers indicating (x, y) co-ordinates
    :return: positions: list of #arm_num of [N, 2], The positions that the arms nodes for plotting purpose
    """
    # First make sure that the angles are legal angles
    for angle in arm_angles:
        assert (angle < pi) and (angle > -pi), 'Your angle has to be within [-pi, pi]'

    # Start computing for positions
    positions = []                                          # Holder for positions to be plotted
    current_pos = np.zeros([len(origin_pos), 2])            # The robotic arm starts from [0, y]
    current_pos[:, 0] = 0                                   # The x axis is 0 for all cases
    current_pos[:, 1] = origin_pos                          # Plug in the
    positions.append(current_pos)
    for i in range(arm_num):                                # Loop through the arms
        current_pos[:, 0] += sin(arm_angles[:, i]) * arm_lengths[:, i]
        current_pos[:, 1] += cos(arm_angles[:, i]) * arm_lengths[:, i]
        positions.append(current_pos)
    return current_pos, positions


def plot_arms(positions, save_dir='.', save_name='robotic_arms_plot.png'):
    """
    Plot the position of the robotic arms given positions of the points
    :param positions: The position of the joints places
    :param save_dir: The saving directory of the plot
    :param save_name: The saving name of the plot
    :return: None
    """
    f = plt.figure()
    print(positions[0])
    print(np.shape(positions[0]))
    plt.plot(positions[0, :, 0], positions[0, :, 1], 's')
    plt.plot(positions[1:, :, 0], positions[1:, :, 1], 'bo')
    f.savefig(join(save_dir, save_name))



if __name__ == '__main__':
    positions = np.array([np.reshape(np.array([0, 2]), [-1, 2]), np.reshape(np.array([1, 2]), [-1, 2]), np.reshape(np.array([2, 0.5]), [-1, 2])])
    print("original position shape", np.shape(positions))
    plot_arms(positions)
