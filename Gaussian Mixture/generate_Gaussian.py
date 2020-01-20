"""
This is the function that generates the gaussian mixture for artificial model
The simulated cluster would be similar to the artifical data set from the INN paper
"""
# Import libs
import numpy as np
import matplotlib.pyplot as plt


# Define the free parameters
dimension = 2
num_cluster = 8
num_points_per_cluster = 100
num_class = 4
cluster_distance_to_center = 10
in_class_variance = 1


def plotData(data_x, data_y, save_dir='data_scatter.png'):
    """
    Plot the scatter plot of the data to show the overview of the data points
    :param data_x: The 2 dimension x values of the data points
    :param data_y: The class of the data points
    :return: None
    """
    f = plt.figure()
    plt.scatter(data_x[:, 0], data_x[:, 1], c=data_y)
    f.savefig(save_dir)


if __name__ == '__main__':
    centers = np.zeros([num_cluster, dimension])            # initialize the center positions
    for i in range(num_cluster):                            # the centers are at the rim of a circle with equal angle
        centers[i, 0] = np.cos(2 * np.pi / num_cluster * i) * cluster_distance_to_center
        centers[i, 1] = np.sin(2 * np.pi / num_cluster * i) * cluster_distance_to_center
    print("centers", centers)
    # Initialize the data points for x and y
    data_x = np.zeros([num_cluster * num_points_per_cluster, dimension])
    data_y = np.zeros(num_cluster * num_points_per_cluster)
    # allocate the class labels
    class_for_cluster = np.random.uniform(low=0, high=num_class, size=num_cluster).astype('int')
    print("class for cluster", class_for_cluster)
    # Loop through the points and assign the cluster x and y values
    for i in range(len(data_x[:, 0])):
        i_class = i // num_points_per_cluster
        data_y[i] = class_for_cluster[i_class]             # Assign y to be 0,0,0....,1,1,1...,2,2,2.... e.g.
        data_x[i, 0] = np.random.normal(centers[i_class, 0], in_class_variance)
        data_x[i, 1] = np.random.normal(centers[i_class, 1], in_class_variance)
    print("data_y", data_y)
    plotData(data_x, data_y)
    np.savetxt('data_x.csv', data_x, delimiter=',')
    np.savetxt('data_y.csv', data_y, delimiter=',')
