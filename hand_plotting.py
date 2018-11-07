
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as plt3d
import matplotlib.lines as mlines
import numpy as np

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_hand_points(hand_points):
    x_coords = hand_points[::3]
    y_coords = hand_points[1::3]
    z_coords = hand_points[2::3]
    
    mean_x_coords = np.mean(x_coords)
    mean_y_coords = np.mean(y_coords)
    mean_z_coords = np.mean(z_coords)
    
    fig = plt.figure()
    fig.set_size_inches(10,10)
    ax = fig.add_subplot(111, projection='3d', aspect='equal')
    hand_plot = ax.scatter(x_coords, y_coords, z_coords, depthshade=False)
    
    def plot_finger(inds_array):
        for i in range(len(inds_array)-1):
            xs = (x_coords[inds_array[i]], x_coords[inds_array[i+1]])
            ys = (y_coords[inds_array[i]], y_coords[inds_array[i+1]])
            zs = (z_coords[inds_array[i]], z_coords[inds_array[i+1]])
            line_seg = plt3d.art3d.Line3D(xs, ys, zs)
            ax.add_line(line_seg)
        
    # Draw thumb
    thumb_inds = [0, 1, 6, 7, 8]
    plot_finger(thumb_inds)
    
    # Draw index
    index_inds = [0, 2, 9, 10, 11]
    plot_finger(index_inds)
    
    # Draw middle
    middle_inds = [0, 3, 12, 13, 14]
    plot_finger(middle_inds)
    
    # Draw ring
    ring_inds = [0, 4, 15, 16, 17]
    plot_finger(ring_inds)
    
    # Draw pinky
    pinky_inds = [0, 5, 18, 19, 20]
    plot_finger(pinky_inds)
    
    # Working out axes
    axis_size = 120.0
    ax.set_xlim(mean_x_coords-axis_size/2.0, mean_x_coords+axis_size/2.0)
    ax.set_ylim(mean_y_coords-axis_size/2.0, mean_y_coords+axis_size/2.0)
    ax.set_zlim(mean_z_coords-axis_size/2.0, mean_z_coords+axis_size/2.0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    
    blue_line = mlines.Line2D([], [], color='blue', label='Ground Truth')
    plt.legend(handles=[blue_line])
        
    plt.show()

def plot_two_hands(hand_points1, hand_points2, save_fig = False, save_name = None, first_hand_label='Prediction', second_hand_label='Ground Truth', pred_uncertainty=None):
    x_coords1 = hand_points1[::3]
    y_coords1 = hand_points1[1::3]
    z_coords1 = hand_points1[2::3]
    
    mean_x_coords = np.mean(x_coords1)
    mean_y_coords = np.mean(y_coords1)
    mean_z_coords = np.mean(z_coords1)
    
    x_coords2 = hand_points2[::3]
    y_coords2 = hand_points2[1::3]
    z_coords2 = hand_points2[2::3]
    
    fig = plt.figure()
    fig.set_size_inches(10,10)
    ax = fig.add_subplot(111, projection='3d', aspect='equal')
    hand_plot1 = ax.scatter(x_coords1, y_coords1, z_coords1, depthshade=False)
    hand_plot2 = ax.scatter(x_coords2, y_coords2, z_coords2, depthshade=False, c='r')

    if pred_uncertainty is not None:
        #the given matrix is of size 3*num_points x 3*num_points
        #each submatrix of size 3x3 around the diagonal is the covariance of the point in 3D space
        for (idx, (x_mean, y_mean, z_mean)) in enumerate(zip(x_coords1, y_coords1, z_coords1)):
            uncertainty_mat = pred_uncertainty[0, 0, idx:idx+3,idx:idx+3]
            mu=np.array([x_mean,y_mean,z_mean])
            sigma=np.matrix(uncertainty_mat)
            data=np.random.multivariate_normal(mu,sigma,1000)
            values = data.T

            kde = stats.gaussian_kde(values)
            density = kde(values)
            density = density/density.max()
            #print(density.shape)
            x, y, z = values
            #dotcolors=[(a,0.00001) for a in density]
            #ax.scatter(x, y, z, c=density, cmap='Blues', s= 0.05*len(x_coords2))
            dotcolors=[(0.2, 0.4, 0.6, 0.1) for a in density]
            ax.scatter(x, y, z, c=dotcolors, edgecolors='None')

    
    def plot_finger(inds_array, x_coords, y_coords, z_coords, color='b'):
        for i in range(len(inds_array)-1):
            xs = (x_coords[inds_array[i]], x_coords[inds_array[i+1]])
            ys = (y_coords[inds_array[i]], y_coords[inds_array[i+1]])
            zs = (z_coords[inds_array[i]], z_coords[inds_array[i+1]])
            line_seg = plt3d.art3d.Line3D(xs, ys, zs, color=color)
            ax.add_line(line_seg)
        
    # Draw thumbs
    thumb_inds = [0, 1, 6, 7, 8]
    plot_finger(thumb_inds, x_coords1, y_coords1, z_coords1)
    plot_finger(thumb_inds, x_coords2, y_coords2, z_coords2, color='r')
    
    # Draw indexes
    index_inds = [0, 2, 9, 10, 11]
    plot_finger(index_inds, x_coords1, y_coords1, z_coords1)
    plot_finger(index_inds, x_coords2, y_coords2, z_coords2, color='r')
    
    # Draw middles
    middle_inds = [0, 3, 12, 13, 14]
    plot_finger(middle_inds, x_coords1, y_coords1, z_coords1)
    plot_finger(middle_inds, x_coords2, y_coords2, z_coords2, color='r')
    
    # Draw rings
    ring_inds = [0, 4, 15, 16, 17]
    plot_finger(ring_inds, x_coords1, y_coords1, z_coords1)
    plot_finger(ring_inds, x_coords2, y_coords2, z_coords2, color='r')
    
    # Draw pinkies
    pinky_inds = [0, 5, 18, 19, 20]
    plot_finger(pinky_inds, x_coords1, y_coords1, z_coords1)
    plot_finger(pinky_inds, x_coords2, y_coords2, z_coords2, color='r')
    
    # Working out axes
    axis_size = 120.0
    ax.set_xlim(mean_x_coords-axis_size/2.0, mean_x_coords+axis_size/2.0)
    ax.set_ylim(mean_y_coords-axis_size/2.0, mean_y_coords+axis_size/2.0)
    ax.set_zlim(mean_z_coords-axis_size/2.0, mean_z_coords+axis_size/2.0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    
    blue_line = mlines.Line2D([], [], color='blue', label=first_hand_label)
    red_line = mlines.Line2D([], [], color='red', label=second_hand_label)
    plt.legend(handles=[blue_line, red_line])
        
    plt.show()
    
    if save_fig:
        plt.savefig(save_name)