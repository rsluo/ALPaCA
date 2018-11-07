
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as plt3d
import matplotlib.lines as mlines
import numpy as np

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import nestle

from numpy import linalg
from random import random

class EllipsoidTool:
    """Some stuff for playing with ellipsoids"""
    def __init__(self): pass
    
    def getMinVolEllipse(self, P=None, tolerance=0.01):
        """ Find the minimum volume ellipsoid which holds all the points
        
        Based on work by Nima Moshtagh
        http://www.mathworks.com/matlabcentral/fileexchange/9542
        and also by looking at:
        http://cctbx.sourceforge.net/current/python/scitbx.math.minimum_covering_ellipsoid.html
        Which is based on the first reference anyway!
        
        Here, P is a numpy array of N dimensional points like this:
        P = [[x,y,z,...], <-- one point per line
             [x,y,z,...],
             [x,y,z,...]]
        
        Returns:
        (center, radii, rotation)
        
        """
        (N, d) = np.shape(P)
        d = float(d)
    
        # Q will be our working array
        Q = np.vstack([np.copy(P.T), np.ones(N)]) 
        QT = Q.T
        
        # initializations
        err = 1.0 + tolerance
        u = (1.0 / N) * np.ones(N)

        # Khachiyan Algorithm
        while err > tolerance:
            V = np.dot(Q, np.dot(np.diag(u), QT))
            M = np.diag(np.dot(QT , np.dot(linalg.inv(V), Q)))    # M the diagonal vector of an NxN matrix
            j = np.argmax(M)
            maximum = M[j]
            step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
            new_u = (1.0 - step_size) * u
            new_u[j] += step_size
            err = np.linalg.norm(new_u - u)
            u = new_u

        # center of the ellipse 
        center = np.dot(P.T, u)
    
        # the A matrix for the ellipse
        A = linalg.inv(
                       np.dot(P.T, np.dot(np.diag(u), P)) - 
                       np.array([[a * b for b in center] for a in center])
                       ) / d
                       
        # Get the values we'd like to return
        U, s, rotation = linalg.svd(A)
        radii = 1.0/np.sqrt(s)
        
        return (center, radii, rotation)

    def getEllipsoidVolume(self, radii):
        """Calculate the volume of the blob"""
        return 4./3.*np.pi*radii[0]*radii[1]*radii[2]

    def plotEllipsoid(self, center, radii, rotation, ax=None, plotAxes=False, cageColor='b', cageAlpha=0.2):
        """Plot an ellipsoid"""
        make_ax = ax == None
        if make_ax:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
        u = np.linspace(0.0, 2.0 * np.pi, 100)
        v = np.linspace(0.0, np.pi, 100)
        
        # cartesian coordinates that correspond to the spherical angles:
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
        # rotate accordingly
        for i in range(len(x)):
            for j in range(len(x)):
                [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center
    
        if plotAxes:
            # make some purdy axes
            axes = np.array([[radii[0],0.0,0.0],
                             [0.0,radii[1],0.0],
                             [0.0,0.0,radii[2]]])
            # rotate accordingly
            for i in range(len(axes)):
                axes[i] = np.dot(axes[i], rotation)
    
    
            # plot axes
            for p in axes:
                X3 = np.linspace(-p[0], p[0], 100) + center[0]
                Y3 = np.linspace(-p[1], p[1], 100) + center[1]
                Z3 = np.linspace(-p[2], p[2], 100) + center[2]
                ax.plot(X3, Y3, Z3, color=cageColor)
    
        # plot ellipsoid
        ax.plot_wireframe(x, y, z,  rstride=6, cstride=6, color=cageColor, alpha=cageAlpha)
        
        if make_ax:
            plt.show()
            plt.close(fig)
            del fig


def plot_ellipsoid_3d(ell, ax):
    """Plot the 3-d Ellipsoid ell on the Axes3D ax."""

    # points on unit sphere
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    z = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    x = np.outer(np.ones_like(u), np.cos(v))

    # transform points to ellipsoid
    for i in range(len(x)):
        for j in range(len(x)):
            x[i,j], y[i,j], z[i,j] = ell.ctr + np.dot(ell.axes,
                                                      [x[i,j],y[i,j],z[i,j]])

    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='#2980b9', alpha=0.2)

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

            ET = EllipsoidTool()
            uncertainty_mat = pred_uncertainty[0, 0, idx:idx+3,idx:idx+3]
            mu=np.array([x_mean,y_mean,z_mean])
            sigma=np.matrix(uncertainty_mat)
            npoints = 1000
            data=np.random.multivariate_normal(mu,sigma,npoints)
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
            (center, radii, rotation) = ET.getMinVolEllipse(values.T, .01)
            ET.plotEllipsoid(center, radii*0.5, rotation, ax=ax, plotAxes=True)
            # print(uncertainty_mat)
            # ell_gen = nestle.Ellipsoid(mu, np.dot(uncertainty_mat.T, uncertainty_mat))
            # print(ell_gen.vol)
            # pointvol = ell_gen.vol / 0.01
            # ells = nestle.bounding_ellipsoids(values, pointvol)
            # for ell in ells:
            #     plot_ellipsoid_3d(ell, ax)
            #     npoints

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

def plot_finger(inds_array, x_coords, y_coords, z_coords, ax, color='b', alpha=1.0):
        for i in range(len(inds_array)-1):
            xs = (x_coords[inds_array[i]], x_coords[inds_array[i+1]])
            ys = (y_coords[inds_array[i]], y_coords[inds_array[i+1]])
            zs = (z_coords[inds_array[i]], z_coords[inds_array[i+1]])
            line_seg = plt3d.art3d.Line3D(xs, ys, zs, color=color, alpha=alpha)
            ax.add_line(line_seg)

def plot_two_hands2(hand_points1, hand_points2, save_fig = False, save_name = None, first_hand_label='Prediction', second_hand_label='Ground Truth', pred_uncertainty=None):
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

        
    # Draw thumbs
    thumb_inds = [0, 1, 6, 7, 8]
    plot_finger(thumb_inds, x_coords1, y_coords1, z_coords1, ax)
    plot_finger(thumb_inds, x_coords2, y_coords2, z_coords2, ax, color='r')
    
    # Draw indexes
    index_inds = [0, 2, 9, 10, 11]
    plot_finger(index_inds, x_coords1, y_coords1, z_coords1, ax)
    plot_finger(index_inds, x_coords2, y_coords2, z_coords2, ax, color='r')
    
    # Draw middles
    middle_inds = [0, 3, 12, 13, 14]
    plot_finger(middle_inds, x_coords1, y_coords1, z_coords1, ax)
    plot_finger(middle_inds, x_coords2, y_coords2, z_coords2, ax, color='r')
    
    # Draw rings
    ring_inds = [0, 4, 15, 16, 17]
    plot_finger(ring_inds, x_coords1, y_coords1, z_coords1, ax)
    plot_finger(ring_inds, x_coords2, y_coords2, z_coords2, ax, color='r')
    
    # Draw pinkies
    pinky_inds = [0, 5, 18, 19, 20]
    plot_finger(pinky_inds, x_coords1, y_coords1, z_coords1, ax)
    plot_finger(pinky_inds, x_coords2, y_coords2, z_coords2, ax, color='r')

    sampled_points_x = []
    sampled_points_y = []
    sampled_points_z = []

    npoints = 100


    if pred_uncertainty is not None:
        #the given matrix is of size 3*num_points x 3*num_points
        #each submatrix of size 3x3 around the diagonal is the covariance of the point in 3D space
        for (idx, (x_mean, y_mean, z_mean)) in enumerate(zip(x_coords1, y_coords1, z_coords1)):

            ET = EllipsoidTool()
            uncertainty_mat = pred_uncertainty[0, 0, idx:idx+3,idx:idx+3]
            mu=np.array([x_mean,y_mean,z_mean])
            sigma=np.matrix(uncertainty_mat)
            
            data=np.random.multivariate_normal(mu,sigma,npoints)
            values = data.T

            #kde = stats.gaussian_kde(values)
            #density = kde(values)
            #density = density/density.max()
            #print(density.shape)
            x, y, z = values
            #dotcolors=[(a,0.00001) for a in density]
            #ax.scatter(x, y, z, c=density, cmap='Blues', s= 0.05*len(x_coords2))
            #dotcolors=[(0.2, 0.4, 0.6, 0.1) for a in density]
            #ax.scatter(x, y, z, c=dotcolors, edgecolors='None')
            #(center, radii, rotation) = ET.getMinVolEllipse(values.T, .01)
            #ET.plotEllipsoid(center, radii*0.5, rotation, ax=ax, plotAxes=True)
            # print(uncertainty_mat)
            # ell_gen = nestle.Ellipsoid(mu, np.dot(uncertainty_mat.T, uncertainty_mat))
            # print(ell_gen.vol)
            # pointvol = ell_gen.vol / 0.01
            # ells = nestle.bounding_ellipsoids(values, pointvol)
            # for ell in ells:
            #     plot_ellipsoid_3d(ell, ax)
            #     npoints
            sampled_points_x += [x]
            sampled_points_y += [y]
            sampled_points_z += [z]

        # Pick one sample of each 3D gaussian and draw a hand
        sampled_points_x= np.array(sampled_points_x)
        sampled_points_y=np.array(sampled_points_y)
        sampled_points_z=np.array(sampled_points_z)
        for hand_sample in range(npoints):

            sampled_hand_x = sampled_points_x[:, hand_sample]
            sampled_hand_y = sampled_points_y[:, hand_sample]
            sampled_hand_z = sampled_points_z[:, hand_sample]

            plot_finger(thumb_inds, sampled_hand_x, sampled_hand_y, sampled_hand_z, ax, color='r', alpha=0.1)
            plot_finger(index_inds, sampled_hand_x, sampled_hand_y, sampled_hand_z, ax, color='r', alpha=0.1)
            plot_finger(middle_inds, sampled_hand_x, sampled_hand_y, sampled_hand_z, ax, color='r', alpha=0.1)
            plot_finger(ring_inds, sampled_hand_x, sampled_hand_y, sampled_hand_z, ax, color='r', alpha=0.1)
            plot_finger(pinky_inds, sampled_hand_x, sampled_hand_y, sampled_hand_z, ax, color='r', alpha=0.1)

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