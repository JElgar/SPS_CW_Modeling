import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def view_data_segments(xs, ys):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        plot: the plot of the graph
    """
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    fig, ax = plt.subplots()
    plt.set_cmap('Dark2')
    ax.scatter(xs, ys, c=colour)
    return fig, ax

def least_squares(xs, ys):
    """ Calculate least squares of data
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        l where l is 1d array containing the least squares of the given values
    """
    ones = np.ones(xs.shape)
    x_o = np.column_stack((ones, xs))
    v = np.linalg.inv(x_o.T.dot(x_o)).dot(x_o.T).dot(ys)
    return v

def linear_reg_draw(xs, ys):
    """ Find linear line of best fit for given data
    Args: 
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    a, b = least_squares(xs, ys)

    x_min = xs.min()
    x_max = xs.max() 
    y_min = a + b * x_min # y value at min x
    y_max = a + b * x_max # y value at max x
    
    fig, ax = view_data_segments(xs, ys)  
    ax.plot([x_min, x_max], [y_min, y_max], 'r-', lw=2)
    plt.show()

def linear_reg(xs, ys, fig, ax):
    """ Find linear line of best fit for given data
    Args: 
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    a, b = least_squares(xs, ys)

    x_min = xs.min()
    x_max = xs.max() 
    y_min = a + b * x_min # y value at min x
    y_max = a + b * x_max # y value at max x
    
    ax.plot([x_min, x_max], [y_min, y_max], 'r-', lw=2)
    return fig, ax


def f(x, a, b, c):
    return a*x**2 + b*x + c

def gls(xs, ys, p):
    """ Calculate least squares of data with order p
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        l where l is px1 array containing the least squares of the given values
    """
    x_o = np.ones(xs.shape)
    for i in range(1, p):
        r = np.power(xs, i)
        x_o = np.column_stack((x_o, r))

    v = np.linalg.inv(x_o.T.dot(x_o)).dot(x_o.T).dot(ys)
    return v

    
def treg(xs, ys, fig, ax):
    i = 0
    while i < len(xs):
        lxs = np.take(xs, list(range(i, i+20)))
        lys = np.take(ys, list(range(i, i+20)))
        fig, ax = linear_reg(lxs, lys, fig, ax)
        i += 20
    return fig, ax


"""
TODO:
    - Implement exponential a + be^x
    - Implement trig a + bsin(cx + d)
"""


def reg(xs, ys, p):
    """ Given xs and ys, plot linear regression with features p
    Args: 
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    fig, ax = view_data_segments(xs, ys)  
    if p == 2:
        fig, ax = treg(xs, ys, fig, ax)
    else: 
        ls = gls(xs, ys, p)
        print(ls)
        lx = np.linspace(xs.min(), xs.max(), len(xs))

        ly = 0
        for i in range(0, p):
           ly += ls[i] * lx ** i
        print(square_error(ys, ly))
        ax.plot(lx, ly, '.')

    plt.show()

def square_error(y, y_est):
    return np.sum((y - y_est) ** 2)


