import os
import sys
import pandas as pd
import numpy as np
import copy as c 
import functools
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
    # This is temporary and eventually will call on only 20 points
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

def best_p (xs, ys):
    """
    Get error values for all poly regressions and return the model with the smallest error
    Args:
        xs : List/array-like of x co-ordinates, size <= 20
        ys : List/array-like of x co-ordinates, size <= 20
    """
    MAX_FEATURES = 8
    NUM_REPEATS = 50

    if (len(xs) != 20):
        print("xs lenght was not equal 20 so stuffs gonna break")
        print("actually not yet")

    
    """
    Find the error in each of the models, shuffle the data and repeat, adding the errors each time
    At the end select the model with the lowest error and jobs a gooden
    """
    xs_shuffle = c.deepcopy(xs)
    ys_shuffle = c.deepcopy(ys)
    errors = [0] * (MAX_FEATURES - 2)
    for i in range(0, NUM_REPEATS):
        rng_state = np.random.get_state()
        np.random.set_state(rng_state)
        np.random.shuffle(xs_shuffle)
        np.random.set_state(rng_state)
        np.random.shuffle(ys_shuffle)
        assert not are_lists_identical(xs, xs_shuffle)
        assert not are_lists_identical(ys, ys_shuffle)
        xs_training, xs_testing = xs_shuffle[6:], xs_shuffle[:6]
        ys_training, ys_testing = ys_shuffle[6:], ys_shuffle[:6]
        print("The lenght of the training data is: " + str(len(xs_training)))

        # Get the least squares, find the error bettwen teh training and test 
        for p in range(2, MAX_FEATURES+1):
            # Get the least squares for the training set
            ls = gls(xs_training, ys_training, p)
            # Using the least squares calculate the predicted ys for the test xs
            ys_hat = f(xs_testing, ls, p)
            # Find the square_error between test_ys and ys_hat
            err = square_error(ys_testing, ys_hat)
            print("The error for p: " + str(p) + " is: " + str(err))
            errors[p-2-1] += err
        
    print(errors)
    # Return the value of the best p
    print("Best p is " + str (errors.index(min(errors)) + 2))
    return errors.index(min(errors)) + 2

def split_regression(xs, ys):
    fig, ax = view_data_segments(xs, ys)  
    i = 0
    while i < len(xs):
        lxs = np.take(xs, list(range(i, i+20)))
        lys = np.take(ys, list(range(i, i+20)))
        fig, ax = regression(lxs, lys, fig, ax)
        i += 20
    plt.show()

def regression(xs, ys, fig, ax):
    """ Given xs and ys, plot linear regression with best_p features
    Args: 
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    # Get the best fitting number of p values
    p = best_p(xs, ys)
    # Calcualte least squares for this value of p 
    ls = gls(xs, ys, p)
    
    # Calculate the values to plot
    lx = np.linspace(xs.min(), xs.max(), len(xs))
    ly = f(lx, ls, p)
    print(square_error(ys, ly))
    ax.plot(lx, ly, '.')
    return fig, ax

# HANDY DANDY FUNCTIONS 

def setup (fileName):
    xs, ys = load_points_from_file(fileName)
    fig, ax = view_data_segments(xs, ys)
    regression(xs, ys, fig, ax)
    show(fig, ax)
 
def show(fig, ax):
    plt.show()

def f(x, ls, p):
    y = 0
    for i in range(0, p):
       y += ls[i] * x ** i
    return y

def square_error(y, y_est):
    return np.sum((y - y_est) ** 2)

def are_lists_identical(cs, bs):
    if functools.reduce(lambda i, j : i and j, map(lambda m, k: m == k, cs, bs), True) :  
        print ("The lists are identical") 
    else : 
        print ("The lists are not identical") 
