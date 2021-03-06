import os
import sys
import pandas as pd
import numpy as np
import copy as c 
import functools
from matplotlib import pyplot as plt

NUM_REPEATS = 50

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

    # TODO This is thrwoing single matrix error (does not have inverse)
    v = np.linalg.inv(x_o.T.dot(x_o)).dot(x_o.T).dot(ys)
    return v

def exp_ls(xs, ys):
    o = np.ones(xs.shape)
    x = np.column_stack((np.ones(xs.shape), np.exp(xs)))
    v = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(ys)
    return v

def sin_ls(xs, ys):
    x = np.column_stack((np.ones(xs.shape), np.sin(xs)))
    v = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(ys)
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
        ax.plot(lx, ly)

    plt.show()

def exp_reg(xs, ys):
    fig, ax = view_data_segments(xs, ys)  
    ls = exp_ls(xs, ys)
    lx = np.linspace(xs.min(), xs.max(), len(xs))
    ly = f_exp(lx, ls)
    ax.plot(lx, ly)
    plt.show()
    

def best_p (xs, ys):
    """
    Get error values for all poly regressions and return the model with the smallest error
    Args:
        xs : List/array-like of x co-ordinates, size <= 20
        ys : List/array-like of x co-ordinates, size <= 20
    """
    MAX_FEATURES = 5

    """
    Find the error in each of the models, shuffle the data and repeat, adding the errors each time
    At the end select the model with the lowest error and jobs a gooden
    """
    xs_shuffle = c.deepcopy(xs)
    ys_shuffle = c.deepcopy(ys)
    errors = [0] * (MAX_FEATURES - 2 + 1) # When multiplying you dont index at zero (that makes sense in my head sorry)
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
        # print("The lenght of the training data is: " + str(len(xs_training)))

        # Get the least squares, find the error bettwen teh training and test 
        for p in range(2, MAX_FEATURES+1):
            # Get the least squares for the training set
            ls = gls(xs_training, ys_training, p)
            # Using the least squares calculate the predicted ys for the test xs
            ys_hat = f(xs_testing, ls, p)
            # Find the square_error between test_ys and ys_hat
            err = square_error(ys_testing, ys_hat)
            # print("The error for p: " + str(p) + " is: " + str(err))
            errors[p-2] += err
        
    # print(errors)
    # Return the value of the best p
    print("Best p is " + str (errors.index(min(errors)) + 2))
    m = min(errors)
    return errors.index(m) + 2, m

def best_model (xs, ys):
    # Get the best fitting poly
    bestp, poly_err = best_p(xs, ys)
    exp_err = 0
    sin_err = 0
    xs_shuffle = c.deepcopy(xs)
    ys_shuffle = c.deepcopy(ys)
    for i in range(0, NUM_REPEATS):
        rng_state = np.random.get_state()
        np.random.set_state(rng_state)
        np.random.shuffle(xs_shuffle)
        np.random.set_state(rng_state)
        np.random.shuffle(ys_shuffle)

        xs_training, xs_testing = xs_shuffle[6:], xs_shuffle[:6]
        ys_training, ys_testing = ys_shuffle[6:], ys_shuffle[:6]
       
        # EXP
        # Get the least squares for the training set
        ls = exp_ls(xs_training, ys_training)
        # Using the least squares calculate the predicted ys for the test xs
        ys_hat = f_exp(xs_testing, ls)
        # Find the square_error between test_ys and ys_hat
        err = square_error(ys_testing, ys_hat)
        exp_err += err
        
        # SIN
        # Get the least squares for the training set
        ls = sin_ls(xs_training, ys_training)
        # Using the least squares calculate the predicted ys for the test xs
        ys_hat = f_sin(xs_testing, ls)
        # Find the square_error between test_ys and ys_hat
        err = square_error(ys_testing, ys_hat)
        sin_err += err
       

    print(exp_err, poly_err)
    if (exp_err < poly_err and exp_err < sin_err):
        return "exp", -1
    
    if (poly_err < exp_err and poly_err < sin_err):
        return "poly", bestp

    if (sin_err < exp_err and sin_err < poly_err):
        return "sin", -1


def split_regression(xs, ys):
    fig, ax = view_data_segments(xs, ys)  
    i = 0
    total_err = 0
    while i < len(xs):
        lxs = np.take(xs, list(range(i, i+20)))
        lys = np.take(ys, list(range(i, i+20)))
        i += 20
        fig, ax, err = regression(lxs, lys, fig, ax)
        total_err += err
    plt.show()
    return total_err

def regression(xs, ys, fig, ax):
    """ Given xs and ys, plot linear regression with best_p features
    Args: 
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    # print("The data this regression has got is: ")
    # print(xs, ys)
    # Get the best fitting number of p values
    # Gets the best model
    model, p = best_model(xs, ys)
    lx = np.linspace(xs.min(), xs.max(), len(xs))
    err = 0
    if model == "poly":
        # Calcualte least squares for this value of p 
        ls = gls(xs, ys, p)
        # Calculate the values to plot
        ly = f(lx, ls, p)
        # print(square_error(ys, ly))
        ax.plot(lx, ly)
       
        # Find ys values according to model
        est_ys = f(xs, ls, p)
        # Find the square_error between test_ys and ys_hat
        err = square_error(ys, est_ys)
    if model == "exp":
        ls = exp_ls(xs, ys)
        ly = f_exp(lx, ls)
        ax.plot(lx, ly)
        
        # Find ys values according to model
        est_ys = f_exp(xs, ls)
        # Find the square_error between test_ys and ys_hat
        err = square_error(ys, est_ys)
    if model == "sin":
        ls = sin_ls(xs, ys)
        ly = f_sin(lx, ls)
        ax.plot(lx, ly)
        
        # Find ys values according to model
        est_ys = f_sin(xs, ls)
        # Find the square_error between test_ys and ys_hat
        err = square_error(ys, est_ys)
        print("The best model was sin!!")

    return fig, ax, err

# Helper functions

def setup (fileName):
    xs, ys = load_points_from_file(fileName)
    #fig, ax = view_data_segments(xs, ys)
    #regression(xs, ys, fig, ax)
    split_regression(xs, ys)
    #show(fig, ax)
 
def show(fig, ax):
    plt.show()

def f_exp(x, ls):
    return ls[0] + ls[1] * np.exp(x)

def f_sin(x, ls):
    return ls[0] + ls[1] * np.sin(x)

def f(x, ls, p):
    y = 0
    for i in range(0, p):
       y += ls[i] * x ** i
    return y

def square_error(y, y_est):
    return np.sum((y - y_est) ** 2)

def are_lists_identical(cs, bs):
    if functools.reduce(lambda i, j : i and j, map(lambda m, k: m == k, cs, bs), True) :  
        return True
    else : 
        return False
