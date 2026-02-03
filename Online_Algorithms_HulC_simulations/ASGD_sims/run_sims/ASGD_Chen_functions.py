# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 10:50:11 2023

@author: Selina Carter

This file produces gradient functions to perform linear and logistic regression
via stochastic gradient descent, as well as a function to produce ASGD plug-in
confidence intervals (see Chen et al 2016,"Statistical Inference for Model
                                    Parameters in Stochastic Gradient Descent")

"""
#import pandas as pd
import numpy as np
from scipy.stats import norm
#from sklearn import linear_model as lm, metrics
import statsmodels.api as sm
#import matplotlib.pyplot as plt



#############################
### Linear regression #######
#############################


def SGD_grad_linear(theta, x_i, y_i):
    #This is the SGD gradient for a single data point (x_i, y_i) in linear regression.
    #x_i is a numpy array with size d, y_i is a scalar (must be float or numpy.float64)
    #theta is the parameter: must be numpyarray with size d.
    assert theta.size==x_i.size, "theta and x_i must have same dimension (x_i.size = " \
        + str(x_i.size) + ", theta.size = "  +  str(theta.size) + ")"
    
    assert type(y_i)==float or type(y_i)==np.float64, \
        "y_i type must be a float or numpy.float. (Currently, type(y_i) = " + str(type(y_i)) + ")" 
        
    
    out = (np.dot(theta, x_i)-y_i)*x_i
    
    assert out.size==x_i.size, "gradient should have same dimension as x_i " + \
        "(Currently, out.size = " + str(out.size) + ", x_i.size = " + str(x_i.size) + ")"
    
    return(out)
    


def SGD_hessian_linear(x_i):
    # This is the SGD Hessian matrix for a single data point (x_i, y_i).
    # x_i is a numpy array with size d, y_i is a scalar (must be float or numpy.float64)
    # Notice that theta is not an argument for the Hessian matrix
    # This is matrix A in the Chen et al 2016 paper (see their Example 2.1)
    
    H = np.outer(x_i, x_i)
    
    return(H)



def eta_t(t, c = .01, alpha=.501):
    # Produces a step size that diminishes to 0 as t gets large.
    # t is for time step (positive integer)
    # c>0 is a constant 
    # alpha is in [.5, 1)
    # Chen et al use alpha=.501, but c can vary and greatly affects SGD's convergence.
    # NOTE: c in (.01, .02) and alpha in (.51, .6) seems to be the "magic" numbers
    #          to make ASGD work for simple linear regression.
    
    assert alpha >.5 and alpha <1, "alpha must be in range (.5, 1)."
    assert c > 0, "c must be positive."
    
    step_size = c*t**(-alpha)
    
    return(step_size)
    


def ASGD_plugin_CI(theta_bar, A_bar, S_bar, n, alpha_level = .05):
    # Produces a confidence interval for theta using plugin estimator (see
    # Chen & Lee 2016, Section 4)
    # alpha_level is the Type 1 error rate
    
    
    #Get inverse of A_bar
    A_bar_inv = np.linalg.inv(A_bar)
    
    #compute variance
    Sigma = np.linalg.multi_dot((A_bar_inv, S_bar, A_bar_inv))
    
    d = len(theta_bar)
    CI = []
    z = norm.ppf(1-alpha_level)
    
    for i in range(d):
        
        lb = theta_bar[i]-z*np.sqrt(Sigma[i,i])/np.sqrt(n) 
        ub = theta_bar[i]+z*np.sqrt(Sigma[i,i])/np.sqrt(n)
        
        CI_i = [lb, ub]
        
        CI.append(CI_i)
    
    
    
    return CI, Sigma
    

def OLS_sandwich_CI(data, alpha_level = .05):
        
    
    X = data[0]
    Y = data[1]
    
    n = X.shape[0]
    #Check if X matrix has column of ones; if not, add it.
    if np.sum(X[:,0]) != n:
        X = np.c_[np.ones(n), X]
        
    
    
    lm_reg = sm.OLS( Y, X )
    lm_results = lm_reg.fit()
    theta_hat = lm_results.params
    
    d = X.shape[1]
    
    A_hat = np.matmul(np.transpose(X),X)/n
    # diff = Y-np.matmul(X, theta_hat)  
    # diff_sq = diff**2
    # #The following is an n x n matrix, where each column is diff_sq
    # diff_sq_matrix = np.stack([diff_sq for _ in range(n)], axis=1) 
    # S_hat = np.linalg.multi_dot((np.transpose(X), diff_sq_matrix, X))/n
    
    S_hat = 0
    for i in range(n):
        S_hat = S_hat + np.outer(X[i], X[i])*(Y[i]-np.dot(X[i], theta_hat))**2
    
    
    S_hat = S_hat/n
    
    Sigma = np.linalg.multi_dot((np.linalg.inv(A_hat), S_hat, np.linalg.inv(A_hat)))
    
    
    CI = []
    z = norm.ppf(1-alpha_level)
    
    for i in range(d):
        
        lb = theta_hat[i]-z*np.sqrt(Sigma[i,i])/np.sqrt(n) 
        ub = theta_hat[i]+z*np.sqrt(Sigma[i,i])/np.sqrt(n)
        
        CI_i = [lb, ub]
        
        CI.append(CI_i)

    return CI, Sigma, theta_hat






#############################
### Logistic regression #####
#############################

def sigmoid(s):
    #s is a scalar
    denom = 1 + np.exp(-s)
    
    return 1/denom
    


def SGD_grad_logistic(theta, x_i, y_i, ytype = "neg11"):
    # This is the SGD gradient for a single data point (x_i, y_i).
    # (see Chen et al 2016, Example 2.2)
    # x_i is a numpy array with size d.
    # ytype = "neg11" means y_i is binary {-1, 1}. type = "01" means y_i is binary {0, 1}
    # theta is the parameter: must be numpyarray with size d.
    assert theta.size==x_i.size, "theta and x_i must have same dimension (x_i.size = " \
        + str(x_i.size) + ", theta.size = "  +  str(theta.size) + ")"
    
    if ytype=="neg11":
        assert np.abs(y_i)==1, "y_i must be either -1 or 1. (Currently, y_i = " + str(np.round(y_i,4)) + ")" 
    if ytype=="01":
        assert y_i==1 or y_i==0, "y_i must be either 0 or 1. (Currently, y_i = " + str(np.round(y_i,4)) + ")"
    

    if ytype=="01":
        prod = np.dot(theta, x_i)
        out = (sigmoid(prod) - y_i)*x_i
    
    
    if ytype=="neg11":
        prod = y_i*np.dot(theta, x_i)
        
        out = -sigmoid(-prod)*y_i*x_i
    
    
    assert out.size==x_i.size, "gradient should have same dimension as x_i " + \
        "(Currently, out.size = " + str(out.size) + ", x_i.size = " + str(x_i.size) + ")"
    
    return(out)


def SGD_A_S_logistic(x_i, theta, ytype="neg11"):
    # This is the SGD A_i, S_i matrix for a single data point (x_i, y_i).
    # x_i and theta are each a numpy array with size d, 
    # Notice that theta itself is an argument
    # Comes from Chen et al 2016 paper (their Example 2.2)
    # Careful: I'm not sure if this result changes for ytype = "01" (Chen paper assumes y in {-1,1})
    
    if ytype != "neg11":
        print("Careful: SGD_A_S_logistic assumes y in {-1,1}, but data is y in {0,1}")
    
    numer = np.outer(x_i, x_i)
    
    prod = np.dot(theta, x_i)
    denom_1 = 1 + np.exp(prod)
    denom_2 = 1 + np.exp(-prod)
    denom = denom_1*denom_2
    
    A_S = numer/denom
    
    return(A_S)