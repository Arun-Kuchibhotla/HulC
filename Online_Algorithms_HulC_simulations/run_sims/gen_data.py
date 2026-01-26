# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 10:31:15 2023

@author: selin

This file has functions that generate data from a linear regression model.

"""


import numpy as np
import matplotlib.pyplot as plt




#############################
### Linear regression #######
#############################

def gen_simple_reg_data(n, s2 = 200, b0 = 1, b1 = 5, x_lb = -5, x_ub = 15, plot = False):
    
    #This function returns an n-sized sample (x,y): 1 covariate (x) and 1 outcome (y),
    # where y is a linear function of x plus an error term. 
    #b0 is intercept parameter, b1 is slope parameter
    #x_lb, x_ub is for range of covariate data (generated from uniform distribution)
    #s2 is variance of error term.
    x = np.random.uniform(low=x_lb, high=x_ub, size=n)
    Y = b0 + b1*x + np.random.normal(loc=0, scale = np.sqrt(s2), size=n)
  
    if plot:
        # Create a scatter plot
        plt.scatter(x, Y)

        # Adding labels and a title
        plt.xlabel('Xl')
        plt.ylabel('Y')
        plt.title('Simple Scatter Plot')
    
        # Display the plot
        plt.show()
    
    #Crete data matrix (numpy array)
    Xt = np.vstack((np.ones(n), x))
    X = np.transpose(Xt)
    out = [X, Y]
    
    return(out)



def gen_reg_data_Chen(n, d, x_lb = -5, x_ub = 15, torch = False):
    
    # Following Chen, Lee et al's paper (Section 6):
    # This function returns an n-sized sample (X,y): 1 column of 1's (for intercept) and
    #   d-1 covariates (X matrix~ n x d) and 1 outcome (y),
    # where y is a linear function of x plus an error term. 
    #According to the paper, these are the linear regression parameters:
    # True parameter vector: theta is d-dimensional vector linearly spaced between 0 and 1.
    # sigma = 1, so covariance is identity matrix.
    # It's not clear to me how they generated the x values (ie., the independent RV).
    #    Hence I'm letting the lower bound (x_lb) and upper bound (x_ub) for x be flexible, then 
    #    I generate x data uniformly over this interval.
    # If torch = False, then the column of 1's (for intercept) is removed, since
    # the torch.nn.linear() function automatically adds the intercept column
    
    
    #Create X matrix
    X = np.random.uniform(low=x_lb,high=x_ub,size=(n,d))
    X[:,0] = 1
    
    #Create the theta vector
    theta = np.linspace(0, 1, num=d)
    
    #Generate Y (note: np.random.multivariate_normal() is super slow, so
    # instead we use a forloop for each value of y using np.random.normal()
    # )
    mu = np.matmul(X, theta)
    #Y = mu + np.random.multivariate_normal(np.zeros(n), np.identity(n), 1)[0]
    Y = np.zeros(n)
    for i in range(n):
        Y[i] = mu[i] + np.random.normal(loc=0, scale = 1)
    
        
    plot = False
    if plot:
        # Create a scatter plot
        plt.scatter(X[:,1], Y)

        # Adding labels and a title
        plt.xlabel('X_1')
        plt.ylabel('Y')
        plt.title('Scatter Plot')
    
        # Display the plot
        plt.show()
    
    
    if torch:
        X = X[:,1:]
    
    out = [X, Y]
    
    return(out)




def gen_normal0_data(n, d, start = 0, stop = 1, cov_type="I", torch = False):
    
    # This function returns an n-sized sample (X,y): 1 column of 1's (for intercept) and
    #   d-1 covariates (X matrix~ n x d) and 1 outcome (y),
    # where y is a linear function of x plus an error term. 
    #According to the Chen et al paper, the linear regression parameters are equally spaced
    # between 0 and 1, that is: theta = np.linspace(start, stop, num=d)
    
    
    if cov_type=="I":
        #Create X matrix
        X = np.random.normal(loc=0.0, scale=1.0, size=(n,d))
        X[:,0] = 1
    
    if cov_type=="Toeplitz":
        Sigma = np.zeros((d-1,d-1))
        r= 0.5
        for row in range(d-1):
            for col in range(d-1):
                Sigma[row,col] = r**np.abs(row-col)
        X = np.random.multivariate_normal(mean=np.zeros(d-1), cov=Sigma, size = n)
        X = np.hstack((np.ones((n,1)), X))
    
    if cov_type=="EquiCorr":
        r = 0.2
        Sigma = r*np.ones((d-1,d-1))
        for row in range(d-1):
            Sigma[row,row] = 1
        X = np.random.multivariate_normal(mean=np.zeros(d-1), cov=Sigma, size = n)
        X = np.hstack((np.ones((n,1)), X))
    
    #Create the theta vector
    theta = np.linspace(start, stop, num=d)
    
    #Generate Y (note: np.random.multivariate_normal() is super slow, so
    # instead we use a forloop for each value of y using np.random.normal()
    # )
    mu = np.matmul(X, theta)
    #Y = mu + np.random.multivariate_normal(np.zeros(n), np.identity(n), 1)[0]
    Y = np.zeros(n)
    for i in range(n):
        Y[i] = mu[i] + np.random.normal(loc=0, scale = 1)
    
        
    plot = False
    if plot:
        # Create a scatter plot
        plt.scatter(X[:,1], Y)

        # Adding labels and a title
        plt.xlabel('X_1')
        plt.ylabel('Y')
        plt.title('Scatter Plot')
    
        # Display the plot
        plt.show()
    
    
    if torch:
        X = X[:,1:]
    
    out = [X, Y]
    
    return(out)



#############################
### Logistic regression #######
#############################


def sigmoid(s):
    #s is a scalar
    denom = 1 + np.exp(-s)
    
    return 1/denom
    

def gen_normal0_logistic_data(n, d, start = 0, stop = 1, ytype = "neg11", cov_type = "I", torch = False):
    
    # This function returns an n-sized sample (X,y): 1 column of 1's (for intercept) and
    #   d-1 covariates (X matrix~ n x d) and 1 outcome (y),
    # where y is a linear function of x plus an error term. 
    #According to the Chen et al paper, the linear regression parameters are equally spaced
    # between 0 and 1, that is: theta = np.linspace(start, stop, num=d)
    # # ytype = "neg11" means y_i is binary {-1, 1}. type = "01" means y_i is binary {0, 1} 
    #
    # cov_type="I" means x's are drawn independently from N(0,1) distribution.
    # cov_type="Toeplitz" means Sigma(i,j) = 0.5^{|i-j|} (using Chen simulation as reference)
    # cov_type = "EquiCorr" means Sigma(i,j) = 0.2 for i != j, and 1 if i = j
    
    if cov_type=="I":
        #Create X matrix
        X = np.random.normal(loc=0.0, scale=1.0, size=(n,d))
        X[:,0] = 1
    
    if cov_type=="Toeplitz":
        Sigma = np.zeros((d-1,d-1))
        r= 0.5
        for row in range(d-1):
            for col in range(d-1):
                Sigma[row,col] = r**np.abs(row-col)
        X = np.random.multivariate_normal(mean=np.zeros(d-1), cov=Sigma, size = n)
        X = np.hstack((np.ones((n,1)), X))
    
    if cov_type=="EquiCorr":
        r = 0.2
        Sigma = r*np.ones((d-1,d-1))
        for row in range(d-1):
            Sigma[row,row] = 1
        X = np.random.multivariate_normal(mean=np.zeros(d-1), cov=Sigma, size = n)
        X = np.hstack((np.ones((n,1)), X))
    
    
    #Create the theta vector
    theta = np.linspace(start, stop, num=d)
    
    #Generate Y (note: np.random.multivariate_normal() is super slow, so
    # instead we use a forloop for each value of y using np.random.normal()
    # )
    prod = np.matmul(X, theta)
    mu = sigmoid(prod)
    #Y = mu + np.random.multivariate_normal(np.zeros(n), np.identity(n), 1)[0]
    Y = np.zeros(n)
    for i in range(n):
        Y[i] = np.random.binomial(1, mu[i])
        if ytype=="neg11":
            if Y[i]==0: #Convert 0's to -1
                Y[i] = -1
    
        
    plot = False
    if plot:
        # Create a scatter plot
        plt.scatter(X[:,1], Y)

        # Adding labels and a title
        plt.xlabel('X_1')
        plt.ylabel('Y')
        plt.title('Scatter Plot (Logistic Regression)')
    
        # Display the plot
        plt.show()
    
    
    if torch:
        X = X[:,1:]
    
    out = [X, Y]
    
    return(out)




