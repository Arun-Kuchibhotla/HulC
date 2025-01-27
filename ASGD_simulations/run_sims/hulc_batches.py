"""
Created on Sat Oct 21 10:50:11 2023

@author: selin

This file contains helper functions for producing the HulC confidence intervals
(https://arxiv.org/abs/2105.14577)

"""

import numpy as np


def P_B(B,D):
    #Returns the upper bound on the miscoverage probablility 
    # (equation 3 from HulC paper 9/23)
    #B = number of batches
    #D = delta; the median bias
    return((0.5-D)**B + (0.5 + D)**B)

def min_B(alpha, D):
    # Returns the smallest integer B>=1 such that P_B(B, D) <= alpha, 
    # according to Algorithm 1 (HulC paper 9/23)
    B = 1
    while True:
        p = P_B(B,D)
        if p <= alpha:
            break
        B += 1
    return B

def B_star(alpha, D):
    # Finds the batch size "B*" according to equation 4 (HulC paper 9/23)
    B = min_B(alpha, D)
    U = np.random.uniform(0, 1)
    numer = alpha - P_B(B, D)
    denom = P_B(B-1, D) - P_B(B,D)
    tau = numer/denom
    if U <= tau: return B-1
    return B

def divide_data_into_batches(data, B):
    # Calculate the size of each batch
    X,Y = data
    n = X.shape[0]
    
    indices_shuffled = np.arange(n)
    np.random.shuffle(indices_shuffled)
    X = X[indices_shuffled,:]
    Y = Y[indices_shuffled]
    
    
    batch_size =n // B  # Integer division to determine base batch size
    remainder = n % B  # Calculate the remaining data points
    batches = []

    # Split the data into batches, with the remaining data points distributed evenly
    for i in range(0, n, batch_size):
        #print(i)
        # Determine the batch size for the current iteration, considering the remainder
        #current_batch_size = batch_size + (1 if i // batch_size < remainder else 0)
        
        # Extract a batch of X and Y
        X_batch = X[i:i + batch_size]
        Y_batch = Y[i:i + batch_size]
        
        #Do not append a batch that is the remainder! This could be small!!
        if X_batch.shape[0]==batch_size:
            assert X_batch.shape[0]>100, "Batch size is too small: only " \
                + str(X_batch.shape[0]) + ". See: divide_data_into_batches()"
            
            # Append the batch to the list
            batches.append([X_batch, Y_batch])
    
    assert len(batches)==B, "Number of batches = " + str(len(batches)) +\
        "does not match desired length B = " + str(B) + "."
        
    return batches    

'''
def hulc_CI_linreg(data, alpha, c = .01):
    # x = data
    batch_size = B_star(alpha, 0)
    # param = [ASGD_CI_lin_reg(data) for b in range(batch_size)]
    batches = divide_data_into_batches(data,batch_size)
    
    d = data[0].shape[1]
    theta_bar_estimates = [ [] for i in range(d)]
    
    for b in batches:
        theta, theta_bar, A_bar, S_bar, n  = ASGD_CI_lin_reg(b, c =c, burn_in = True, burn_in_threshold = 1)
        #print(theta_bar)
        
        for i in range(d):
            theta_bar_estimates[i].append(theta_bar[i])
        

    Hulc_lb = np.min(theta_bar_estimates, axis = 1)
    Hulc_ub = np.max(theta_bar_estimates, axis = 1)
    
    Hulc_CI = []
    for i in range(d):
        CI = [Hulc_lb[i], Hulc_ub[i]]
        Hulc_CI.append(CI)

    return(Hulc_CI)
'''



