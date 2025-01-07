# -*- coding: utf-8 -*-
"""
Created on Mon Jan 6 2025

@author: selin

This file tests functions in ASGD_HulC_manual.py

"""
import numpy as np
import matplotlib.pyplot as plt

import os
run_locally=True
if run_locally:
    os.chdir("C://Users//selin//Dropbox//Hulc_simulations//hulc_simulations//HulC_python")
#os.chdir("//home//shcarter//Hulc//simulations//plots")

import gen_data as gld
import ASGD_HulC_manual as ahm   


#%%
#############################
### Linear regression #######
#############################

#%%
test_this = False
if test_this:
    
    XY = gld.gen_normal0_data(n=10000, d=6, cov_type="I") #cov_type = I, Toeplitz, EquiCorr
    #t_CI = tstat_CI_linreg_manual(XY, alpha=.05, output = "ASGD",\
    #                                                           c = 1, alpha_lr = .501, burn_in = False,\
    #                                                           burn_in_threshold = 1, \
    #                                                           initializer = False)
        
    hulc_CI = ahm.hulc_CI_linreg_manual(XY, alpha=.05, output = "ASGD",\
                                    c = 1, alpha_lr = .501, burn_in = False,\
                                    burn_in_threshold = 1,\
                                    initializer = False, epochs = 1,\
                                    return_thetas = True)
    



#%%
#############################
### Logistic regression #####
#############################

#%%
test_this = False
if test_this:
    np.random.seed(2)
    XY = gld.gen_normal0_logistic_data(n=1*10**4, d=5, cov_type="I") #cov_type = I, Toeplitz, or EquiCorr
    theta, theta_bar, A_bar, S_bar, n = ahm.ASGD_CI_log_reg_manual(XY, ytype = "neg11", output="ASGD", c = 1, alpha_lr = .501,\
                                                               burn_in = False,\
                                                           burn_in_threshold = 1, \
                                                               initializer = False,  plot_theta_dim=3,\
                                                        plot=True, verbose = False,\
                                                            epochs = 1, fixed_step = False)
    print("theta = " + str(theta))
    print("theta_bar =" + str(theta_bar))


test_this = False
if test_this:
    XY = gld.gen_normal0_logistic_data(n=1*10**4, d=5, cov_type="EquiCorr") #cov_type = I, Toeplitz, EquiCorr
    theta, theta_bar, A_bar, S_bar, n = ahm.ASGD_CI_log_reg_manual(XY, ytype = "neg11", output="WASGD", c = .01,alpha_lr = .501,\
                                                               burn_in = True,\
                                                           burn_in_threshold = 1, \
                                                    initializer = False,  plot_theta_dim=4,\
                                                        plot=True, verbose = False,\
                                                            epochs = 1, fixed_step = False)
        
test_this = False
if test_this:        
    XY = gld.gen_normal0_logistic_data(n=1*10**4, d=5, ytype="01", cov_type="I") #cov_type = I, Toeplitz, EquiCorr
    theta, theta_bar, A_bar, S_bar, n = ahm.ASGD_CI_log_reg_manual(XY, ytype = "01", output="ASGD", c = 1,alpha_lr = .501,\
                                                               burn_in = True,\
                                                           burn_in_threshold = 1, \
                                                    initializer = False,  plot_theta_dim=3,\
                                                        plot=True, verbose = False,\
                                                            epochs = 1, fixed_step = False)



test_this = False
if test_this:        
    XY = gld.gen_normal0_logistic_data(n=10**3, d=100, ytype="01", cov_type="EquiCorr") #cov_type = I, Toeplitz, EquiCorr
    theta, theta_bar, A_bar, S_bar, n = ahm.ASGD_CI_log_reg_manual(XY, ytype = "01", output="ASGD", c = 1,alpha_lr = .501,\
                                                               burn_in = True,\
                                                           burn_in_threshold = 1, \
                                                    initializer = False,  plot_theta_dim=3,\
                                                        plot=True, verbose = False,\
                                                            epochs = 1, fixed_step = False)


test_this = False
if test_this:
    D=5
    theta_bar_i = []
    theta_i = []
    for i in range(100):
        np.random.seed(i + 100)
    
        XY = gld.gen_normal0_logistic_data(n=20000, d=D, cov_type="I") #cov_type = I, Toeplitz, EquiCorr
        theta, theta_bar, A_bar, S_bar, n = ahm.ASGD_CI_log_reg_manual(XY, ytype = "neg11", output = "ASGD",\
                                                                   c = 1, alpha_lr = .501, burn_in = False,\
                                                                   burn_in_threshold = 1, \
                                                            initializer = False,  plot_theta_dim=0,\
                                                                plot=False, verbose = False,\
                                                                    epochs = 1, fixed_step = False)
        theta_bar_i.append(theta_bar)
        theta_i.append(theta)
        
    print(np.min(theta_bar_i, axis = 0))   
    print(np.max(theta_bar_i, axis = 0))   
    
    true_theta = np.linspace(0, 1, num=D)
    for d in range(D):
        
        plt.hist(np.array(theta_bar_i)[:,d], bins=8, edgecolor='black', \
                 color = "yellow", alpha = .5)  
        plt.axvline(true_theta[d], color='red', linestyle='-', alpha = .4)
        plt.title(r'(Logistic regression) Manual ASGD progress to estimate $\theta_' +\
                  str(d) + '$')
        plt.show()

#We can see that the hardest parameter to estimate is the last dimension (theta[4])




test_this = False
if test_this:
    D=100
    theta_bar_i = []
    theta_i = []
    for i in range(100):
        np.random.seed(i + 100)
    
        XY = gld.gen_normal0_logistic_data(n=10**3, d=D, cov_type="EquiCorr") #cov_type = I, Toeplitz, EquiCorr
        theta, theta_bar, A_bar, S_bar, n = ahm.ASGD_CI_log_reg_manual(XY, ytype = "neg11", output = "ASGD",\
                                                                   c = 1, alpha_lr = .501, burn_in = False,\
                                                                   burn_in_threshold = 1, \
                                                            initializer = False,  plot_theta_dim=0,\
                                                                plot=False, verbose = False,\
                                                                    epochs = 1, fixed_step = False)
        theta_bar_i.append(theta_bar)
        theta_i.append(theta)
        
    print(np.min(theta_bar_i, axis = 0))   
    print(np.max(theta_bar_i, axis = 0))   
    
    true_theta = np.linspace(0, 1, num=D)
    for d in range(D):
        
        plt.hist(np.array(theta_bar_i)[:,d], bins=8, edgecolor='black', \
                 color = "yellow", alpha = .5)  
        plt.axvline(true_theta[d], color='red', linestyle='-', alpha = .4)
        plt.title(rf'(Logistic regression) Manual ASGD progress to estimate $\theta_{{{d}}}$')
        plt.show()






