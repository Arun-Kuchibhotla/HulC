# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 10:50:11 2023

@author: Selina Carter

This file contains functions that produce confidence intervals for parameter
theta in linear or logistic regression.

Functions provided are for 4 confidence interval techniques:
    - sandwich confidence interval (Wald) for linear and logistic regression (used as a baseline)
    - ASGD plug-in confidence interval (see Chen et al 2016,
                                        "Statistical Inference for Model
                                        Parameters in Stochastic Gradient Descent")
    - HulC on batch of ASGD estimators (see Kumar et al 2021, "The HulC:
                                        Confidence Regions from Convex Hulls")
    - t-statistic-based confidence interval on batch of ASGD estimators       
    

To 

"""
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D #Creates a custom legend in plots
from datetime import datetime

import os
run_locally=True
if run_locally:
    os.chdir("C://Users//selin//Dropbox//Hulc_simulations//hulc_simulations//HulC_python//ASGD_simulations//run_sims")
#os.chdir("//home//shcarter//Hulc//simulations//plots")

import hulc_batches as hulc
import ASGD_Chen_functions as chen  # contains: SGD_grad_linear(theta, x_i, y_i), 
                                    #           SGD_hessian_linear(x_i),
                                    #           eta_t(t, c = .01, alpha=.501)

#%%
#############################
### Linear regression #######
#############################


def ASGD_CI_lin_reg_manual(data, output = "ASGD", c = .01,  alpha_lr = .501, 
                           burn_in=False, burn_in_threshold = 1, \
                           initializer = False, plot = False, plot_theta_dim = 0, \
                               verbose = False,
                               epochs = 1, fixed_step = False,
                               shift_k = True,
                               plot_footnote=True):
    # Linear regression
    # Performs (W)ASGD with confidence intervals (plug-in method by Chen & Lee) - see Algorithm 7 in Overleaf https://www.overleaf.com/project/652d5761fbda66b388df8d72)
    # This function outputs 5 components:
    #          theta_SGD: the final iterate of theta using SGD.
    #          theta_ASGD: the final iterate of ASGD (if output=="ASGD") or of WASGD (if output=="WASGD")
    #          A_bar: the final iterate of the (W)ASGD estimated matrix A using Chen & Lee plug-in method.
    #          S_bar: the final iterate of the (W)ASGD estimated matrix S using Chen & Lee plug-in method.
    #          n: sample size from data (this is needed in order to calculate the Chen & Lee plug-in covariance)
    # data: A list [X, Y] where X  is an n x d matrix of real numbers and Y is an array of real numbers of length n.
    # output: Either "ASGD" (Averaged SGD) or "WASGD" (Adaptive Weighted ASGD, by Wei et al 2023) - see Algorithm 9 in in Overleaf https://www.overleaf.com/project/652d5761fbda66b388df8d72)
    # c: The "c" parameter in the eta_t() function used to calculate the step size.
    # alpha_lr: The "alpha" parameter in the eta_t() function used to calculate the step size.
    # burn_in: If True, then (W)ASGD iterates ("theta_bar") are calculated starting at iterate = sqrt(n), ending at step n.
    #            (ie., throw away the "burn-in")
    #          If False, then (W)ASGD iterates ("theta_bar") start being calculated at iterate = 1, ending at step n.
    #             (ie, don't throw away the "burn-in")
    # burn_in_threshold: This is only activated if burn_in=True. This should be a positive number.
    #     For example, burn_in_threshold = 3 means that (W)ASGD iterates ("theta_bar") starts being calculated at
    #     3*sqrt(n) to n steps. The default is 1, as suggested by Chen & Lee.
    # Initializer: If True, then iterate 0 for SGD ("theta") is initialized "intelligently", 
    #    i.e., by first setting theta_hat=0 and running vanilla SGD using a fixed step size .001 for n/3 steps.
    #               (Note that this method was recommended by Chen & Lee).
    #              If False, theta_hat is simply initialized to 0.
    # plot: If True, then plots are produced that track progress of (W)ASGD ("theta_bar")
    #       with respect to time steps t.
    # plot_theta_dim: The coordinate of theta to plot, ex., theta_to_plot=0 plots the intercept.
    # verbose: If True, then prints progress of theta every 1000th time step.
    #          If False, then doesn't print progress.
    # epochs: Should be an integer. The number of epochs to run the (W)ASGD algorithm.
    #         Default is 1. Note that the "traditional" (W)ASGD algorithm uses 1 epoch.
    # fixed_step: If True, means that an adaptive step size using chen.eta_t() function is ignored.
    #             Instead, (W)ASGD algorithm is performed with step size = c.
    #            If False, then chen.eta_t() function is used. False is the default.
    # shift_k = True means shift "pythonic" dimensions 0...d-1 to 1,..., d for plots, so smallest dimesion is 1 instead of 0.
    #           False means keep "pythonic" dimension 0...d-1
    # plot_footnote = True means the plot shows hyperparameters at the bottom. =False turns off this feature.
       
    init_value = 0.0 #This is value that will make the vector of init values for theta
    
    X = data[0]
    Y = data[1]
    
    n = X.shape[0]
    d = X.shape[1]
    
    assert burn_in_threshold**2 <= n, "burn_in_threshold must be in range (0, " \
        + str(np.int32(np.floor(np.sqrt(n)))) + "). Currently, n = " + str(n) + \
            " and start_t = " + str(np.int32(np.floor(burn_in_threshold*np.sqrt(n)))) + "."
    
    
    if initializer: #Performs "intelligent" starting values of theta.
    
        #Initialize theta to 0
        theta = init_value*np.ones(d)

        for tt in range(1, np.int32(np.floor(n/3))):  
            
            step = .001 #This is a fixed step size, and we perform vanilla SGD for the "intelligent"
                        # initialization.
           
            #SGD
            theta_new = theta - step*chen.SGD_grad_linear(theta, X[tt-1], Y[tt-1])
            theta = theta_new.copy()
            #print(theta)
        if verbose:
            print("Starting theta = " + str(theta))
    else:
        #Initialize theta to 0
        theta = init_value*np.ones(d)
        if verbose:
            print("Starting theta = " + str(theta))
        
   
    
    # We create an array that will store minimum and maximum estimates
    # for theta across the iterates. This is only a diagnostic aid in case
    # conververgence is not achieved.
    min_theta_bar = init_value*np.ones(d)
    max_theta_bar = init_value*np.ones(d)
    
    
    if burn_in:
        start_t = np.int32(np.floor(burn_in_threshold*np.sqrt(n)))
        if verbose:
            print("Burn-in samples before time step t = " + str(start_t) + \
                  " will not be used to calculate ASGD estimator.")
    else: start_t = 1
    
    
    if plot:
        #Store ASGD and SGD estimates in list
        theta_bar_t = []
        theta_hat_t = []
        theta_bar_W_t = []
        max_abs_x = []
        max_abs_x_i = 0
    
    T = 0 #Tracker that also uses epochs
    for epoch in range(epochs):
        
        #Shuffle data randomly at each epoch (to avoid cycles)
        indices_shuffled = np.arange(n)
        np.random.shuffle(indices_shuffled)
        X = X[indices_shuffled,:]
        Y = Y[indices_shuffled]
        
        for tt in range(1, n+1):
            T = T + 1
            if fixed_step:
                step = c
            else:
                if output=="ASGD":
                    step = chen.eta_t(T, c=c, alpha = alpha_lr)
                else: #for WASGD
                    step = chen.eta_t(T, c=1, alpha = alpha_lr)
           
            #SGD update
            theta_new = theta - step*chen.SGD_grad_linear(theta, X[tt-1], Y[tt-1])
            
            
            if np.isnan(np.sum(theta_new)) or np.isinf(np.sum(theta_new)):
                
                print("ASGD_CI_lin_reg_manual(): theta_new contains NaN or inf: theta_new = " \
                + str(theta_new) + " at time step T = " + str(T))
                print("parameters: output = " + output + ", c=" + str(c) +\
                      ", alpha_lr =" + str(alpha_lr) + ", burn_in=" + str(burn_in) +\
                          ", burn_in_threshold=" + str(burn_in_threshold) +\
                              ", initializer=" + str(initializer) + ", epochs=" + str(epochs) +\
                                  ", fixed_step=" + str(fixed_step))
                
                #Create junk vectors to return    
                theta_bar = theta_new.copy()
                A_bar = chen.SGD_hessian_linear(X[tt-1])
                g_theta_bar_t2 = chen.SGD_grad_linear(theta_bar, X[tt-1], Y[tt-1]) 
                S_bar = np.outer(g_theta_bar_t2, g_theta_bar_t2)
                return theta, theta_bar, A_bar, S_bar, n
            
            #Burn-in condition: only update theta_bar if burn-in is over
            if T==start_t:
                #Initialize ASGD estimate (theta_bar) using SGD estimate (theta_new)
                theta_bar = theta_new.copy()
                
                #Initialize matrices A_bar and S_bar
                A_bar = chen.SGD_hessian_linear(X[tt-1])
                g_theta_bar_t2 = chen.SGD_grad_linear(theta_bar, X[tt-1], Y[tt-1]) 
                S_bar = np.outer(g_theta_bar_t2, g_theta_bar_t2)
                
                assert np.sum(theta_bar)!=0, "theta_bar is initlized to 0 at time step T = " \
                    + str(T) +"; check burn-in condition in ASGD_CI_lin_reg_manual()."
                assert np.sum(A_bar)!=0, "A_bar is initlized to 0 at time step T = " \
                    + str(T) +"; check burn-in condition in ASGD_CI_lin_reg_manual()."
                assert np.sum(S_bar)!=0, "S_bar is initlized to 0 at time step T = " \
                    + str(T) +"; check burn-in condition in ASGD_CI_lin_reg_manual()."
                    
                #Initialize WASGD estimate (theta_bar_AW) using SGD estimate (theta_new)
                if output=="WASGD":
                    theta_bar_W = theta_new.copy()
                
                
            #Burn-in condition: only update theta_bar if burn-in is over
            if T > start_t:
                
                # Update ASGD estimate (theta_bar_new) using old ASGD estimate
                # (theta_bar) and SGD estimate (theta_new)
                theta_bar_new = (1/T)*(theta_new-theta_bar) + theta_bar
            
                #Update matrices A_bar and S_bar
                A_bar_new = (1/T)*(chen.SGD_hessian_linear(X[tt-1])-A_bar) + A_bar
                g_theta_bar = chen.SGD_grad_linear(theta_bar, X[tt-1], Y[tt-1])  
                S_bar_new = (1/T)*(np.outer(g_theta_bar, g_theta_bar)-S_bar) + S_bar
            
                
                theta_bar = theta_bar_new.copy()
                A_bar = A_bar_new.copy()
                S_bar = S_bar_new.copy()
            
            
                #Adaptive weighted ASGD (WASGD) from Wei et al 2023:
                if output=="WASGD":
                    
                    theta_bar_W_new = ((T-1)/T)*theta_bar_W + ((1-T**alpha_lr)/T)*theta \
                        + (T**(alpha_lr-1))*theta
                    theta_bar_W = theta_bar_W_new.copy()
                     
            
                #Keep track of min and max theta_bars (to check for "wildness")
                for i in range(d):
                    min_theta_bar[i] = np.min([min_theta_bar[i], theta_bar_new[i]])
                    max_theta_bar[i] = np.max([max_theta_bar[i], theta_bar_new[i]])
                    
                if (T-1)%1000==0 and verbose:
                    #print("     grad = " + str(g_theta_bar))
                    print("********** Time step T = " + str(T) + " ***********")
                    print("theta_bar_new = " + str(theta_bar_new) )
                    print("theta_new = " + str(theta_new) )
                    if output=="WASGD":
                        print("theta_bar_W_new = " + str(theta_bar_W_new) )
            
                    
            #Update SGD estimate outside of burn-in condition
            theta = theta_new.copy()
            
            
            if plot:
                
                p = plot_theta_dim
                
                if T >= start_t:
                    
                    #This plot tracks progress of SGD and ASGD estimate over time steps T
                    #ax.plot(T, theta[p], marker = "," , color = 'blue', linestyle='-')
                    #ax.plot(T, theta_bar[p], marker = ",", linestyle='--', color = 'red')
                    theta_bar_t.append(theta_bar[p])
                    theta_hat_t.append(theta[p])
                    if output=="WASGD":
                        theta_bar_W_t.append(theta_bar_W[p])
                    
                    
                    #This plot compares magnitude of ASGD estimate compared to max(|X_t|), 
                    # which is the theoretical bound for the |ASGD| estimate.
                    
                    if np.absolute(X[tt-1][p]) > max_abs_x_i:
                        max_abs_x_i = np.absolute(X[tt-1][p])
                    max_abs_x.append(max_abs_x_i)
                    #abs_theta_bar = np.absolute(theta_bar[p])
                    #ax2.plot(T, abs_theta_bar, marker = ",", linestyle='--', color = 'red')
                    #ax2.fill_between(T, (abs_theta_bar-.5), (abs_theta_bar+.5), color='b', alpha=.1)

    
    if plot:
        #This plot tracks progress of ASGD vs SGD over time steps t
        
        # Print true theta for reference
        true_theta_p = np.linspace(0, 1, num=d)[p]
        
        
        #SGD line:
        plt.plot(np.arange(start_t-1,T,1), theta_hat_t, marker = "," , color = '#7f7f7f', linestyle='-')
        #ASGD line:
        plt.plot(np.arange(start_t-1,T,1), theta_bar_t, marker = ",", linestyle='-', color = 'red')
        if output=="WASGD":
            #WASGD line:
            plt.plot(np.arange(start_t-1,T,1), theta_bar_W_t, marker = ",", linestyle='--', color = 'blue')
        #plt.ylim((min_theta_bar[p]-2, max_theta_bar[p]+2))
        plt.axhline(0, color='black', linestyle='-', alpha = .4)
        plt.axvline(0, color='black', linestyle='-', alpha = .4)
    
        plt.xlabel('t')

        N = '%.1E' % n #puts n into scientific notation (shorter)
        step_info = r', step $\eta =' + str(c) + '$' if  fixed_step else  r', step $\eta_t =' + str(c) + 't^{-' + str(alpha_lr) +'}$'
            
        if shift_k==False: #Keeps "pythonic" dimensions 0,...,d-1 for plotting
            p_to_print = p
        if shift_k==True: #Prints dimensions 1,...,d in titles
            p_to_print= p+1
        
        plot_text_result = False #If true, puts final iterate inside graph in addition to legend, but sometimes not the best placement of the annotation.
        if output=="WASGD":
            WASGD_label = r"WASGD $\hat{\theta}_" + str(p_to_print) + "$ = " + str(np.round(theta_bar_W[p], 3))
            if plot_text_result==True:
                plt.text(x=T-.3*T, y=(max_theta_bar[p]-min_theta_bar[p])/1, s=WASGD_label, fontsize=12, color='blue')
    
    
        ASGD_label = r"ASGD $e^\top_" + str(p_to_print) + r"  \bar{\theta}_T$ =" + str(np.round(theta_bar[p], 3))
        SGD_label = r"SGD $e^\top_" + str(p_to_print) + r"  \hat{\theta}_T$ =" + str(np.round(theta[p], 3))
        #ASGD_label = r"ASGD $\bar{\theta}_" + str(p_to_print) + "$ = " + str(np.round(theta_bar[p], 3))
        #SGD_label = r"SGD $\hat{\theta}_" + str(p_to_print) + "$ = " + str(np.round(theta[p], 3))
        
        if plot_text_result==True:
            plt.text(x=T-.3*T, y=(max_theta_bar[p]-min_theta_bar[p])/2, s=ASGD_label, fontsize=12, color='red')
            plt.text(x=T-.3*T, y=(max_theta_bar[p]-min_theta_bar[p])/30, s=SGD_label, fontsize=12, color='k')
        
        
        #Legend
        if output!="WASGD":
            custom_lines = [Line2D([0], [0], color="#545454", lw=1),
                            Line2D([0], [0], color="red", lw=1)]
            
            plt.legend(custom_lines, [SGD_label, ASGD_label], labelcolor='linecolor')
        
        if output=="WASGD":
            custom_lines = [Line2D([0], [0], color="#545454", lw=1),
                            Line2D([0], [0], color="red", lw=1),
                            Line2D([0], [0], color="blue", lw=1)]
            
            plt.legend(custom_lines, [SGD_label, ASGD_label, WASGD_label], labelcolor='linecolor')
        

        plt.ylabel(r'$e^\top_' + str(p_to_print) + r'\theta_t$')
        plt.title(r'(Linear regression) ASGD progress to estimate $e^\top_' + str(p_to_print) +\
                  r'   \theta_{\infty}$=' + str(true_theta_p))
        
        if plot_footnote==True:
            plt.annotate("n = " + N + ", epochs = " + str(epochs) + ", t0 = " + str(start_t) + ", " +  \
                         "initializer = " + str(initializer) + ", dim = " + str(d) + step_info,
                    xy = (1.0, -0.2),
                    xycoords='axes fraction',
                    ha='right',
                    va="center",
                    color = "gray",
                    fontsize=10)
        
        
        
        plt.savefig('plots//ASGD progress//ASGD progress_' + \
                    datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + ".pdf",\
                        bbox_inches='tight', dpi=500)
        plt.show()
                
        
    
    if output=="WASGD":
        theta_bar = theta_bar_W.copy()
    
    
    
    #Rename outputs
    theta_SGD = theta
    theta_ASGD = theta_bar  #If output=="WASGD", then theta_ASGD is in fact WASGD's output
    
    
    return theta_SGD, theta_ASGD, A_bar, S_bar, n







def hulc_CI_linreg_manual(data, alpha, output = "ASGD", c = .01, alpha_lr = .501, burn_in = False,
                          burn_in_threshold = 1,\
                          initializer = False, epochs = 1,
                          return_thetas = True):
    # x = data
    batch_size = hulc.B_star(alpha, 0)
    # param = [ASGD_CI_lin_reg(data) for b in range(batch_size)]
    batches = hulc.divide_data_into_batches(data, batch_size)
    print("batches B = " + str(len(batches)))
    d = data[0].shape[1]
    theta_bar_estimates = [ [] for i in range(d)]
    
    for batch in batches:
        
        assert batch[0].shape[0]>0, "Batch size is too small: only " \
            + str(batch[0].shape[0]) + ". See: hulc_CI_linreg_manual()"
        
        theta, theta_bar, A_bar, S_bar, n = ASGD_CI_lin_reg_manual(batch, output = output, c = c, \
                                                                   alpha_lr = alpha_lr,\
                                                                   burn_in = burn_in,\
                                                                   burn_in_threshold = burn_in_threshold, \
                                                            initializer = initializer,  plot_theta_dim=0,\
                                                                plot=False, verbose = False,\
                                                                    epochs = epochs)
            #print(theta_bar)
        
        if output=="ASGD" or output=="WASGD":
            theta_bar = theta_bar
        elif output=="SGD":
            theta_bar = theta.copy()
        
        for i in range(d):
            theta_bar_estimates[i].append(theta_bar[i])
        

    Hulc_lb = np.min(theta_bar_estimates, axis = 1)
    Hulc_ub = np.max(theta_bar_estimates, axis = 1)
    
    Hulc_CI = []
    for i in range(d):
        CI = [Hulc_lb[i], Hulc_ub[i]]
        Hulc_CI.append(CI)

    if return_thetas==True:
        return Hulc_CI, theta_bar_estimates
    else:
        return Hulc_CI



def tstat_CI_linreg_manual(data, alpha, output = "ASGD", c = .01, alpha_lr = .501, burn_in = False,
                          burn_in_threshold = 1,\
                          initializer = False, epochs = 1):
    # x = data
    batch_size = hulc.B_star(alpha, 0)
    # param = [ASGD_CI_lin_reg(data) for b in range(batch_size)]
    batches = hulc.divide_data_into_batches(data, batch_size)
    #print(len(batches))
    d = data[0].shape[1]
    theta_bar_estimates = [ [] for i in range(d)]
    
    for batch in batches:
        
        assert batch[0].shape[0]>0, "Batch size is too small: only " \
            + str(batch[0].shape[0]) + ". See: hulc_CI_linreg_manual()"
        
        theta, theta_bar, A_bar, S_bar, n = ASGD_CI_lin_reg_manual(batch, output = output, c = c, \
                                                                   alpha_lr = alpha_lr,\
                                                                   burn_in = burn_in,\
                                                                   burn_in_threshold = burn_in_threshold, \
                                                            initializer = initializer,  plot_theta_dim=0,\
                                                                plot=False, verbose = False,\
                                                                    epochs = epochs)
            #print(theta_bar)
        
        if output=="ASGD" or output=="WASGD":
            theta_bar = theta_bar
        elif output=="SGD":
            theta_bar = theta.copy()
        
        for i in range(d):
            theta_bar_estimates[i].append(theta_bar[i])
    
    #Compute critical value for t_{alpha/2}
    t_threshold = t.ppf(q=1-alpha/2, df=batch_size-1) #from scipy.stats package
    
    
    ###Denominator###
    theta_bar_estimates = np.array(theta_bar_estimates)
    theta_bar_bar = np.mean(theta_bar_estimates, axis = 1)
    theta_bar_bar_matrix = np.repeat(theta_bar_bar, batch_size).reshape(theta_bar_estimates.shape)
    
    sq_devs = (theta_bar_estimates - theta_bar_bar_matrix)**2
    sum_sq_devs = np.sum(sq_devs, axis = 1)
    s_hat = np.sqrt((1/(batch_size-1))*sum_sq_devs)
    
    deviation = t_threshold*s_hat/np.sqrt(batch_size)
        
    #Lower bound
    lb = theta_bar_bar - deviation
    #Upper bound
    ub = theta_bar_bar + deviation

    t_CI = np.array([lb, ub]).reshape((2, d))
    t_CI = t_CI.transpose(1,0)

    return(t_CI)




#%%
#############################
### Logistic regression #####
#############################




def ASGD_CI_log_reg_manual(data, ytype = "neg11", output = "ASGD", c = .01,  alpha_lr = .501, 
                           burn_in=False, burn_in_threshold = 1, \
                           initializer = False, plot = False, plot_theta_dim = 0, \
                               verbose = False,
                               epochs = 1, fixed_step = False,
                               shift_k=True,
                               plot_footnote=True):
    # Logistic regression
    # Performs (W)ASGD with confidence intervals (plug-in method by Chen & Lee) - see Algorithm 7 in Overleaf https://www.overleaf.com/project/652d5761fbda66b388df8d72)
    # This function outputs 5 components:
    #          theta_SGD: the final iterate of theta using SGD.
    #          theta_ASGD: the final iterate of ASGD (if output=="ASGD") or of WASGD (if output=="WASGD")
    #          A_bar: the final iterate of the (W)ASGD estimated matrix A using Chen & Lee plug-in method.
    #          S_bar: the final iterate of the (W)ASGD estimated matrix S using Chen & Lee plug-in method.
    #          n: sample size from data (this is needed in order to calculate the Chen & Lee plug-in covariance)
    # data: A list [X, Y] where X  is an n x d matrix of real numbers and Y is an array of length n.
    # ytype: outcome type of Y (type="neg11" means Y is in {-1, 1}, type="01" means y is in {0, 1})
    # output: Either "ASGD" (Averaged SGD) or "WASGD" (Adaptive Weighted ASGD, by Wei et al 2023) - see Algorithm 9 in in Overleaf https://www.overleaf.com/project/652d5761fbda66b388df8d72)
    # c: The "c" parameter in the eta_t() function used to calculate the step size.
    # alpha_lr: The "alpha" parameter in the eta_t() function used to calculate the step size.
    # burn_in: If True, then (W)ASGD iterates ("theta_bar") are calculated starting at iterate = sqrt(n), ending at step n.
    #            (ie., throw away the "burn-in")
    #          If False, then (W)ASGD iterates ("theta_bar") start being calculated at iterate = 1, ending at step n.
    #             (ie, don't throw away the "burn-in")
    # burn_in_threshold: This is only activated if burn_in=True. This should be a positive number.
    #     For example, burn_in_threshold = 3 means that (W)ASGD iterates ("theta_bar") starts being calculated at
    #     3*sqrt(n) to n steps. The default is 1, as suggested by Chen & Lee.
    # Initializer: If True, then iterate 0 for SGD ("theta_hat") is initialized "intelligently", 
    #    i.e., by first setting theta_hat=0 and running vanilla SGD using a fixed step size .001 for n/3 steps.
    #               (Note that this method was recommended by Chen & Lee).
    #              If False, theta_hat is simply initialized to 0.
    # plot: If True, then plots are produced that track progress of (W)ASGD ("theta_bar")
    #       with respect to time steps t.
    # plot_theta_dim: The coordinate of theta to plot, ex., theta_to_plot=0 plots the intercept.
    # verbose: If True, then prints progress of theta every 1000th time step.
    #          If False, then doesn't print progress.
    # epochs: Should be an integer. The number of epochs to run the (W)ASGD algorithm.
    #         Default is 1. Note that the "traditional" (W)ASGD algorithm uses 1 epoch.
    # fixed_step: If True, means that an adaptive step size using chen.eta_t() function is ignored.
    #             Instead, (W)ASGD algorithm is performed with step size = c.
    #            If False, then chen.eta_t() function is used. False is the default.
    # shift_k = True means shift "pythonic" dimensions 0...d-1 to 1,..., d for plots, so smallest dimesion is 1 instead of 0.
    #           False means keep "pythonic" dimension 0...d-1
    # plot_footnote = True means the plot shows hyperparameters at the bottom. =False turns off this feature.
    
    X = data[0]
    Y = data[1]
    
    n = X.shape[0]
    d = X.shape[1]
    
   
    
    assert burn_in_threshold**2 <= n, "burn_in_threshold must be in range (0, " \
        + str(np.int32(np.floor(np.sqrt(n)))) + "). Currently, n = " + str(n) + \
            " and start_t = " + str(np.int32(np.floor(burn_in_threshold*np.sqrt(n)))) + "."
    
    
    if initializer: #Performs "intelligent" starting values of theta.
        
        #Initialize theta to 0
        theta = np.zeros(d)

        for tt in range(1, np.int32(np.floor(n/3))):  
            
            step = .001 #This is a fixed step size, and we perform vanilla SGD for the "intelligent"
                        # initialization.
           
            #SGD
            theta_new = theta - step*chen.SGD_grad_logistic(theta, X[tt-1], Y[tt-1], ytype=ytype)
            theta = theta_new.copy()

        if verbose:
            print("Starting theta = " + str(theta))
    else:
        #Initialize theta to 0
        theta = np.zeros(d)
        if verbose:
            print("Starting theta = " + str(theta))
        
    
    
    # We create an array that will store minimum and maximum estimates
    # for theta across the iterates. This is only a diagnostic aid in case
    # conververgence is not achieved.
    min_theta_bar = np.zeros(d)  
    max_theta_bar = np.zeros(d)  
    
    
    if burn_in:
        start_t = np.int32(np.floor(burn_in_threshold*np.sqrt(n)))
        if verbose:
            print("Burn-in samples before time step t = " + str(start_t) + \
                  " will not be used to calculate ASGD estimator.")
    else: start_t = 1
    
    
    if plot:
        #Store ASGD and SGD estimates in list
        theta_bar_t = []
        theta_hat_t = []
        theta_bar_W_t = []

    
    T = 0 #Time step tracker
    for epoch in range(epochs):
        
        #Shuffle data randomly at each epoch (to avoid cycles)
        indices_shuffled = np.arange(n)
        np.random.shuffle(indices_shuffled)
        X = X[indices_shuffled,:]
        Y = Y[indices_shuffled]
        
        for tt in range(1, n+1):
            T = T + 1
            if fixed_step:
                step = c
            else:
                if output=="ASGD":
                    step = chen.eta_t(T, c=c, alpha = alpha_lr)
                else: #for WASGD
                    step = chen.eta_t(T, c=1, alpha = alpha_lr)
           
            #SGD update
            theta_new = theta - step*chen.SGD_grad_logistic(theta, X[tt-1], Y[tt-1], ytype = ytype)
            
            
            if np.isnan(np.sum(theta_new)) or np.isinf(np.sum(theta_new)):
                print("ASGD_CI_lin_reg_manual(): theta_new contains NaN or inf: theta_new = " \
                 + str(theta_new) + " at time step T = " + str(T))
                print("parameters: output = " + output + ", ytype=" + ytype + ", c=" + str(c) +\
                       ", alpha_lr =" + str(alpha_lr) + ", burn_in=" + str(burn_in) +\
                           ", burn_in_threshold=" + str(burn_in_threshold) +\
                               ", initializer=" + str(initializer) + ", epochs=" + str(epochs) +\
                                   ", fixed_step=" + str(fixed_step))
                 
                #Create junk vectors to return    
                theta_bar = theta_new.copy()
                A_bar = chen.SGD_A_S_logistic(X[tt-1], theta_bar)
                S_bar = A_bar.copy()
                return theta, theta_bar, A_bar, S_bar, n
            
            #Burn-in condition: only update theta_bar if burn-in is over
            if T==start_t:
                #Initialize ASGD estimate (theta_bar) using SGD estimate (theta_new)
                theta_bar = theta_new.copy()
                
                #Initialize matrices A_bar and S_bar
                A_bar = chen.SGD_A_S_logistic(X[tt-1], theta_bar)
                S_bar = A_bar.copy()
                
                assert np.sum(theta_bar)!=0, "theta_bar is initlized to 0 at time step T = " \
                    + str(T) +"; check burn-in condition in ASGD_CI_lin_reg_manual()."
                assert np.sum(A_bar)!=0, "A_bar = S_bar is initlized to 0 at time step T = " \
                    + str(T) +"; check burn-in condition in ASGD_CI_lin_reg_manual()."
                    
                #Initialize WASGD estimate (theta_bar_AW) using SGD estimate (theta_new)
                if output=="WASGD":
                    theta_bar_W = theta_new.copy()
                
                
            #Burn-in condition: only update theta_bar if burn-in is over
            if T > start_t:
                
                # Update ASGD estimate (theta_bar_new) using old ASGD estimate
                # (theta_bar) and SGD estimate (theta_new)
                theta_bar_new = (1/T)*(theta_new-theta_bar) + theta_bar
            
                #Update matrices A_bar and S_bar
                A_bar_new = (1/T)*(chen.SGD_A_S_logistic(X[tt-1], theta_bar)-A_bar) + A_bar
                S_bar_new = A_bar_new.copy()
            
                
                theta_bar = theta_bar_new.copy()
                A_bar = A_bar_new.copy()
                S_bar = S_bar_new.copy()
            
            
                #Adaptive weighted ASGD (WASGD) from Wei et al 2023:
                if output=="WASGD":
                    
                    theta_bar_W_new = ((T-1)/T)*theta_bar_W + ((1-T**alpha_lr)/T)*theta \
                        + (T**(alpha_lr-1))*theta
                    theta_bar_W = theta_bar_W_new.copy()
                     
            
                #Keep track of min and max theta_bars (to check for "wildness")
                for i in range(d):
                    min_theta_bar[i] = np.min([min_theta_bar[i], theta_bar_new[i]])
                    max_theta_bar[i] = np.max([max_theta_bar[i], theta_bar_new[i]])
                    
                if (T-1)%1000==0 and verbose:
                    #print("     grad = " + str(g_theta_bar))
                    print("********** Time step T = " + str(T) + " ***********")
                    print("theta_bar_new = " + str(theta_bar_new) )
                    print("theta_new = " + str(theta_new) )
                    if output=="WASGD":
                        print("theta_bar_W_new = " + str(theta_bar_W_new) )
            
                    
            #Update SGD estimate outside of burn-in condition
            theta = theta_new.copy()
            
            
            if plot:
                
                p = plot_theta_dim
                
                if T >= start_t:
                    
                    #This plot tracks progress of SGD and ASGD estimate over time steps T
                    #ax.plot(T, theta[p], marker = "," , color = 'blue', linestyle='-')
                    #ax.plot(T, theta_bar[p], marker = ",", linestyle='--', color = 'red')
                    theta_bar_t.append(theta_bar[p])
                    theta_hat_t.append(theta[p])
                    if output=="WASGD":
                        theta_bar_W_t.append(theta_bar_W[p])
                    
                    

    
    if plot:
        #This plot tracks progress of ASGD vs SGD over time steps t
        
        # Print true theta for reference
        true_theta_p = np.linspace(0, 1, num=d)[p]
        
        #SGD line:
        plt.plot(np.arange(start_t-1,T,1), theta_hat_t, marker = "," , color = '#7f7f7f', linestyle='-')
        
        #ASGD line:
        plt.plot(np.arange(start_t-1,T,1), theta_bar_t, marker = ",", linestyle='-', color = 'red')

        if output=="WASGD":
            #WASGD line:
            plt.plot(np.arange(start_t-1,T,1), theta_bar_W_t, marker = ",", linestyle='--', color = 'blue')
        #plt.ylim((min_theta_bar[p]-2, max_theta_bar[p]+2))
        plt.axhline(0, color='black', linestyle='-', alpha = .4)
        plt.axvline(0, color='black', linestyle='-', alpha = .4)
        
        if shift_k==False: #Keeps "pythonic" dimensions 0,...,d-1 for plotting
            p_to_print = p
        if shift_k==True: #Prints dimensions 1,...,d in titles
            p_to_print= p+1
        
        ASGD_label = r"ASGD $e^\top_" + str(p_to_print) + r"  \bar{\theta}_T$ =" + str(np.round(theta_bar[p], 3))
        SGD_label = r"SGD $e^\top_" + str(p_to_print) + r"  \hat{\theta}_T$ =" + str(np.round(theta[p], 3))
        #ASGD_label = r"ASGD $\bar{\theta}_" + str(p_to_print) + "$ = " + str(np.round(theta_bar[p], 3))
        #SGD_label = r"SGD $\hat{\theta}_" + str(p_to_print) + "$ = " + str(np.round(theta[p], 3))
        
        plot_text_result = False #If true, puts final iterate inside graph in addition to legend, but sometimes not the best placement of the annotation.
        
        if plot_text_result==True:
            plt.text(x=T-.3*T, y=(max_theta_bar[p]-min_theta_bar[p])/2, s=ASGD_label, fontsize=12, color='red')
            plt.text(x=T-.3*T, y=(max_theta_bar[p]-min_theta_bar[p])/30, s=SGD_label, fontsize=12, color='k')
        
        if output=="WASGD":
            WASGD_label = r"WASGD $\hat{\theta}_" + str(p_to_print) + "$ = " + str(np.round(theta_bar_W[p], 3))
            if plot_text_result==True:
                plt.text(x=T-.3*T, y=(max_theta_bar[p]-min_theta_bar[p])/1, s=WASGD_label, fontsize=12, color='blue')
        
        #Legend
        if output!="WASGD":
            custom_lines = [Line2D([0], [0], color="#545454", lw=1),
                            Line2D([0], [0], color="red", lw=1)]
            
            plt.legend(custom_lines, [SGD_label, ASGD_label], labelcolor='linecolor')
        
        if output=="WASGD":
            custom_lines = [Line2D([0], [0], color="#545454", lw=1),
                            Line2D([0], [0], color="red", lw=1),
                            Line2D([0], [0], color="blue", lw=1)]
            
            plt.legend(custom_lines, [SGD_label, ASGD_label, WASGD_label], labelcolor='linecolor')
        
        
        plt.xlabel('t')
        #plt.ylabel(r'$\theta_' + str(p_to_print) + "$")
        #plt.title(r'(Logistic regression) Manual ASGD progress to estimate $\theta_{' +\
        #          str(p_to_print) + '}$=' + str(true_theta_p))
        plt.ylabel(r'$e^\top_' + str(p_to_print) + r'\theta_t$')
        plt.title(r'(Logistic regression) ASGD progress to estimate $e^\top_' + str(p_to_print) +\
                  r'   \theta_\infty$=' + str(true_theta_p))
        N = '%.1E' % n #puts n into scientific notation (shorter)
        step_info = r', step $\eta =' + str(c) + '$' if  fixed_step else  r', step $\eta_t =' + str(c) + 't^{-' + str(alpha_lr) +'}$'
            
        if plot_footnote==True:
            plt.annotate("n = " + N + ", epochs = " + str(epochs) + ", t0 = " + str(start_t) + ", " +  \
                         "initializer = " + str(initializer) + ", dim = " + str(d) + step_info,
                    xy = (1.0, -0.2),
                    xycoords='axes fraction',
                    ha='right',
                    va="center",
                    color = "gray",
                    fontsize=10)
        plt.savefig('plots//ASGD progress//ASGD progress_' + \
                    datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + ".pdf",\
                        bbox_inches='tight', dpi=500)
        plt.show()
                
        
    
    if output=="WASGD":
        theta_bar = theta_bar_W.copy()
    
    
    return theta, theta_bar, A_bar, S_bar, n






def hulc_CI_logreg_manual(data,  alpha, ytype = "neg11", output = "ASGD", c = .01, alpha_lr = .501, burn_in = False,
                          burn_in_threshold = 1,\
                          initializer = False, epochs = 1,
                          return_thetas = True):
    # x = data
    batch_size = hulc.B_star(alpha, 0)
    # param = [ASGD_CI_lin_reg(data) for b in range(batch_size)]
    batches = hulc.divide_data_into_batches(data, batch_size)
    #print(len(batches))
    d = data[0].shape[1]
    theta_bar_estimates = [ [] for i in range(d)]
    
    for batch in batches:
        
        assert batch[0].shape[0]>0, "Batch size is too small: only " \
            + str(batch[0].shape[0]) + ". See: hulc_CI_logreg_manual()"
        
        theta, theta_bar, A_bar, S_bar, n = ASGD_CI_log_reg_manual(batch, ytype=ytype, output = output, c = c, \
                                                                   alpha_lr = alpha_lr,\
                                                                   burn_in = burn_in,\
                                                                   burn_in_threshold = burn_in_threshold, \
                                                            initializer = initializer,  plot_theta_dim=0,\
                                                                plot=False, verbose = False,\
                                                                    epochs = epochs)
            #print(theta_bar)
        
        if output=="ASGD" or output=="WASGD":
            theta_bar = theta_bar
        elif output=="SGD":
            theta_bar = theta.copy()
        
        for i in range(d):
            theta_bar_estimates[i].append(theta_bar[i])
        

    Hulc_lb = np.min(theta_bar_estimates, axis = 1)
    Hulc_ub = np.max(theta_bar_estimates, axis = 1)
    
    Hulc_CI = []
    for i in range(d):
        CI = [Hulc_lb[i], Hulc_ub[i]]
        Hulc_CI.append(CI)
    
    if return_thetas==True:
        return Hulc_CI, theta_bar_estimates
    else:
        return Hulc_CI




def tstat_CI_logreg_manual(data, alpha, ytype = "neg11", output = "ASGD", c = .01, alpha_lr = .501, burn_in = False,
                          burn_in_threshold = 1,\
                          initializer = False, epochs = 1):
    # x = data
    batch_size = hulc.B_star(alpha, 0)
    # param = [ASGD_CI_lin_reg(data) for b in range(batch_size)]
    batches = hulc.divide_data_into_batches(data, batch_size)
    #print(len(batches))
    d = data[0].shape[1]
    theta_bar_estimates = [ [] for i in range(d)]
    
    for batch in batches:
        
        assert batch[0].shape[0]>0, "Batch size is too small: only " \
            + str(batch[0].shape[0]) + ". See: hulc_CI_linreg_manual()"
        
        theta, theta_bar, A_bar, S_bar, n = ASGD_CI_log_reg_manual(batch, ytype=ytype, output = output, c = c, \
                                                                   alpha_lr = alpha_lr,\
                                                                   burn_in = burn_in,\
                                                                   burn_in_threshold = burn_in_threshold, \
                                                            initializer = initializer,  plot_theta_dim=0,\
                                                                plot=False, verbose = False,\
                                                                    epochs = epochs)
            #print(theta_bar)
        
        if output=="ASGD" or output=="WASGD":
            theta_bar = theta_bar
        elif output=="SGD":
            theta_bar = theta.copy()
        
        for i in range(d):
            theta_bar_estimates[i].append(theta_bar[i])
        
    #Compute critical value for t_{alpha/2}
    t_threshold = t.ppf(q=1-alpha, df=batch_size-1) #from scipy.stats package
    
    
    ###Denominator###
    theta_bar_estimates = np.array(theta_bar_estimates)
    theta_bar_bar = np.mean(theta_bar_estimates, axis = 1)
    theta_bar_bar_matrix = np.repeat(theta_bar_bar, batch_size).reshape(theta_bar_estimates.shape)
    
    sq_devs = (theta_bar_estimates - theta_bar_bar_matrix)**2
    sum_sq_devs = np.sum(sq_devs, axis = 1)
    denom = np.sqrt((1/(batch_size-1))*sum_sq_devs)
    
    deviation = (t_threshold/np.sqrt(batch_size))*denom
        
    #Lower bound
    lb = theta_bar_bar - deviation
    #Upper bound
    ub = theta_bar_bar + deviation

    t_CI = np.array([lb, ub]).reshape((2, d))
    t_CI = t_CI.transpose(1,0)

    return(t_CI)




