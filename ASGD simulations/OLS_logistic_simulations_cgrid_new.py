# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 10:50:11 2023

@author: selin

This file generates does logistic regression data in multiple dimensions
and performs ASGD confidence interval accordinng to the parameters in
Chen & Lee's paper (Section 6)

"""
import numpy as np
import statsmodels.api as sm
from scipy.stats import t
from datetime import datetime
import pandas as pd
import copy
import os
import warnings
run_locally=True
if run_locally:
    os.chdir("C://Users//selin//Dropbox//Hulc_simulations//hulc_simulations//HulC_python")
#else:
#    os.chdir("//zfsauton2//home//shcarter//hulc")

import gen_data as gld
import ASGD_Chen_functions as chen  # contains: SGD_grad_linear(theta, x_i, y_i), 
                                    #           SGD_hessian_linear(x_i),
                                    #           eta_t(t, c = .01, alpha=.501)


import ASGD_HulC_manual as ahm      # contains: ASGD_CI_lin_reg_manual()
                                    #           hulc_CI_linreg_manual()
                                    #           tstat_CI_linreg_manual()

import timeit #calculates runtime 



#%%
#####################################
## Confidence interval simulations
#####################################

# In the following for-loop, I'll produce N samples and check the width 
# and coverage of the plug-in estimator vs OLS.





def run_sims(model_type = "Logistic", S=400, N=10**4, D=5, ytype = "neg11", output = "ASGD", \
                      cov_type = "I", \
                      alpha_level = .05, XY_type = "normal_0", c_grid = [1], alpha_lr = .505, \
                       burn_in=False, burn_in_threshold = 1, initializer = True, epochs_for_HulC = 1, \
                           fixed_step = False, save_plots=True):


    # model_type = "Logistic" or "OLS"
    # alpha_level = .05 #Type  I error rate

    # S  = number of loops (Chen et al in Table 1 use 500)
    # N  = number of samples per loop
    # D = dimension of linear regression (1 intercept + D-1 slopes)

    # ytype = "neg11"  #for logistic regression only; options are "neg11" (y is in {-1,1}) or "01" (y is in {0,1})
    # output = "ASGD" #options are "ASGD", "WASGD", or "SGD"
    # cov_type = "EquiCorr" #options are "I", "Toeplitz", "EquiCorr" (this controls out X is generated; se gld.gen_normal0_data)

    #gen_data = How to generate the data (see gen_data.py)
    

    
    if XY_type=="normal_0":
        if model_type == "Logistic":
            gen_data_type = gld.gen_normal0_logistic_data
        if model_type == "OLS":
            gen_data_type = gld.gen_normal0_data
        start= 0 
        stop = 1
        true_thetas=np.linspace(start, stop, D)

    
   
    
    #If estimated thetas are above nan_threshold (in absolute value),
    # then convert estimate to nan
    nan_threshold = 1e+10
    
    adjust_c = False
    if adjust_c==True:
        #For large dimension D, we mustn't let c get too large. So we adjust it.
        c_grid_new = []
        if D>50:
            for c in c_grid:
                c_new = c*5/D
                c_grid_new.append(c_new)
            c_grid = c_grid_new.copy()
    
    # Initialize an empty dictionary to store results for each c
    empty_dict = {} 
    for c in c_grid:
        empty_dict[c] = []
    
    empty_dict_zeros = {} 
    for c in c_grid:
        empty_dict_zeros[c] = np.zeros(D)
    

    #(W)(A)SGD estimates and CI using Chen et al plug-in estimator
    ASGD_theta = copy.deepcopy(empty_dict)
    CI_ASGD = copy.deepcopy(empty_dict)
    width_ASGD = copy.deepcopy(empty_dict)
    coverage_ASGD = copy.deepcopy(empty_dict)
    coverage_ASGD_with_nans = copy.deepcopy(empty_dict)
    num_NaNs_ASGD = copy.deepcopy(empty_dict_zeros)
    runtime_ASGD = copy.deepcopy(empty_dict)
    ASGD_notes = copy.deepcopy(empty_dict)
    
    #OLS sandwhich estimator & CI
    lm_theta = copy.deepcopy(empty_dict)
    CI_lm = copy.deepcopy(empty_dict)
    width_lm = copy.deepcopy(empty_dict)
    coverage_lm = copy.deepcopy(empty_dict)
    coverage_lm_with_nans = copy.deepcopy(empty_dict)
    num_NaNs_lm = copy.deepcopy(empty_dict_zeros)
    runtime_OLS = copy.deepcopy(empty_dict)
    OLS_notes = copy.deepcopy(empty_dict)
    
    #Hulc CI
    CI_Hulc = copy.deepcopy(empty_dict)
    width_Hulc = copy.deepcopy(empty_dict)
    coverage_Hulc = copy.deepcopy(empty_dict)
    coverage_Hulc_with_nans = copy.deepcopy(empty_dict)
    num_NaNs_Hulc = copy.deepcopy(empty_dict_zeros)
    runtime_HulC = copy.deepcopy(empty_dict)
    HulC_notes = copy.deepcopy(empty_dict)
    
    
    #t-stat CI
    CI_tstat = copy.deepcopy(empty_dict)
    width_tstat = copy.deepcopy(empty_dict)
    coverage_tstat = copy.deepcopy(empty_dict)
    coverage_tstat_with_nans = copy.deepcopy(empty_dict)
    num_NaNs_tstat = copy.deepcopy(empty_dict_zeros)
    runtime_tstat = copy.deepcopy(empty_dict)
    tstat_notes = copy.deepcopy(empty_dict)




    date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")


    for s in range(S):
        if s%1==0:
            print("s = " + str(s))

        #Draw samples
        if model_type=="Logistic":
            XY = gen_data_type(n=N, d=D, ytype=ytype, cov_type = cov_type, torch = False)
        if model_type=="OLS":
            XY = gen_data_type(n=N, d=D, cov_type = cov_type, torch = False)
        
        
        for c in c_grid:
            if s%5==0:
                print("     c = " + str(c))

            if output=="WASGD":
                c_effective = 1
            else:
                c_effective = c
            
            
            #########################################
            # ASGD plug-in
            #########################################
            
            
            
            start_ASGD = timeit.default_timer()
            if model_type=="Logistic":
                theta, theta_bar, A_bar, S_bar, n = ahm.ASGD_CI_log_reg_manual(XY, ytype=ytype, output = output,\
                                                                           c = c, alpha_lr = alpha_lr,\
                                                                           burn_in = burn_in,\
                                                                           burn_in_threshold = burn_in_threshold, \
                                                                           initializer = initializer,\
                                                                           plot = False, \
                                                                           epochs = 1) 
            if model_type == "OLS":
                theta, theta_bar, A_bar, S_bar, n = ahm.ASGD_CI_lin_reg_manual(XY, output = output,\
                                                                           c = c, \
                                                                           alpha_lr = alpha_lr,\
                                                                           burn_in = burn_in,\
                                                                           burn_in_threshold = burn_in_threshold, \
                                                                           initializer = initializer,\
                                                                           plot = False, \
                                                                           epochs = 1)
            stop_ASGD = timeit.default_timer()
            
            
            runtime_ASGD[c].append(stop_ASGD-start_ASGD)
            

            
            #Don't save thetas if values are too huge; replace with nan
            count_too_big = np.sum(np.abs(theta_bar)>nan_threshold)
            if count_too_big > 0:
                theta_bar_nan = theta_bar.copy()
                theta_bar_nan[np.abs(theta_bar_nan)>nan_threshold] = np.nan
                ASGD_theta[c].append(theta_bar_nan)
                ASGD_note = "theta > " + str(nan_threshold) + "; "
            else:
                ASGD_theta[c].append(theta_bar)
                ASGD_note = ""
            
            #If any theta_k value is nan, count it to num_NaNs_ASGD.
            if np.isnan(np.sum(theta_bar)):
                num_NaNs_ASGD[c] = num_NaNs_ASGD[c] + (np.isnan(theta_bar)).astype(int)
            
            
            #If any of the output is nan, print a warning
            all_values = np.column_stack((theta, theta_bar, A_bar, S_bar)).flatten()
            if np.isnan(np.sum(all_values)):
                print("#########################")
                print("NOTE: Chen et al's ASGD algorithm produced Nan values for model_type = " + str(model_type) + ", s = " + str(s) + ", N = "\
                     + str(N) + ", D = " + str(D) + ", output = " + output + ", cov_type = " + cov_type + ", c = "\
                         + str(c) + ", alpha_lr = " + str(alpha_lr) + ".")
                print("We will save nan values for this iteration.")
                print("#########################")
               
                ASGD_note = ASGD_note + "CIs are NaNs"
            
            ASGD_notes[c].append(ASGD_note)  
                   
            if output=="ASGD" or output=="WASGD": 
                theta_bar = theta_bar #Keep theta_bar
            elif output=="SGD":
                theta_bar = theta.copy() #Replace ASGD estimate with SGD estimate (to avoid multiple variable names throughout)
            
            
        
        
            #produce Chen plug-in CIs
            CI_asgd, Sigma = chen.ASGD_plugin_CI(theta_bar, A_bar, S_bar, N)
            CI_asgd = np.array(CI_asgd)
            CI_ASGD[c].append(CI_asgd)
            
            #If any CI_asgd value is nan, coverage is automatically False in coverage_ASGD
            width_ASGD[c].append(CI_asgd[:,1]-CI_asgd[:,0])
            covered_ASGD = []
            for d in range(D):
                covered = true_thetas[d] >= CI_asgd[d,0] and true_thetas[d] <= CI_asgd[d,1]
                covered_ASGD.append(covered)
            
            coverage_ASGD[c].append(covered_ASGD)
            
            #If any value is nan, we record coverage as nan (instead of False)
            # in coverage_ASGD_with_nans
            covered_ASGD_with_NaN = []
            for d in range(D):
                d_isnan = np.isnan(CI_asgd[d,0] + CI_asgd[d,1])
                covered = true_thetas[d] >= CI_asgd[d,0] and true_thetas[d] <= CI_asgd[d,1]
                if d_isnan==True:
                    covered = np.nan
                covered_ASGD_with_NaN.append(covered)

            
            coverage_ASGD_with_nans[c].append(covered_ASGD_with_NaN)
            
           
            #########################################
            # OLS confidence intervals
            #########################################
            
            start_OLS = timeit.default_timer()
        
            X_ = XY[0].copy()
            Y_ = XY[1].copy()
            n = X_.shape[0]
            #Check if X matrix has column of ones; if not, add it.
            if np.sum(X_[:,0]) != n:
                X_ = np.c_[np.ones(n), X_]
            
            if model_type=="Logistic":
            
                if ytype=="neg11":
                    Y_[np.where(Y_==-1)]=0
                
                try: #This try/except procedure avoids error due to singularity issue
                    lm_reg = sm.Logit( Y_, X_ )
                    lm_results = lm_reg.fit(cov_type='HC1', disp=0) #cov_type='HC1' produces sandwhich  (robust) CI
                    lm_theta[c].append(lm_results.params)
                    CI_lm_ = lm_results.conf_int(alpha=alpha_level )
                except: # any type of error
                    print("                  Singular matrix encountered in logistic regression;" +\
                          " plugging in (np.nan, np.nan) for CI")
                    print("                  (N=" + str(N) + ", c=" + str(c) + ")")
                    CI_lm_ = np.zeros(shape=(D,2))
                    CI_lm_[CI_lm_==0] = np.nan
                    
                    num_NaNs_lm[c] = num_NaNs_lm[c] + np.repeat(1, D)
                    

        
            if model_type=="OLS":
                try: #This try/except procedure avoids error due to singularity issue
                    lm_reg = sm.OLS( Y_, X_ )
                    lm_results = lm_reg.fit(cov_type='HC1') #cov_type='HC1' produces sandwhich  (robust) CI
                    lm_theta[c].append(lm_results.params)
                    CI_lm_ = lm_results.conf_int(alpha=alpha_level )
                except: #any type of error
                    print("                  Singular matrix encountered in OLS regression;" +\
                          " plugging in (np.nan, np.nan) for CI")
                    print("                  (N=" + str(N) + ", c=" + str(c) + ")")
                    CI_lm_ = np.zeros(shape=(D,2))
                    CI_lm_[CI_lm_==0] = np.nan
                    
                    num_NaNs_lm[c] = num_NaNs_lm[c] + np.repeat(1, D)
                    
            
            stop_OLS = timeit.default_timer()
            runtime_OLS[c].append(stop_OLS-start_OLS)
            
            CI_lm_ = np.array(CI_lm_)
            CI_lm[c].append(CI_lm_)
        
            width_lm[c].append(CI_lm_[:,1]-CI_lm_[:,0])
    
        
        
            covered_lm = []
            for d in range(D):
                covered = true_thetas[d] >= CI_lm_[d,0] and true_thetas[d] <= CI_lm_[d,1]
                covered_lm.append(covered)
            
            coverage_lm[c].append(covered_lm)
            
            
            #If any value is nan (or all are non-nan), we record separately
            # in coverage_lm_with_nans
            covered_lm_with_NaN = []
            for d in range(D):
                d_isnan = np.isnan(CI_lm_[d,0] + CI_lm_[d,1])
                covered = true_thetas[d] >= CI_lm_[d,0] and true_thetas[d] <= CI_lm_[d,1]
                if d_isnan==True:
                    covered = np.nan
                covered_lm_with_NaN.append(covered)

            
            coverage_lm_with_nans[c].append(covered_lm_with_NaN)
            
            
            
            
            if np.isnan(np.sum(CI_lm_)):
                OLS_note = "NaNs produced in CI (probably singular matrix)"
            else:
                OLS_note = ""
        
            OLS_notes[c].append(OLS_note)
            
            #########################################
            # HulC CI's
            #########################################
            
            start_HulC = timeit.default_timer()
            if model_type=="Logistic":
                h_CI, theta_bar_estimates = ahm.hulc_CI_logreg_manual(XY, alpha_level,
                                                                      ytype, output, c,
                                                                      alpha_lr, burn_in, 
                                                                      burn_in_threshold, 
                                                                      initializer, epochs_for_HulC,
                                                                      return_thetas = True)
            if model_type=="OLS":
                h_CI, theta_bar_estimates = ahm.hulc_CI_linreg_manual(XY, alpha_level,
                                                                      output, c,
                                                                      alpha_lr, burn_in, 
                                                                      burn_in_threshold, 
                                                                      initializer, epochs_for_HulC)
            
            stop_HulC = timeit.default_timer()
            runtime_HulC[c].append(stop_HulC-start_HulC)
            
                
            h_CI = np.array(h_CI)
            CI_Hulc[c].append(h_CI)
            
          
            
            #Don't save confidence bounds if values are too huge; replace with nan
            count_too_big = np.sum(np.abs(h_CI)>nan_threshold)
            if count_too_big > 0:
                CI_h_nan = h_CI.copy()
                CI_h_nan[np.abs(CI_h_nan)>nan_threshold] = np.nan
                Hulc_note = "CI bound > " + str(nan_threshold) + "; "
            else:
                Hulc_note = ""
            
            #If any confidence bound is nan, count it to num_NaNs_Hulc.
            if np.isnan(np.sum(h_CI)):
                num_NaNs_Hulc[c] = num_NaNs_Hulc[c] + np.sum((np.isnan(h_CI)).astype(int), axis=1)
            
            
            
            #If any of the output is nan, print a warning
            if np.isnan(np.sum(h_CI)):
                print("#########################")
                print("NOTE: For HulC implementation of Chen et al's ASGD algorithm, Nan values were produced for model_type = " + str(model_type) + ", s = " + str(s) + ", N = "\
                     + str(N) + ", D = " + str(D) + ", output = " + output + ", cov_type = " + cov_type + ", c = "\
                         + str(c) + ", alpha_lr = " + str(alpha_lr) + ".")
                print("We will save nan values for this iteration.")
                print("#########################")
                Hulc_note = Hulc_note + "CIs are NaNs"
            
            HulC_notes[c].append(Hulc_note)
            
            #If any h_CI value is nan, coverage is automatically False in coverage_Hulc
            width_Hulc[c].append(h_CI[:,1]-h_CI[:,0])
            covered_hulc = []
            for d in range(D):
                covered = true_thetas[d] >= h_CI[d,0] and true_thetas[d] <= h_CI[d,1]
                covered_hulc.append(covered)
            
            coverage_Hulc[c].append(covered_hulc)
            
            #If any value is nan, we record coverage as nan (instead of False)
            # in coverage_Hulc_with_nans
            covered_hulc_with_NaN = []
            for d in range(D):
                d_isnan = np.isnan(h_CI[d,0] + h_CI[d,1])
                covered = true_thetas[d] >= h_CI[d,0] and true_thetas[d] <= h_CI[d,1]
                if d_isnan==True:
                    covered = np.nan
                covered_hulc_with_NaN.append(covered)

            
            coverage_Hulc_with_nans[c].append(covered_hulc_with_NaN)
            
            
            
            
            #########################################
            # t-stat CI's
            #########################################
            
            start_tstat = timeit.default_timer()
            
            #Here, we shouldn't run ASGD on B batches again, since this was already
            # done for HulC CI
            recycle_HulC_thetas = True
            if recycle_HulC_thetas==True:
                #Compute critical value for t_{alpha/2}
                theta_bar_estimates = np.array(theta_bar_estimates)
                batch_size = theta_bar_estimates.shape[1]
                t_threshold = t.ppf(q=1-alpha_level/2, df=batch_size-1) #from scipy.stats package
                
                
                ###Denominator###
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
        
                tstat_CI = np.array([lb, ub]).reshape((2, D))
                tstat_CI = tstat_CI.transpose(1,0)
            
            #In case you want to re-compute the theta estimates for B batches, you run this.
            #(Note the result will be different than the HulC estimates of theta because the seed changes.)
            if recycle_HulC_thetas==False:
                if model_type=="Logistic":
                    tstat_CI = ahm.tstat_CI_logreg_manual(XY, alpha_level, ytype, output, c, alpha_lr, burn_in, \
                                                  burn_in_threshold, 
                                                initializer, epochs_for_HulC)
                if model_type=="OLS":
                    tstat_CI = ahm.tstat_CI_linreg_manual(XY, alpha_level, output, c, alpha_lr, burn_in, \
                                                 burn_in_threshold, 
                                                initializer, epochs_for_HulC)
                    
            stop_tstat = timeit.default_timer()
            runtime_tstat[c].append(stop_tstat-start_tstat)
            
            
            CI_tstat[c].append(tstat_CI)
            

            #Don't save confidence bounds if values are too huge; replace with nan
            count_too_big = np.sum(np.abs(tstat_CI)>nan_threshold)
            if count_too_big > 0:
                CI_tstat_nan = tstat_CI.copy()
                CI_tstat_nan[np.abs(CI_tstat_nan)>nan_threshold] = np.nan
                tstat_note = "CI bound > " + str(nan_threshold) + "; "
            else:
                tstat_note = ""
            
            #If any confidence bound is nan, count it to num_NaNs_Hulc.
            if np.isnan(np.sum(tstat_CI)):
                num_NaNs_tstat[c] = num_NaNs_tstat[c] + np.sum((np.isnan(tstat_CI)).astype(int), axis=1)
            
            
            
            #If any of the output is nan, print a warning
            if np.isnan(np.sum(tstat_CI)):
                print("#########################")
                print("NOTE: For t-stat implementation of Chen et al's ASGD algorithm, Nan values were produced for model_type = " + str(model_type) + ", s = " + str(s) + ", N = "\
                     + str(N) + ", D = " + str(D) + ", output = " + output + ", cov_type = " + cov_type + ", c = "\
                         + str(c) + ", alpha_lr = " + str(alpha_lr) + ".")
                print("We will save nan values for this iteration.")
                print("#########################")
                tstat_note = tstat_note + "CIs are NaNs"
            
            tstat_notes[c].append(tstat_note)
            
            #If any tstat_CI value is nan, coverage is automatically False in coverage_tstat
            width_tstat[c].append(tstat_CI[:,1]-tstat_CI[:,0])
            covered_tstat = []
            for d in range(D):
                covered = true_thetas[d] >= tstat_CI[d,0] and true_thetas[d] <= tstat_CI[d,1]
                covered_tstat.append(covered)
            
            coverage_tstat[c].append(covered_tstat)
            
            #If any value is nan, we record coverage as nan (instead of False)
            # in coverage_tstat_with_nans
            covered_tstat_with_NaN = []
            for d in range(D):
                d_isnan = np.isnan(tstat_CI[d,0] + tstat_CI[d,1])
                covered = true_thetas[d] >= tstat_CI[d,0] and true_thetas[d] <= tstat_CI[d,1]
                if d_isnan==True:
                    covered = np.nan
                covered_tstat_with_NaN.append(covered)

            coverage_tstat_with_nans[c].append(covered_tstat_with_NaN)
    
    
    #Take averages across the S runs
    ASGD_theta_avg_dict = copy.deepcopy(empty_dict)
    width_ASGD_avg_dict = copy.deepcopy(empty_dict)
    width_ASGD_median_dict = copy.deepcopy(empty_dict)
    CI_ASGD_lb_dict = copy.deepcopy(empty_dict)
    CI_ASGD_ub_dict = copy.deepcopy(empty_dict)
    runtime_ASGD_avg_dict = copy.deepcopy(empty_dict)
    
    lm_theta_avg_dict = copy.deepcopy(empty_dict)
    width_lm_avg_dict = copy.deepcopy(empty_dict)
    width_lm_median_dict = copy.deepcopy(empty_dict)
    CI_lm_lb_dict = copy.deepcopy(empty_dict)
    CI_lm_ub_dict = copy.deepcopy(empty_dict)
    runtime_OLS_avg_dict = copy.deepcopy(empty_dict)
    
    width_hulc_avg_dict = copy.deepcopy(empty_dict)
    width_hulc_median_dict = copy.deepcopy(empty_dict)
    CI_hulc_lb_dict = copy.deepcopy(empty_dict)
    CI_hulc_ub_dict = copy.deepcopy(empty_dict)
    runtime_HulC_avg_dict = copy.deepcopy(empty_dict)
    
    width_tstat_avg_dict = copy.deepcopy(empty_dict)
    width_tstat_median_dict = copy.deepcopy(empty_dict)
    CI_tstat_lb_dict = copy.deepcopy(empty_dict)
    CI_tstat_ub_dict = copy.deepcopy(empty_dict)
    runtime_tstat_avg_dict = copy.deepcopy(empty_dict)
    
    
    
    
    # I expect to see RuntimeWarnings in this block
    warnings.filterwarnings(action='ignore', message='Mean of empty slice')
    warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
    
    for c in c_grid:
        
        ASGD_theta_avg_dict[c] = np.nanmean(ASGD_theta[c], axis = 0) #I don't end up using ASGD_theta_avg, but it's here for safe keeping
        width_ASGD_avg_dict[c] = np.nanmean(width_ASGD[c], axis = 0)
        width_ASGD_median_dict[c] = np.nanmedian(width_ASGD[c], axis = 0)
        CI_ASGD_lb_dict[c] = np.nanmean(CI_ASGD[c], axis =0)[:,0]
        CI_ASGD_ub_dict[c] = np.nanmean(CI_ASGD[c], axis =0)[:,1]     
        runtime_ASGD_avg_dict[c] = np.mean(runtime_ASGD[c])
        
        lm_theta_avg_dict[c] = np.mean(lm_theta[c], axis = 0) #I don't end up using lm_theta_avg, but it's here for safe keeping
        width_lm_avg_dict[c] = np.nanmean(width_lm[c], axis = 0)
        width_lm_median_dict[c] = np.nanmedian(width_lm[c], axis = 0)
        CI_lm_lb_dict[c] = np.nanmean(CI_lm[c],axis=0)[:,0]
        CI_lm_ub_dict[c] = np.nanmean(CI_lm[c],axis=0)[:,1]
        runtime_OLS_avg_dict[c] = np.mean(runtime_OLS[c])
        
        width_hulc_avg_dict[c]  = np.nanmean(width_Hulc[c], axis = 0)
        width_hulc_median_dict[c] = np.nanmedian(width_Hulc[c], axis = 0)
        CI_hulc_lb_dict[c] = np.nanmean(CI_Hulc[c],axis=0)[:,0]
        CI_hulc_ub_dict[c] = np.nanmean(CI_Hulc[c],axis=0)[:,1]
        runtime_HulC_avg_dict[c] = np.mean(runtime_HulC[c])
        
        width_tstat_avg_dict[c] = np.nanmean(width_tstat[c], axis = 0)
        width_tstat_median_dict[c] = np.nanmedian(width_tstat[c], axis = 0)
        CI_tstat_lb_dict[c] = np.nanmean(CI_tstat[c],axis=0)[:,0]
        CI_tstat_ub_dict[c] = np.nanmean(CI_tstat[c],axis=0)[:,1]
        runtime_tstat_avg_dict[c] = np.mean(runtime_tstat[c])
    
    
    
    ASGD_nan_flag_dict = copy.deepcopy(empty_dict)
    Hulc_nan_flag_dict = copy.deepcopy(empty_dict)
    tstat_nan_flag_dict = copy.deepcopy(empty_dict)
    OLS_nan_flag_dict = copy.deepcopy(empty_dict)
    
    for c in c_grid:
        ASGD_nan_flag_dict[c] = "" if list(set(ASGD_notes[c]))[0]=="" else " (" + str(list(set(ASGD_notes))[0]) + ")"
        Hulc_nan_flag_dict[c] = "" if list(set(HulC_notes[c]))[0]=="" else " (" + str(list(set(HulC_notes))[0]) + ")"
        tstat_nan_flag_dict[c] = "" if list(set(tstat_notes[c]))[0]=="" else " (" + str(list(set(tstat_notes))[0]) + ")"
        OLS_nan_flag_dict[c] = "NaNs produced in CI (probably singular matrix)" if "NaNs produced in CI (probably singular matrix)" in list(set(OLS_notes[c])) else ""
    
    
    #Calculate coverages
    coverages_ASGD_dict = copy.deepcopy(empty_dict)
    coverages_lm_dict = copy.deepcopy(empty_dict)
    coverages_Hulc_dict = copy.deepcopy(empty_dict)
    coverages_tstat_dict = copy.deepcopy(empty_dict)
    
    coverages_ASGD_with_nans_dict = copy.deepcopy(empty_dict)
    coverages_lm_with_nans_dict = copy.deepcopy(empty_dict)
    coverages_Hulc_with_nans_dict = copy.deepcopy(empty_dict)
    coverages_tstat_with_nans_dict = copy.deepcopy(empty_dict)
    
    
    for c in c_grid:
        
        coverages_ASGD_dict[c] = np.array(coverage_ASGD[c])
        coverages_lm_dict[c] = np.array(coverage_lm[c])
        coverages_Hulc_dict[c] = np.array(coverage_Hulc[c])
        coverages_tstat_dict[c] = np.array(coverage_tstat[c])
        
        coverages_ASGD_with_nans_dict[c] = np.array(coverage_ASGD_with_nans[c])
        coverages_lm_with_nans_dict[c] = np.array(coverage_lm_with_nans[c])
        coverages_Hulc_with_nans_dict[c] = np.array(coverage_Hulc_with_nans[c])
        coverages_tstat_with_nans_dict[c] = np.array(coverage_tstat[c])
        
        
    
        print("-----" + output + " (c = " + str(c) + ")------")
        print(".........................................................")
        
        print("-----HULC COVERAGE (c = " + str(c) + ")-----")
        for i in range(D):
            if D>10 and i%5==0:
                print("HulC coverage for theta_" + str(i) + " = " + str(np.round(100*np.nanmean(coverages_Hulc_with_nans_dict[c], axis = 0)[i], 1)) + "% ;  " + Hulc_nan_flag_dict[c])
        
        print("-----t-STAT COVERAGE (c = " + str(c) + ")-----")
        for i in range(D):
            if D>10 and i%5==0:
                print("t-stat coverage for theta_" + str(i) + " = " + str(np.round(100*np.nanmean(coverages_tstat_with_nans_dict[c], axis = 0)[i], 1)) + "% ;  " + tstat_nan_flag_dict[c])
        
        print("-----OLS COVERAGE (c = " + str(c) + ")-----")
        for i in range(D):
            if D>10 and i%5==0:
                print("OLS coverage for theta_" + str(i) + " = " + str(np.round(100*np.nanmean(coverages_lm_with_nans_dict[c], axis = 0)[i], 1))  + "% ;  " + OLS_nan_flag_dict[c])
        
        if output=="ASGD":
            print("-----ASGD Plug-in COVERAGE (c = " + str(c) + ")-----")
            for i in range(D):
                if D>10 and i%5==0:
                    print("ASGD coverage for theta_" + str(i) + " = " + str(np.round(100*np.nanmean(coverages_ASGD_with_nans_dict[c], axis = 0)[i], 1)) + "% ;  " + ASGD_nan_flag_dict[c])
        if output=="WASGD":
            print("-----WASGD Plug-in COVERAGE-----")
            for i in range(D):
                if D>10 and i%5==0:
                    print("WASGD coverage for theta_" + str(i) + " = " + str(np.round(100*np.nanmean(coverages_ASGD_with_nans_dict[c], axis = 0)[i], 1)) + "% ;  " + ASGD_nan_flag_dict[c])
        elif output=="SGD":
            print("-----SGD Plug-in COVERAGE (c = " + str(c) + ")-----")
            for i in range(D):
                if D>10 and i%5==0:
                    print("SGD coverage for theta_" + str(i) + " = " + str(np.round(100*np.nanmean(coverages_ASGD_with_nans_dict[c], axis = 0)[i], 1)) + "% ;  " + ASGD_nan_flag_dict[c])
        print(" ")
        print(" ")
        print(" ")
    
    
    ASGD_coverage_rate_dict = copy.deepcopy(empty_dict)
    OLS_coverage_rate_dict = copy.deepcopy(empty_dict)
    HulC_coverage_rate_dict = copy.deepcopy(empty_dict)
    tstat_coverage_rate_dict = copy.deepcopy(empty_dict)
    
    ASGD_coverage_rate_without_nans_dict = copy.deepcopy(empty_dict)
    OLS_coverage_rate_without_nans_dict = copy.deepcopy(empty_dict)
    HulC_coverage_rate_without_nans_dict = copy.deepcopy(empty_dict)
    tstat_coverage_rate_without_nans_dict = copy.deepcopy(empty_dict)
    

    
    for c in c_grid:
        ASGD_coverage_rate_dict[c] = 100*np.mean(coverages_ASGD_dict[c], axis = 0)
        OLS_coverage_rate_dict[c] = 100*np.mean(coverages_lm_dict[c], axis = 0)
        HulC_coverage_rate_dict[c] = 100*np.mean(coverages_Hulc_dict[c], axis = 0)
        tstat_coverage_rate_dict[c] = 100*np.mean(coverages_tstat_dict[c], axis = 0)
        
        ASGD_coverage_rate_without_nans_dict[c] = 100*np.nanmean(coverages_ASGD_with_nans_dict[c], axis = 0)
        OLS_coverage_rate_without_nans_dict[c] = 100*np.nanmean(coverages_lm_with_nans_dict[c], axis = 0)
        HulC_coverage_rate_without_nans_dict[c] = 100*np.nanmean(coverages_Hulc_with_nans_dict[c], axis = 0)
        tstat_coverage_rate_without_nans_dict[c] = 100*np.nanmean(coverages_tstat_with_nans_dict[c], axis = 0)
        
        
    
    ################
    # Width ratios
    ################
    
    ASGD_lm_width_ratio_dict = copy.deepcopy(empty_dict)
    HulC_ASGD_width_ratio_dict = copy.deepcopy(empty_dict)
    HulC_tstat_width_ratio_dict = copy.deepcopy(empty_dict)
    HulC_lm_width_ratio_dict = copy.deepcopy(empty_dict)
    tstat_lm_width_ratio_dict = copy.deepcopy(empty_dict)
    
    # Minimum widths
    width_ASGD_min_dict = copy.deepcopy(empty_dict)
    width_lm_min_dict = copy.deepcopy(empty_dict)
    width_Hulc_min_dict = copy.deepcopy(empty_dict)
    width_tstat_min_dict = copy.deepcopy(empty_dict)
    
    # Maximum widths
    width_ASGD_max_dict = copy.deepcopy(empty_dict)
    width_lm_max_dict = copy.deepcopy(empty_dict)
    width_Hulc_max_dict = copy.deepcopy(empty_dict)
    width_tstat_max_dict = copy.deepcopy(empty_dict)
    
    # Mean width ratios
    ASGD_lm_width_ratio_avg_dict = copy.deepcopy(empty_dict)
    HulC_ASGD_width_ratio_avg_dict = copy.deepcopy(empty_dict)
    HulC_tstat_width_ratio_avg_dict = copy.deepcopy(empty_dict)
    HulC_lm_width_ratio_avg_dict = copy.deepcopy(empty_dict)
    tstat_lm_width_ratio_avg_dict = copy.deepcopy(empty_dict)
    
    # Median width ratios    
    ASGD_lm_width_ratio_med_dict = copy.deepcopy(empty_dict)
    HulC_ASGD_width_ratio_med_dict = copy.deepcopy(empty_dict)
    HulC_tstat_width_ratio_med_dict = copy.deepcopy(empty_dict)
    HulC_lm_width_ratio_med_dict = copy.deepcopy(empty_dict)
    tstat_lm_width_ratio_med_dict = copy.deepcopy(empty_dict)
    
    #Min width ratios    
    ASGD_lm_width_ratio_min_dict = copy.deepcopy(empty_dict)
    HulC_ASGD_width_ratio_min_dict = copy.deepcopy(empty_dict)
    HulC_tstat_width_ratio_min_dict = copy.deepcopy(empty_dict)
    HulC_lm_width_ratio_min_dict = copy.deepcopy(empty_dict)
    tstat_lm_width_ratio_min_dict = copy.deepcopy(empty_dict)
    
    #Max width ratios    
    ASGD_lm_width_ratio_max_dict = copy.deepcopy(empty_dict)
    HulC_ASGD_width_ratio_max_dict = copy.deepcopy(empty_dict)
    HulC_tstat_width_ratio_max_dict = copy.deepcopy(empty_dict)
    HulC_lm_width_ratio_max_dict = copy.deepcopy(empty_dict)
    tstat_lm_width_ratio_max_dict = copy.deepcopy(empty_dict)
    
    
    for c in c_grid:
        #widths
        width_ASGD_np = np.array(width_ASGD[c]) #.reshape((S,D))
        width_lm_np = np.array(width_lm[c]) #.reshape((S,D))
        width_Hulc_np = np.array(width_Hulc[c]) #.reshape((S,D))
        width_tstat_np = np.array(width_tstat[c]) #.reshape((S,D))
        
        #width ratios
        ASGD_lm_width_ratio_dict[c] = width_ASGD_np/width_lm_np
        HulC_ASGD_width_ratio_dict[c] = width_Hulc_np/width_ASGD_np
        HulC_tstat_width_ratio_dict[c] = width_Hulc_np/width_tstat_np
        HulC_lm_width_ratio_dict[c] = width_Hulc_np/width_lm_np
        tstat_lm_width_ratio_dict[c] = width_tstat_np/width_lm_np
    
        # Minimum widths
        width_ASGD_min_dict[c] = np.nanmin(width_ASGD_np, axis = 0)
        width_lm_min_dict[c] = np.nanmin(width_lm_np, axis = 0)
        width_Hulc_min_dict[c] = np.nanmin(width_Hulc_np, axis = 0)
        width_tstat_min_dict[c] = np.nanmin(width_tstat_np, axis = 0)
    
        # Maximum widths
        width_ASGD_max_dict[c] = np.nanmax(width_ASGD_np, axis = 0)
        width_lm_max_dict[c] = np.nanmax(width_lm_np, axis = 0)
        width_Hulc_max_dict[c] = np.nanmax(width_Hulc_np, axis = 0)
        width_tstat_max_dict[c] = np.nanmax(width_tstat_np, axis = 0)
    
        # Mean width ratios    
        ASGD_lm_width_ratio_avg_dict[c] = np.nanmean(ASGD_lm_width_ratio_dict[c], axis = 0)
        HulC_ASGD_width_ratio_avg_dict[c] = np.nanmean(HulC_ASGD_width_ratio_dict[c], axis = 0)
        HulC_tstat_width_ratio_avg_dict[c] = np.nanmean(HulC_tstat_width_ratio_dict[c], axis = 0)
        HulC_lm_width_ratio_avg_dict[c] = np.nanmean(HulC_lm_width_ratio_dict[c], axis = 0)
        tstat_lm_width_ratio_avg_dict[c] = np.nanmean(tstat_lm_width_ratio_dict[c], axis = 0)
    
        # Median width ratios        
        ASGD_lm_width_ratio_med_dict[c] = np.nanmedian(ASGD_lm_width_ratio_dict[c], axis = 0)
        HulC_ASGD_width_ratio_med_dict[c] = np.nanmedian(HulC_ASGD_width_ratio_dict[c], axis = 0)
        HulC_tstat_width_ratio_med_dict[c] = np.nanmedian(HulC_tstat_width_ratio_dict[c], axis = 0)
        HulC_lm_width_ratio_med_dict[c] = np.nanmedian(HulC_lm_width_ratio_dict[c], axis = 0)
        tstat_lm_width_ratio_med_dict[c] = np.nanmedian(tstat_lm_width_ratio_dict[c], axis = 0)
    
        #Min width ratios        
        ASGD_lm_width_ratio_min_dict[c] = np.nanmin(ASGD_lm_width_ratio_dict[c], axis = 0)
        HulC_ASGD_width_ratio_min_dict[c] = np.nanmin(HulC_ASGD_width_ratio_dict[c], axis = 0)
        HulC_tstat_width_ratio_min_dict[c] = np.nanmin(HulC_tstat_width_ratio_dict[c], axis = 0)
        HulC_lm_width_ratio_min_dict[c] = np.nanmin(HulC_lm_width_ratio_dict[c], axis = 0)
        tstat_lm_width_ratio_min_dict[c] = np.nanmin(tstat_lm_width_ratio_dict[c], axis = 0)
        
        #Max width ratios        
        ASGD_lm_width_ratio_max_dict[c] = np.nanmax(ASGD_lm_width_ratio_dict[c], axis = 0)
        HulC_ASGD_width_ratio_max_dict[c] = np.nanmax(HulC_ASGD_width_ratio_dict[c], axis = 0)
        HulC_tstat_width_ratio_max_dict[c] = np.nanmax(HulC_tstat_width_ratio_dict[c], axis = 0)
        HulC_lm_width_ratio_max_dict[c] = np.nanmax(HulC_lm_width_ratio_dict[c], axis = 0)
        tstat_lm_width_ratio_max_dict[c] = np.nanmax(tstat_lm_width_ratio_dict[c], axis = 0)
    
    
    

    
    
    
    for c in c_grid:

        if output=="WASGD":
            c_effective = 1
        else:
            c_effective = c

        #Long version (tidy)
        
        fields_long =['date','N','S','D', 'task', 'output', 'c', 'alpha_lr', "burn_in", 'burn_in_threshold',
                'initializer', 'epochs_for_HulC', 'fixed_step', 'cov_type',
                    'CI_type',
                    'theta_k',
                    'coverage',
                    'coverage_excluding_nans',
                    'count_of_nans',
                    
                    'width_avg', 'width_median', 
                    'width_min', 'width_max',
                    
                    'HulC-CI_type_width_ratio_avg', 'HulC-CI_type_width_ratio_median',
                    'HulC-CI_type_width_ratio_min', 'HulC-CI_type_width_ratio_max',
                    
                    'CI_type_OLS_width_ratio_avg', 'CI_type_OLS_width_ratio_median',
                    'CI_type_OLS_width_ratio_min', 'CI_type_OLS_width_ratio_max',
                    
                    'avg_runtime',
                    
                    'ASGD_notes', 'HulC_notes', 'tstat_notes', 'OLS_notes' ]
    
        CI_types = ["ASGD", "OLS", "Hulc", "tstat"]
        CI_types_long = []
        for x in CI_types:
            CI_types_long += [x]*D
        
        D_range = np.tile(np.linspace(0,D-1, num = D).astype("int32"), len(CI_types))
        
        r = len(CI_types_long) #number of rows
    
        assert r==len(D_range)
    
        #Make SURE all these vectors obey the order of CI_types above: "ASGD", "OLS", "Hulc", "tstat"
        coverage = np.concatenate((ASGD_coverage_rate_dict[c], OLS_coverage_rate_dict[c],
                                   HulC_coverage_rate_dict[c], tstat_coverage_rate_dict[c]))
        coverage_without_nans = np.concatenate((ASGD_coverage_rate_without_nans_dict[c], OLS_coverage_rate_without_nans_dict[c],
                                   HulC_coverage_rate_without_nans_dict[c], tstat_coverage_rate_without_nans_dict[c]))
        count_of_nans = np.concatenate((num_NaNs_ASGD[c], num_NaNs_lm[c],
                                   num_NaNs_Hulc[c], num_NaNs_tstat[c]))
        width_avg = np.concatenate((width_ASGD_avg_dict[c], width_lm_avg_dict[c],
                                    width_hulc_avg_dict[c], width_tstat_avg_dict[c]))
        width_median = np.concatenate((width_ASGD_median_dict[c], width_lm_median_dict[c],
                                       width_hulc_median_dict[c], width_tstat_median_dict[c]))
        width_min = np.concatenate((width_ASGD_min_dict[c], width_lm_min_dict[c],
                                    width_Hulc_min_dict[c], width_tstat_min_dict[c]))
        width_max = np.concatenate((width_ASGD_max_dict[c], width_lm_max_dict[c],
                                    width_Hulc_max_dict[c], width_tstat_max_dict[c]))
        
        assert len(coverage)==D*4
        assert len(coverage_without_nans)==D*4
        assert len(count_of_nans)==D*4
        assert len(width_avg)==D*4
        assert len(width_median)==D*4
        assert len(width_min)==D*4
        assert len(width_max)==D*4
        
        HulC_CI_type_width_ratio_avg = np.concatenate((HulC_ASGD_width_ratio_avg_dict[c],
                                                       HulC_lm_width_ratio_avg_dict[c],
                                                       np.repeat(1,D),
                                                       HulC_tstat_width_ratio_avg_dict[c]))
        HulC_CI_type_width_ratio_median = np.concatenate((HulC_ASGD_width_ratio_med_dict[c],
                                                          HulC_lm_width_ratio_med_dict[c], np.repeat(1,D),
                                                          HulC_tstat_width_ratio_med_dict[c]))
        HulC_CI_type_width_ratio_min = np.concatenate((HulC_ASGD_width_ratio_min_dict[c],
                                                       HulC_lm_width_ratio_min_dict[c],
                                                       np.repeat(1,D), 
                                                       HulC_tstat_width_ratio_min_dict[c]))
        HulC_CI_type_width_ratio_max = np.concatenate((HulC_ASGD_width_ratio_max_dict[c],
                                                       HulC_lm_width_ratio_max_dict[c],
                                                       np.repeat(1,D),
                                                       HulC_tstat_width_ratio_max_dict[c]))
    
        assert len(HulC_CI_type_width_ratio_avg)==D*4
        assert len(HulC_CI_type_width_ratio_median)==D*4
        assert len(HulC_CI_type_width_ratio_min)==D*4
        assert len(HulC_CI_type_width_ratio_max)==D*4
        
        CI_type_OLS_width_ratio_avg = np.concatenate((ASGD_lm_width_ratio_avg_dict[c],
                                                      np.repeat(1,D),
                                                      HulC_lm_width_ratio_avg_dict[c],
                                                      tstat_lm_width_ratio_avg_dict[c]))
        CI_type_OLS_width_ratio_median = np.concatenate((ASGD_lm_width_ratio_med_dict[c],
                                                         np.repeat(1,D),
                                                         HulC_lm_width_ratio_med_dict[c],
                                                         tstat_lm_width_ratio_med_dict[c]))
        CI_type_OLS_width_ratio_min = np.concatenate((ASGD_lm_width_ratio_min_dict[c],
                                                      np.repeat(1,D),
                                                      HulC_lm_width_ratio_min_dict[c],
                                                      tstat_lm_width_ratio_min_dict[c]))
        CI_type_OLS_width_ratio_max = np.concatenate((ASGD_lm_width_ratio_max_dict[c],
                                                      np.repeat(1,D),
                                                      HulC_lm_width_ratio_max_dict[c],
                                                      tstat_lm_width_ratio_max_dict[c]))
        
        
        assert len(CI_type_OLS_width_ratio_avg)==D*4
        assert len(CI_type_OLS_width_ratio_median)==D*4
        assert len(CI_type_OLS_width_ratio_min)==D*4
        assert len(CI_type_OLS_width_ratio_max)==D*4
        
        avg_runtime = np.concatenate((np.repeat(runtime_ASGD_avg_dict[c], D),
                                      np.repeat(runtime_OLS_avg_dict[c], D),
                                      np.repeat(runtime_HulC_avg_dict[c], D),
                                      np.repeat(runtime_tstat_avg_dict[c], D)))
        
        
        
        rows_long = [ [date]*r, [N]*r, [S]*r, [D]*r, [model_type]*r, [output]*r, [c_effective]*r,
                           [alpha_lr]*r, [burn_in]*r, [burn_in_threshold]*r, [initializer]*r,
                           [epochs_for_HulC]*r, [fixed_step]*r, [cov_type]*r,
                           CI_types_long,
                           D_range,
                           coverage,
                           coverage_without_nans,
                           count_of_nans,
                           
                           width_avg, width_median,
                           width_min, width_max,
                           
                           HulC_CI_type_width_ratio_avg, HulC_CI_type_width_ratio_median,
                           HulC_CI_type_width_ratio_min, HulC_CI_type_width_ratio_max,
                           
                           CI_type_OLS_width_ratio_avg, CI_type_OLS_width_ratio_median,
                           CI_type_OLS_width_ratio_min, CI_type_OLS_width_ratio_max,
                           
                           avg_runtime,
                           
                           [ASGD_nan_flag_dict[c]]*r, [Hulc_nan_flag_dict[c]]*r,
                           [tstat_nan_flag_dict[c]]*r, [OLS_nan_flag_dict[c]]*r\
                           ]
    
        
        assert len(rows_long)==len(fields_long)
        row_dict_long = dict(zip(fields_long, rows_long))
        
        df = pd.DataFrame(data=row_dict_long)
        
       
        
        file_exists = os.path.isfile('Hulc_sims_long.csv')
        if file_exists==False:
            df.to_csv('Hulc_sims_long.csv', na_rep='')
        if file_exists==True:
            df.to_csv('Hulc_sims_long.csv', header=None, mode='a', na_rep='')
    
    
#%%

run_once = False
if run_once==True:
    model_type = "Logistic" # "Logistic" or "OLS"
    alpha_level = .05 #Type  I error rate
    
    S = 5 #number of loops (Chen et al in Table 1 use 200)
    N = 10**3 #number of samples per loop
    D = 100 #Dimension of linear regression (1 intercept + D-1 slopes)
    
    ytype = "neg11" #if model_type=="Logistic", options are "neg11" (y is in {-1,1}) or "01" (y is in {0,1})
    output = "ASGD" #options are "ASGD", "WASGD", or "SGD"
    cov_type = "Toeplitz" #options are "I", "Toeplitz", "EquiCorr" (this controls out X is generated; se gld.gen_normal0_data)
    
    #How to generate the data
    XY_type = "normal_0" # only one option ("normal_0") currently, could be made more flexible.
    
    
    #Parameters if running ASGD
    c_grid = [10]  
    alpha_lr = .505
    burn_in=False
    burn_in_threshold = 1
    initializer = True
    epochs_for_HulC = 1
    fixed_step = False
    
    #Parameters if running SGD
    # adjust c and epochs_for_HulC above
    
    save_plots = False

    np.random.seed(12)
    run_sims(model_type, S, N, D, ytype, output, cov_type, alpha_level, XY_type, c_grid,
             alpha_lr, burn_in, burn_in_threshold, initializer, epochs_for_HulC,
             fixed_step, save_plots)
    

#%%
#Loop over N_grid, cov_type_grid, cov_type_grid
loop = True
if loop==True:
    np.random.seed(123)
    model_type = "Logistic"  # "Logistic" or "OLS"
    
    alpha_level = .05 #Type  I error rate
    
    S = 200 #number of loops (Chen et al in Table 1 use 200)
    D = 100 #Dimension of linear regression (1 intercept + D-1 slopes)
    ytype = "neg11" 
    output = "ASGD"
    #How to generate the data
    XY_type = "normal_0"
    
    
    #Parameters if running ASGD
    alpha_lr = .505
    burn_in=False
    burn_in_threshold = 1
    initializer = True
    epochs_for_HulC = 1
    fixed_step = False
    save_plots = False
    
    
    #c_grid = [.01, .05, .1, .2, .5, .75, 1, 1.5, 2]    #Use this for D<100
    c_grid = [.05, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2]  #Use this for D=100
    #N_grid = [10**3, 10**4, 5*10**4, 10**5] 
    N_grid = [10**3] 
    cov_type_grid = ["EquiCorr"] #["I", "Toeplitz", "EquiCorr"]
     
    
    for N in N_grid:
        for cov_type in cov_type_grid:
            print(model_type + ", D = " + str(D) + ", c = "+ str(c_grid) + ", N = " + str(N) + ", cov_type = " + cov_type)
            run_sims(model_type, S, N, D, ytype, output, cov_type, alpha_level, XY_type, c_grid, alpha_lr, burn_in,\
                              burn_in_threshold, initializer, epochs_for_HulC, fixed_step, save_plots)  
    
    