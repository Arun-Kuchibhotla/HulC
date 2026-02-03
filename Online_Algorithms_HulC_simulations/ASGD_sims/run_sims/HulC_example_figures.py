# -*- coding: utf-8 -*-
"""
Created on Nov 1, 2024

@author: Selina Carter

This file generates linear & logistic regression data in multiple dimensions
and performs ASGD for example figures. The goal is to show that ASGD accuracy
is sensitive to hyperparameter c.

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D #Creates a custom legend in plots

import os
run_locally=True
if run_locally:
    os.chdir("C://Users//selin//Dropbox//Hulc_simulations//hulc_simulations//HulC_python//ASGD_simulations//run_sims")
#os.chdir("//home//shcarter//Hulc//simulations//plots")

import gen_data as gld
import ASGD_HulC_manual as ahm

#%%
#############################
### Linear regression #######
#############################


np.random.seed(2)
XY = gld.gen_normal0_data(n=1*10**3, d=5, cov_type="I") #cov_type = I, Toeplitz, EquiCorr


#Example of bad initialization procedure (initializer = False)
np.random.seed(2) 
#Theta dimension 0
theta, theta_bar, A_bar, S_bar, n = ahm.ASGD_CI_lin_reg_manual(XY, output="ASGD", c =.01, alpha_lr = .505,\
                                                           burn_in = False,\
                                                       burn_in_threshold = 1, \
                                                initializer = False,  plot_theta_dim=0,\
                                                    plot=True, verbose = False,\
                                                        epochs = 1, fixed_step = False, shift_k=True, plot_footnote=False) 

np.random.seed(2) 
#Theta dimension 4
theta, theta_bar, A_bar, S_bar, n = ahm.ASGD_CI_lin_reg_manual(XY, output="ASGD", c =.01, alpha_lr = .505,\
                                                           burn_in = False,\
                                                       burn_in_threshold = 1, \
                                                initializer = False,  plot_theta_dim=4,\
                                                    plot=True, verbose = False,\
                                                        epochs = 1, fixed_step = False, shift_k=True, plot_footnote=False) 







#Example of non-convergence (when hyperparameter c is too small)
np.random.seed(2) 
#Theta dimension 0
#Saved as ASGD_n1000_smallc_theta1.png
theta, theta_bar, A_bar, S_bar, n = ahm.ASGD_CI_lin_reg_manual(XY, output="ASGD", c =.01, alpha_lr = .505,\
                                                           burn_in = False,\
                                                       burn_in_threshold = 1, \
                                                initializer = True,  plot_theta_dim=0,\
                                                    plot=True, verbose = False,\
                                                        epochs = 1, fixed_step = False, shift_k=True, plot_footnote=False) 

np.random.seed(2) #Theta dimension 5
#Saved as ASGD_n1000_smallc_theta5.png
theta, theta_bar, A_bar, S_bar, n = ahm.ASGD_CI_lin_reg_manual(XY, output="ASGD", c =.01, alpha_lr = .505,\
                                                           burn_in = False,\
                                                       burn_in_threshold = 1, \
                                                initializer = True,  plot_theta_dim=4,\
                                                    plot=True, verbose = False,\
                                                        epochs = 1, fixed_step = False, shift_k=True, plot_footnote=False) 
    

#We show this bias is systematic
D=5
theta_bar_i = []
theta_i = []
reps = 10000
for i in range(reps):
    np.random.seed(i + 100)
    XY = gld.gen_normal0_data(n=1*10**3, d=D, cov_type="I") #cov_type = I, Toeplitz, EquiCorr
    theta, theta_bar, A_bar, S_bar, n = ahm.ASGD_CI_lin_reg_manual(XY, c = .01, burn_in = False,\
                                                               burn_in_threshold = 1, \
                                                        initializer = True,  plot_theta_dim=0,\
                                                            plot=False, verbose = False,\
                                                                epochs = 1, fixed_step = False,\
                                                                    shift_k=True, plot_footnote=False)
    theta_bar_i.append(theta_bar)
    theta_i.append(theta)
    
print(np.min(theta_bar_i, axis = 0))   
print(np.max(theta_bar_i, axis = 0))   
print(np.mean(theta_bar_i, axis = 0)) 


true_theta = np.linspace(0, 1, num=D)
# These are saved as ASGD_n1000_smallc_theta1_hist.png and ASGD_n1000_smallc_theta5_hist.png
for p in range(D):
    p_to_print = p +1
    plt.hist(np.array(theta_bar_i)[:,p], bins=20, edgecolor='black', \
             color = "pink", alpha = .5)  
    plt.axvline(true_theta[p], color='green', linestyle='-', alpha = 1, lw=2)
    plt.ylabel('Frequency')
    plt.xlabel(r'$e^\top_'+ str(p_to_print)+ r'\bar{\theta}_T$')
    #plt.xlabel(r'$\bar{\theta}_n^{(' + str(p_to_print) + ")}$")
    plt.title(r'(Linear regression) ASGD estimates of $e^\top_'+str(p_to_print) +\
              r'\theta_{\infty}$='+str(true_theta[p]) + ' (' + str(reps) + ' simulations)')
    #plt.title(r'(Linear regression) ASGD estimates of $\theta_{\infty}^{(' +\
    #          str(p_to_print) + r')}$='+str(true_theta[p]) + ' (' + str(reps) + ' simulations)')
    
    custom_lines = [Line2D([0], [0], color="pink", lw=10),
                    Line2D([0], [0], color="green", lw=2)]
    if p_to_print<D:
        plt.legend(custom_lines, [r'ASGD $e^\top_'+str(p_to_print)+r'\bar{\theta}_T$',\
                                  r'Target $e^\top_'+str(p_to_print) +r'\theta_{\infty}$=' + str(true_theta[p])],\
                   labelcolor=["red", "green"],\
                   loc = "upper right")   
    if p_to_print==D:
        plt.legend(custom_lines, [r'ASGD $e^\top_'+str(p_to_print)+r'\bar{\theta}_T$',\
                                  r'Target $e^\top_'+str(p_to_print) +r'\theta_{\infty}$=' + str(true_theta[p])],\
                   labelcolor=["red", "green"],\
                   loc = "upper center")   
    
    
    if p_to_print==1:
        plt.savefig('plots//ASGD progress//ASGD_n1000_smallc_theta1_hist.pdf',\
                        bbox_inches='tight')
        plt.show()
    if p_to_print==5:
        plt.savefig('plots//ASGD progress//ASGD_n1000_smallc_theta5_hist.pdf',\
                        bbox_inches='tight')
        plt.show()
    if p_to_print!=1 and p_to_print!=5:
        plt.show()
    
    
#%%
#Example of off-target (when hyperparameter c is too big)

np.random.seed(305) #Theta dimension 0
XY = gld.gen_normal0_data(n=1*10**3, d=5, cov_type="I") #cov_type = I, Toeplitz, EquiCorr
#saved as ASGD_n1000_largec_theta1.png
theta, theta_bar, A_bar, S_bar, n = ahm.ASGD_CI_lin_reg_manual(XY, output="ASGD", c =2, alpha_lr = .505,\
                                                           burn_in = False,\
                                                       burn_in_threshold = 1, \
                                                initializer = True,  plot_theta_dim=0,\
                                                    plot=True, verbose = False,\
                                                        epochs = 1, fixed_step = False, shift_k=True, plot_footnote=False) 
np.random.seed(305) #Theta dimension 4
#saved as ASGD_n1000_largec_theta5.png
theta, theta_bar, A_bar, S_bar, n = ahm.ASGD_CI_lin_reg_manual(XY, output="ASGD", c =2, alpha_lr = .505,\
                                                           burn_in = False,\
                                                       burn_in_threshold = 1, \
                                                initializer = True,  plot_theta_dim=4,\
                                                    plot=True, verbose = False,\
                                                        epochs = 1, fixed_step = False, shift_k=True, plot_footnote=False) 


#We show this bias is systematic
D=5
theta_bar_i = []
theta_i = []
reps = 10000
for i in range(reps):
    np.random.seed(i + 100)

    XY = gld.gen_normal0_data(n=1*10**3, d=D, cov_type="I") #cov_type = I, Toeplitz, EquiCorr
    theta, theta_bar, A_bar, S_bar, n = ahm.ASGD_CI_lin_reg_manual(XY, c = 2, burn_in = False,\
                                                               burn_in_threshold = 1, \
                                                        initializer = True,  plot_theta_dim=0,\
                                                            plot=False, verbose = False,\
                                                                epochs = 1, fixed_step = False,\
                                                                    shift_k=True, plot_footnote=False)
    theta_bar_i.append(theta_bar)
    theta_i.append(theta)
    
print(np.min(theta_bar_i, axis = 0))   
print(np.max(theta_bar_i, axis = 0))       

print(np.mean(theta_bar_i, axis = 0))

np.where(np.array(theta_bar_i)[:,0]==np.min(theta_bar_i, axis = 0)[0])




# These are saved as ASGD_n1000_largec_theta1_hist.png and ASGD_n1000_largec_theta5_hist.png

exclude_huge_estimates = True #If you set to False, histogram looks like one tall skyscraper due to outliers.
for p in range(D):
    
    if exclude_huge_estimates==False:
        theta_p_to_plot = np.array(theta_bar_i)[:,p]
    if exclude_huge_estimates==True:
        theta_p = np.array(theta_bar_i)[:,p]
        sd = np.sqrt(np.var(theta_p))
        middle_theta_p = np.median(theta_p)
        lb = np.quantile(theta_p, q=.005)
        ub = np.quantile(theta_p, q=.995)
        theta_p_to_plot = theta_p[np.where((lb<=theta_p) & (theta_p<=ub)) ]
        
    p_to_print = p +1
    plt.hist(theta_p_to_plot, bins=20, edgecolor='black', \
             color = "pink", alpha = .5)  
    plt.axvline(true_theta[p], color='green', linestyle='-', alpha = 1, lw=2)
    plt.ylabel('Frequency')
    plt.xlabel(r'$e^\top_'+ str(p_to_print)+ r'\bar{\theta}_T$')
    #plt.xlabel(r'$\bar{\theta}_n^{(' + str(p_to_print) + ")}$")
    #plt.title(r'(Linear regression) ASGD estimates of $\theta_{\infty}^{(' +\
    #          str(p_to_print) + r')}$='+str(true_theta[p]) + ' (' + str(reps) + ' simulations)')
    plt.title(r'(Linear regression) ASGD estimates of $e^\top_'+str(p_to_print) +\
              r'\theta_{\infty}$='+str(true_theta[p]) + ' (' + str(reps) + ' simulations)')
             
    custom_lines = [Line2D([0], [0], color="pink", lw=10),
                    Line2D([0], [0], color="green", lw=2)]

    plt.legend(custom_lines, [r'ASGD $e^\top_'+str(p_to_print)+r'\bar{\theta}_T$',\
                              r'Target $e^\top_'+str(p_to_print) +r'\theta_{\infty}$=' + str(true_theta[p])],\
               labelcolor=["red", "green"])   
    if p_to_print==1:
        plt.savefig('plots//ASGD progress//ASGD_n1000_largec_theta1_hist.pdf',\
                        bbox_inches='tight')
        plt.show()
    if p_to_print==5:
        plt.savefig('plots//ASGD progress//ASGD_n1000_largec_theta5_hist.pdf',\
                        bbox_inches='tight')
        plt.show()
    if p_to_print!=1 and p_to_print!=5:
        plt.show()
 
    
#%%      
    
#Example "Goldilocks zone" (when hyperparameter c is medium size)

np.random.seed(305) #Theta dimension 0
XY = gld.gen_normal0_data(n=1*10**3, d=5, cov_type="I") #cov_type = I, Toeplitz, EquiCorr
#saved as ASGD_n1000_goodc_theta1.png
theta, theta_bar, A_bar, S_bar, n = ahm.ASGD_CI_lin_reg_manual(XY, output="ASGD", c =.5, alpha_lr = .505,\
                                                           burn_in = False,\
                                                       burn_in_threshold = 1, \
                                                initializer = True,  plot_theta_dim=0,\
                                                    plot=True, verbose = False,\
                                                        epochs = 1, fixed_step = False, shift_k=True, plot_footnote=False) 
np.random.seed(305) #Theta dimension 4
#saved as ASGD_n1000_goodc_theta5.png
theta, theta_bar, A_bar, S_bar, n = ahm.ASGD_CI_lin_reg_manual(XY, output="ASGD", c =.5, alpha_lr = .505,\
                                                           burn_in = False,\
                                                       burn_in_threshold = 1, \
                                                initializer = True,  plot_theta_dim=4,\
                                                    plot=True, verbose = False,\
                                                        epochs = 1, fixed_step = False, shift_k=True, plot_footnote=False) 


#We show this bias is systematic
D=5
theta_bar_i = []
theta_i = []
reps = 10000
for i in range(reps):
    np.random.seed(i + 100)

    XY = gld.gen_normal0_data(n=1*10**3, d=D, cov_type="I") #cov_type = I, Toeplitz, EquiCorr
    theta, theta_bar, A_bar, S_bar, n = ahm.ASGD_CI_lin_reg_manual(XY, c = .5, burn_in = False,\
                                                               burn_in_threshold = 1, \
                                                        initializer = True,  plot_theta_dim=0,\
                                                            plot=False, verbose = False,\
                                                                epochs = 1, fixed_step = False,\
                                                                    shift_k=True, plot_footnote=False)
    theta_bar_i.append(theta_bar)
    theta_i.append(theta)
    
print(np.min(theta_bar_i, axis = 0))   
print(np.max(theta_bar_i, axis = 0))   
print(np.mean(theta_bar_i, axis = 0))       

np.where(np.array(theta_bar_i)[:,0]==np.min(theta_bar_i, axis = 0)[0])

# These are saved as ASGD_n1000_goodc_theta1_hist.png and ASGD_n1000_goodc_theta5_hist.png

exclude_huge_estimates = True #If you set to False, histogram looks like one tall skyscraper due to outliers.
for p in range(D):
    

    p_to_print = p +1
    plt.hist(np.array(theta_bar_i)[:,p], bins=20, edgecolor='black', \
             color = "pink", alpha = .5)  
    plt.axvline(true_theta[p], color='green', linestyle='-', alpha = 1, lw=2)
    plt.ylabel('Frequency')
    plt.xlabel(r'$\bar{\theta}_n^{(' + str(p_to_print) + ")}$")
    plt.xlabel(r'$e^\top_'+ str(p_to_print)+ r'\bar{\theta}_T$')
    #plt.xlabel(r'$\bar{\theta}_n^{(' + str(p_to_print) + ")}$")
    #plt.title(r'(Linear regression) ASGD estimates of $\theta_{\infty}^{(' +\
    #          str(p_to_print) + r')}$='+str(true_theta[p]) + ' (' + str(reps) + ' simulations)')
    plt.title(r'(Linear regression) ASGD estimates of $e^\top_'+str(p_to_print) +\
              r'\theta_{\infty}$='+str(true_theta[p]) + ' (' + str(reps) + ' simulations)')
    
    custom_lines = [Line2D([0], [0], color="pink", lw=10),
                    Line2D([0], [0], color="green", lw=2)]

    plt.legend(custom_lines, [r'ASGD $e^\top_'+str(p_to_print)+r'\bar{\theta}_T$',\
                              r'Target $e^\top_'+str(p_to_print) +r'\theta_{\infty}$=' + str(true_theta[p])],\
               labelcolor=["red", "green"])   
    if p_to_print==1:
        plt.savefig('plots//ASGD progress//ASGD_n1000_goodc_theta1_hist.pdf',\
                        bbox_inches='tight')
        plt.show()
    if p_to_print==5:
        plt.savefig('plots//ASGD progress//ASGD_n1000_goodc_theta5_hist.pdf',\
                        bbox_inches='tight')
        plt.show()
    if p_to_print!=1 and p_to_print!=5:
        plt.show()
    
        

#%%
#############################
### Logistic regression #######
#############################


np.random.seed(2)
XY = gld.gen_normal0_logistic_data(n=5*10**4, d=100, ytype="neg11", cov_type = "I", torch = False)

#Example of non-convergence (when hyperparameter c is too small)
np.random.seed(2) #Theta dimension 99
#Saved as ASGD_n1000_smallc_theta1.png
theta, theta_bar, A_bar, S_bar, n = ahm.ASGD_CI_log_reg_manual(XY, output="ASGD", c =.75, alpha_lr = .505,\
                                                           burn_in = False,\
                                                       burn_in_threshold = 1, \
                                                initializer = True,  plot_theta_dim=99,\
                                                    plot=True, verbose = False,\
                                                        epochs = 1, fixed_step = False, shift_k=True, plot_footnote=False) 


    
#We show this bias is systematic
D=100
N=5*10**4
c = .75
cov_type = "I"
theta_bar_i = []
theta_i = []
reps = 100
for i in range(reps):
    np.random.seed(i + 100)
    print(i)
    XY = gld.gen_normal0_logistic_data(n=N, d=D, ytype="neg11", cov_type = cov_type, torch = False)#cov_type = I, Toeplitz, EquiCorr
    theta, theta_bar, A_bar, S_bar, n = ahm.ASGD_CI_log_reg_manual(XY, c = c, burn_in = False,\
                                                               burn_in_threshold = 1, \
                                                        initializer = True,  plot_theta_dim=0,\
                                                            plot=False, verbose = False,\
                                                                epochs = 1, fixed_step = False,\
                                                                    shift_k=True, plot_footnote=False)
    theta_bar_i.append(theta_bar)
    theta_i.append(theta)
    
print(np.min(theta_bar_i, axis = 0))   
print(np.max(theta_bar_i, axis = 0))   
print(np.mean(theta_bar_i, axis = 0)) 


true_theta = np.linspace(0, 1, num=D)
# These are saved as ASGD_n1000_smallc_theta1_hist.png and ASGD_n1000_smallc_theta5_hist.png
for p in range(D):
    p_to_print = p +1
    plt.hist(np.array(theta_bar_i)[:,p], bins=20, edgecolor='black', \
             color = "pink", alpha = .5)  
    plt.axvline(true_theta[p], color='green', linestyle='-', alpha = 1, lw=2)
    plt.ylabel('Frequency')
    plt.xlabel(r'$e^\top_'+ str(p_to_print)+ r'\bar{\theta}_T$')
    plt.title(r'(Logistic regression) ASGD estimates of $e^\top_'+str(p_to_print) +\
              r'\theta_{\infty}$='+str(true_theta[p]) + ' (' + str(reps) + ' simulations)')
    
    custom_lines = [Line2D([0], [0], color="pink", lw=10),
                    Line2D([0], [0], color="green", lw=2)]
    if p_to_print<D:
        plt.legend(custom_lines, [r'ASGD $e^\top_'+str(p_to_print)+r'\bar{\theta}_T$',\
                                  r'Target $e^\top_'+str(p_to_print) +r'\theta_{\infty}$=' + str(true_theta[p])],\
                   labelcolor=["red", "green"],\
                   loc = "upper right")   
    if p_to_print==D:
        plt.legend(custom_lines, [r'ASGD $e^\top_'+str(p_to_print)+r'\bar{\theta}_T$',\
                                  r'Target $e^\top_'+str(p_to_print) +r'\theta_{\infty}$=' + str(true_theta[p])],\
                   labelcolor=["red", "green"],\
                   loc = "upper center")   
    plt.annotate("N = " + str(N) + ", D = " + str(D) + ", c = " + str(c) + ", cov_type = " + cov_type,
            xy = (1.0, -0.2),
            xycoords='axes fraction',
            ha='right',
            va="center",
            color = "gray",
            fontsize=10)
    plt.show()
    
