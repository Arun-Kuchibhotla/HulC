packages_to_check <- c("sgd", "sandwich", "tidyverse", "assertthat")
installed_pkgs <- rownames(installed.packages())

for (pkg in packages_to_check) {
  if (!(pkg %in% installed_pkgs)) {
    message(paste("Package", pkg, "not found. Installing..."))
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }  else {
    library(pkg, character.only = TRUE)
  }
}

#Change to your preferred local or remote working directories
local=T
if (local){
  setwd("C:/Users/selin/Dropbox/Hulc_simulations/hulc_simulations/HulC_R")
} else{
  setwd("/home/shcarter/Hulc/simulations/R_sims")
}



source("gen_data.R")
source("HulC_sgd_functions.R")








run_sims = function(model_type = "OLS", S=100, N=10**3, D=5, 
                    method = "ai-sgd", cov_type = "I", alpha_level = .05, 
                    c_grid = c(0.01, 0.05, 0.1,  0.2,  0.5, 0.75, 1, 1.5, 2), alpha_lr = .505,
                    epsilon=.8, beta=.25, sigma=1,
                    initializer = T, epochs_for_HulC = 1,
                    fixed_step = F,  verbose=T, csv_name = "extra_sims.csv",
                    use_all_wald_estimates=F){
  
  
  # model_type = "Logistic" or "OLS"

  # S  = number of loops (Chen et al in Table 1 use 500)
  # N  = number of samples per loop
  # D = dimension of linear regression (1 intercept + D-1 slopes)
  
  # method = "ai-sgd" #options are "ai-sgd" (Averaged Implicit SGD), "implicit" (Implicit SGD), "sgd", or other methods in sgd::sgd() in argument sgd.control(method=); as well as "root-sgd", "truncated-sgd", "noisy-truncated-sgd"
  # cov_type = "EquiCorr" #options are "I", "Toeplitz", "EquiCorr" (this controls out X is generated)
  # alpha_level = .05 #Type  I error rate 
  # c_grid: vector of hyperparameters c_ for the eta_t() function used to calculate the step size: eta_t(c_, alpha_lr) = c_*t^(-alpha_lr)
  # alpha_lr: The "alpha" parameter in the eta_t() function used to calculate the step size: eta_t(c_, alpha_lr) = c_*t^(-alpha_lr). Note this value should be between 0.5 and 1, non-inclusive.
  # epsilon: value in interval (0,1); used only for method="truncated-sgd" or "noisy-truncated-sgd"
  # sigma: >0; used only for method="noisy-truncated-sgd"
  # beta: value in (0, 0.5); used only for method="noisy-truncated-sgd"
  
  #Assertions
  assertthat::assert_that(model_type %in% c("OLS", "Logistic"))
  assertthat::assert_that(method %in% c("ai-sgd", "implicit", "sgd", "root-sgd", "truncated-sgd", "noisy-truncated-sgd"))
  assertthat::assert_that(cov_type %in% c("I", "EquiCorr", "Toeplitz"))
  
  #Set unnecessary arguments to NA
  if (!(method %in% c("truncated-sgd", "noisy-truncated-sgd"))){
    epsilon=NA
  }
  if (method!="noisy-truncated-sgd"){
    beta=NA
    sigma=NA
  }
  burn_in=F   #I haven't set up this argument, but it will be recorded in the csv file
  burn_in_threshold = NA #I haven't set up this argument, but it will be recorded in the csv file
  
  #Set data generating function 
  if (model_type == "Logistic"){
    gen_data_type = gen_normal0_logistic_data
  }
  if (model_type == "OLS"){
    gen_data_type = gen_normal0_data
  }
  
  #Define true theta vector
  start= 0 
  stop = 1
  true_thetas=seq(start, stop, length.out=D)
  
  #Print every s_print steps (a positive integer)
  s_print = 1

  #If estimated thetas are above nan_threshold (in absolute value),
  # then convert estimate to nan
  nan_threshold = 1e+10
  
  adjust_c = F
  if (adjust_c==T){
    #For large dimension D, we mustn't let c get too large. So we adjust it.
    c_grid_new = c()
    if (D>50){
      for (c_ in c_grid){
        c_new = c_*5/D
        c_grid_new = c(c_grid_new, c_new)
        c_grid = c_grid_new
      }
      
    }
  }

  
  # Initialize an empty dictionary to store results for each c
  empty_list_0 = list() #For each value of c, makes an S x 1 matrix
  for (c_ in c_grid){
    empty_list_0[[as.character(c_)]] = matrix(NA, nrow=S, ncol = 1)
  }
  
  empty_list_1 = list() #For each value of c, makes an S x D matrix
  for (c_ in c_grid){
    empty_list_1[[as.character(c_)]] = matrix(NA, nrow=S, ncol = D)
  }
  
  
  #Set loss function and control
  if (model_type=="Logistic"){
    grad_f=logistic_gradient
    if (method %in%  c("implicit", "sgd", "ai-sgd")){#these methods use the sgd package
      model="glm"
      model.control=list(family=binomial(link="logit"))
    }
  }else if(model_type=="OLS"){
    grad_f=OLS_gradient
    if (method %in%  c("implicit", "sgd", "ai-sgd")){#these methods use the sgd package
      model="lm"
      model.control=list(family=gaussian(link = "identity"))
    }
  }
  
  #OLS sandwich estimator & CI
  lm_theta = empty_list_1
  CI_lm_lb = empty_list_1
  CI_lm_ub = empty_list_1
  width_lm = empty_list_1
  coverage_lm = empty_list_1
  coverage_lm_with_nans = empty_list_1
  num_NaNs_lm = empty_list_1
  runtime_OLS = empty_list_0
  Wald_notes = empty_list_0
  
  #Hulc CI
  CI_hulc_lb = empty_list_1
  CI_hulc_ub = empty_list_1
  width_hulc = empty_list_1
  coverage_hulc = empty_list_1
  coverage_hulc_with_nans = empty_list_1
  num_NaNs_hulc = empty_list_1
  runtime_hulc = empty_list_0
  hulc_notes = empty_list_0
  
  
  #t-stat CI
  CI_tstat_lb = empty_list_1
  CI_tstat_ub = empty_list_1
  width_tstat = empty_list_1
  coverage_tstat = empty_list_1
  coverage_tstat_with_nans = empty_list_1
  num_NaNs_tstat = empty_list_1
  runtime_tstat = empty_list_0
  tstat_notes = empty_list_0
  
  
  
  
  date = now()
  
  
  for (s in 1:S){
    if (s%%s_print==0){
      print(paste0("s=", s, "/", S))
      print(paste0("  ", "method=", method, ", N=", N, ", D=", D, ", cov_type=", cov_type, ", initializer=", initializer))
    }
    
    #Draw samples
    set.seed(s)
    if (model_type=="Logistic"){
      XY = gen_data_type(n=N, d=D, ytype="01", cov_type = cov_type)
    }
    if (model_type=="OLS"){
      XY = gen_data_type(n=N, d=D, cov_type = cov_type)
    }
    
    #Check if X matrix has column of ones; if so, remove it (the sgd() and lm() functions automatically add an intercept term).
    if (sum(XY["X1"]) == N){
      XY_no_intercept = XY[names(XY)!="X1"]
    }
    
    X_ = XY[-1]
    Y_ = XY[1]
    
    
    #Initializer for theta (according to Chen et al 2016 method in personal email)
    if (initializer){
      if (verbose){
        print("          Running initializer...")
      }
      
      #Initialize theta to 0
      theta = rep(0,D)
      
      step = .001 #This is a fixed step size, and we perform vanilla SGD for the "intelligent" initialization.
      for (tt in 1:(floor(N/3))){
        #SGD
        theta_new = theta - step*grad_f(theta, as.matrix(X_[tt,]), as.matrix(Y_[tt,]))
        theta = theta_new
      } 
    } else {
      #Initialize theta to 0
      #theta = rep(0,D)
      theta = rnorm(D, 0, .001)
    } #End of intitializer

    if (verbose){
      print(paste0("     Starting theta = ", as.character(round(theta[1],5)), ", ... , ", as.character(round(theta[D],5))))
    }


    for (c_ in c_grid){ #Big loop for each value of c_
      if (s%%s_print==0 & verbose){
          print(paste0("     s=", s, ", N=", N, ", D=", D, ", method=", method, ", cov_type=", cov_type, ", c=", c_))
      }
      


      

      
      #########################################
      # Wald confidence intervals
      #########################################
      if (s%%s_print==0 & verbose){
        print("          Running Wald (linear model)...")
      }
      
      start_wald = now()

      if(use_all_wald_estimates==F){
          if (model_type == "OLS"){
            mod=tryCatch(expr={lm(Y ~ ., data=XY_no_intercept)},
                         warning = function(w) {
                           # This code block executes if a warning occurs
                           print(paste("A warning occurred:", w$message))
                           mod = "Failed Wald"
                         })
          } else if (model_type == "Logistic"){
            mod=tryCatch(expr={glm(Y ~ ., data=XY_no_intercept, family="binomial")},
                     warning = function(w) {
                       # This code block executes if a warning occurs
                       print(paste("A warning occurred:", w$message))
                       mod = "Failed Wald"
                       })
          }
          
          if (length(mod)==1){ #Case when glm doesn't work
            wald_theta = rep(NA, D)
            wald_sd_est = rep(NA, D)
            wald_CI_lb = rep(NA, D)
            wald_CI_ub = rep(NA, D)
            wald_covered = rep(0, D)
            width = rep(NA, D)
            num_nans = rep(1, D)
            lm_note = "NaNs produced in CI (probably singular matrix)"
          } else { #Case if glm works
            wald_theta <- unname(mod$coeff)
            wald_sd_est <- sqrt(diag(vcovHC(mod, type = "HC")))
            wald_CI_lb =  wald_theta - qnorm(1-alpha_level/2)*wald_sd_est
            wald_CI_ub =  wald_theta + qnorm(1-alpha_level/2)*wald_sd_est
            wald_covered = as.numeric(wald_CI_lb<= true_thetas & true_thetas<=wald_CI_ub)
            width = wald_CI_ub-wald_CI_lb
            num_nans = rep(0, D)
            lm_note=""
          }
      }else{
        if (model_type == "OLS"){
          mod=lm(Y ~ ., data=XY_no_intercept)
        } else if (model_type == "Logistic"){
          mod=glm(Y ~ ., data=XY_no_intercept, family="binomial")
        }
        wald_theta <- unname(mod$coeff)
        wald_sd_est <- sqrt(diag(vcovHC(mod, type = "HC")))
        wald_CI_lb =  wald_theta - qnorm(1-alpha_level/2)*wald_sd_est
        wald_CI_ub =  wald_theta + qnorm(1-alpha_level/2)*wald_sd_est
        wald_covered = as.numeric(wald_CI_lb<= true_thetas & true_thetas<=wald_CI_ub)
        width = wald_CI_ub-wald_CI_lb
        num_nans = rep(0, D)
        lm_note=""
      }
      
      
      #Store results
      lm_theta[[as.character(c_)]][s, ] = wald_theta
      CI_lm_lb[[as.character(c_)]][s, ] = wald_CI_lb
      CI_lm_ub[[as.character(c_)]][s, ] = wald_CI_ub
      width_lm[[as.character(c_)]][s, ] = width
      coverage_lm[[as.character(c_)]][s, ] = wald_covered
      coverage_lm_with_nans[[as.character(c_)]][s, ] = wald_covered
      
      num_NaNs_lm[[as.character(c_)]][s, ] = num_nans
      runtime_OLS[[as.character(c_)]][s] = as.numeric(now()-start_wald)
      Wald_notes[[as.character(c_)]][s] = lm_note
      
      
      #########################################
      # HulC CI's
      #########################################

      if (s%%s_print==0 & verbose){
        print(paste0("          Running HulC using ", method, "..."))
      }
      start_HulC = now()

      #Run HulC on the method
      if (method %in%  c("implicit", "sgd", "ai-sgd")){
        hulc_out <- sgd_HulC(alpha = alpha_level, randomize = TRUE,
                             formula=Y~., data=XY_no_intercept,
                             model=model, model.control=model.control,
                             sgd.control = list(method=method, lr = "one-dim",
                                                lr.control=c(scale=1, gamma=1, alpha=c_, c=alpha_lr), 
                                                #size=N, #Get a full matrix of all theta values over the N estimates
                                                start=theta, #Starting value of parameter estimates
                                                npasses=epochs_for_HulC, #the maximum number of passes over the data. 
                                                pass=T, #tol be ignored and run the algorithm for all of npasses
                                                verbose=F))
      }else if (method=="root-sgd"){
        hulc_out <- root_sgd_HulC(data=XY, grad_f=grad_f, alpha=alpha_level, randomize = TRUE,
                                  c_=c_,  alpha_lr=alpha_lr, 
                                  initial_theta=theta, plot_it=F,
                                  true_theta=true_thetas,
                                  epochs=epochs_for_HulC, fixed_step=fixed_step)
      }else if (method=="truncated-sgd"){
        hulc_out <- truncated_sgd_HulC(data=XY, grad_f=grad_f, alpha=alpha_level, randomize = TRUE,
                                       epsilon=epsilon, 
                                       c_=c_,  alpha_lr=alpha_lr, 
                                       initial_theta = theta, plot_it=F,
                                       true_theta=true_thetas,
                                       epochs=epochs_for_HulC, fixed_step=fixed_step)
      }else if (method=="noisy-truncated-sgd"){
        hulc_out <- noisy_truncated_sgd_HulC(data=XY, grad_f=grad_f, alpha=alpha_level, randomize = TRUE,
                                       epsilon=epsilon, sigma=sigma, beta=beta,
                                       c_=c_,  alpha_lr=alpha_lr, 
                                       initial_theta = theta, plot_it=F,
                                       true_theta=true_thetas,
                                       epochs=epochs_for_HulC, fixed_step=fixed_step)
      }
      
      
      

      hulc_lb = hulc_out$ci[,1]
      hulc_ub = hulc_out$ci[,2]
      #Don't save confidence bounds if values are too huge; replace with nan
      count_too_big = sum(c(abs(hulc_lb)>nan_threshold, abs(hulc_ub)>nan_threshold))
      num_NANs = sum(is.nan(hulc_lb), is.nan(hulc_ub))
      
      #print("##Line 333###")
      #print(paste0("count_too_big=", count_too_big))
      #print(paste0("num_NANs=", num_NANs))
      
      if (is.na(count_too_big)){
        count_too_big = S
      }
      
      if (count_too_big + num_NANs > 0){
        
        hulc_lb_nan = hulc_lb
        hulc_ub_nan = hulc_ub
        
        hulc_lb_nan[abs(hulc_lb)>nan_threshold] = NA
        hulc_ub_nan[abs(hulc_ub)>nan_threshold] = NA
        
        hulc_lb_nan[is.nan(hulc_lb_nan)] = NA
        hulc_ub_nan[is.nan(hulc_ub_nan)] = NA
        
        has_nan = abs(hulc_lb)>nan_threshold | abs(hulc_lb)>nan_threshold | is.na(hulc_lb) | is.na(hulc_ub)
        
        #If any h_CI value is nan, coverage is automatically False in coverage_Hulc
        coverage_hulc_with_nans_c = as.numeric(hulc_lb<= true_thetas & true_thetas<=hulc_ub)
        coverage_hulc_with_nans_c[has_nan] = 0
        
        
        hulc_note = paste0("CI bound > ", as.character(nan_threshold))
        
        num_nans = as.numeric(abs(hulc_lb)>nan_threshold) + as.numeric(abs(hulc_ub)>nan_threshold) + is.na(hulc_lb) + is.na(hulc_ub)
        num_nans[num_nans>0 | is.na(num_nans)] = 1
        
        
      } else {
        hulc_note = ""
        num_nans = rep(0, D)
        coverage_hulc_with_nans_c = as.numeric(hulc_lb<= true_thetas & true_thetas<=hulc_ub)

      }


      

      #Store results
      
      CI_hulc_lb[[as.character(c_)]][s, ] = hulc_lb
      CI_hulc_ub[[as.character(c_)]][s, ] = hulc_ub
      width_hulc[[as.character(c_)]][s, ] = hulc_ub-hulc_lb
      coverage_hulc[[as.character(c_)]][s, ] = as.numeric(hulc_lb<= true_thetas & true_thetas<=hulc_ub)

      coverage_hulc_with_nans[[as.character(c_)]][s, ] = coverage_hulc_with_nans_c
      
      num_NaNs_hulc[[as.character(c_)]][s, ] = num_nans
      runtime_hulc[[as.character(c_)]][s] = as.numeric(now()-start_HulC)
      hulc_notes[[as.character(c_)]][s] = hulc_note

    
      
      
      #If any of the output is nan, print a warning
      if (sum(num_nans)>0){
        print("#########################")
        print(paste0("NOTE: For HulC implementation of method=", method ,
                     " Nan values were produced for model_type=", model_type,
                     ", s=", as.character(s), ", N=", as.character(N), ", D=", as.character(D),
                     ", cov_type=", cov_type, ", c_=", as.character(c_), "."))
        print("We will save NA values for this iteration.")
        print("#########################")
      }
  
      
      
      
      
      #########################################
      # t-stat CI's
      #########################################
      
      
      if (s%%s_print==0 & verbose){
        print(paste0("          Computing t-stat confidence intervals..."))
      }
      
      start_tstat = now()
      
      batch_size = hulc_out$B
      t_threshold = qt(p=1-alpha_level/2, df = batch_size-1)
      
      
      ###Denominator###
      theta_bar_estimates = hulc_out$batch_estimates
      theta_bar_bar = apply(theta_bar_estimates, MARGIN=2, mean, na.rm = TRUE)
      theta_bar_bar_matrix = matrix(rep(theta_bar_bar, batch_size), nrow=batch_size, byrow=T)
      
      sq_devs = (theta_bar_estimates - theta_bar_bar_matrix)**2
      sum_sq_devs = apply(sq_devs, MARGIN=2,  sum)
      s_hat = sqrt((1/(batch_size-1))*sum_sq_devs)
      
      deviation = t_threshold*s_hat/sqrt(batch_size)
      
      #Lower bound
      tstat_lb = theta_bar_bar - deviation
      #Upper bound
      tstat_ub = theta_bar_bar + deviation

      #Don't save confidence bounds if values are too huge; replace with nan
      count_too_big = sum(c(abs(tstat_lb)>nan_threshold, abs(tstat_ub)>nan_threshold))
      num_NANs = sum(is.nan(tstat_lb), is.nan(tstat_ub))
      
      
      #print("##Line 430###")
      #print(paste0("count_too_big=", count_too_big))
      #print(paste0("num_NANs=", num_NANs))
      if (is.na(count_too_big)){
        count_too_big = S
      }

      if (count_too_big + num_NANs > 0){
        tstat_lb_nan = tstat_lb
        tstat_ub_nan = tstat_ub

        tstat_lb_nan[abs(tstat_lb)>nan_threshold] = NA
        tstat_ub_nan[abs(tstat_ub)>nan_threshold] = NA
        
        tstat_lb_nan[is.nan(tstat_lb_nan)] = NA
        tstat_ub_nan[is.nan(tstat_ub_nan)] = NA  
        
        
        has_nan = abs(tstat_lb_nan)>nan_threshold | abs(tstat_ub_nan)>nan_threshold | is.na(tstat_lb_nan) | is.na(tstat_ub_nan)
        
        #If any tstat_CI value is nan, coverage is automatically False in coverage_tstat
        coverage_tstat_with_nans_c = as.numeric(tstat_lb<= true_thetas & true_thetas<=tstat_ub)
        coverage_tstat_with_nans_c[has_nan] = 0
        
        
        tstat_note = paste0("CI bound > ", as.character(nan_threshold))
        
        num_nans = as.numeric(abs(tstat_lb_nan)>nan_threshold) + as.numeric(abs(tstat_ub_nan)>nan_threshold) + is.na(tstat_lb_nan) + is.na(tstat_ub_nan)
        num_nans[num_nans>0 | is.na(num_nans)] = 1
        
      } else {
        tstat_note = ""
        num_nans = rep(0, D)
        coverage_tstat_with_nans_c = as.numeric(tstat_lb<= true_thetas & true_thetas<=tstat_ub)
      }
      
      
      #Store results
      CI_tstat_lb[[as.character(c_)]][s, ] = tstat_lb
      CI_tstat_ub[[as.character(c_)]][s, ] = tstat_ub
      width_tstat[[as.character(c_)]][s, ] = tstat_ub-tstat_lb
      coverage_tstat[[as.character(c_)]][s, ] = as.numeric(tstat_lb<= true_thetas & true_thetas<=tstat_ub)
      coverage_tstat_with_nans[[as.character(c_)]][s, ] = coverage_tstat_with_nans_c
      

      num_NaNs_tstat[[as.character(c_)]][s, ] = num_nans
      runtime_tstat[[as.character(c_)]][s] = as.numeric(now()-start_tstat)
      tstat_notes[[as.character(c_)]][s] = tstat_note
      
      

   
      } # End of for (c in c_grid)
    } #End of for (s in 1:S)

  
  # Initialize an empty dictionary to store results for each c
  empty_list_c_D = list() #For each value of c, makes an 1 x D matrix
  for (c_ in c_grid){
    empty_list_c_D[[as.character(c_)]] = matrix(NA, nrow=1, ncol = D)
  }
  
  empty_list_c_0 = list() #For each value of c, makes an 1 x D matrix
  for (c_ in c_grid){
    empty_list_c_0[[as.character(c_)]] = NA
  }
  
  
  #Take averages across the S runs
  width_lm_avg_dict = empty_list_c_D
  width_lm_median_dict = empty_list_c_D
  CI_lm_lb_dict = empty_list_c_D
  CI_lm_ub_dict = empty_list_c_D
  runtime_OLS_avg_dict = empty_list_c_0
  
  width_hulc_avg_dict = empty_list_c_D
  width_hulc_median_dict = empty_list_c_D
  CI_hulc_lb_dict = empty_list_c_D
  CI_hulc_ub_dict = empty_list_c_D
  runtime_hulc_avg_dict = empty_list_c_0
  
  width_tstat_avg_dict = empty_list_c_D
  width_tstat_median_dict = empty_list_c_D
  CI_tstat_lb_dict = empty_list_c_D
  CI_tstat_ub_dict = empty_list_c_D
  runtime_tstat_avg_dict = empty_list_c_0
  

  for (c_ in c_grid){
    
    width_lm_avg_dict[[as.character(c_)]] = apply(width_lm[[as.character(c_)]], mean, MARGIN = 2, na.rm = TRUE)
    width_lm_median_dict[[as.character(c_)]] = apply(width_lm[[as.character(c_)]], median, MARGIN = 2, na.rm = TRUE)
    CI_lm_lb_dict[[as.character(c_)]] = apply(CI_lm_lb[[as.character(c_)]], mean, MARGIN = 2, na.rm = TRUE)
    CI_lm_ub_dict[[as.character(c_)]] = apply(CI_lm_ub[[as.character(c_)]], mean, MARGIN = 2, na.rm = TRUE)
    runtime_OLS_avg_dict[[as.character(c_)]] = mean(runtime_OLS[[as.character(c_)]])
    
    width_hulc_avg_dict[[as.character(c_)]]  = apply(width_hulc[[as.character(c_)]], mean, MARGIN = 2, na.rm = TRUE)
    width_hulc_median_dict[[as.character(c_)]] = apply(width_hulc[[as.character(c_)]], median, MARGIN = 2, na.rm = TRUE)
    CI_hulc_lb_dict[[as.character(c_)]] = apply(CI_hulc_lb[[as.character(c_)]], mean, MARGIN=2, na.rm = TRUE)
    CI_hulc_ub_dict[[as.character(c_)]] = apply(CI_hulc_ub[[as.character(c_)]], mean, MARGIN=2, na.rm = TRUE)
    runtime_hulc_avg_dict[[as.character(c_)]] = mean(runtime_hulc[[as.character(c_)]])
    
    width_tstat_avg_dict[[as.character(c_)]] = apply(width_tstat[[as.character(c_)]], mean, MARGIN = 2, na.rm = TRUE)
    width_tstat_median_dict[[as.character(c_)]] = apply(width_tstat[[as.character(c_)]], median, MARGIN = 2, na.rm = TRUE)
    CI_tstat_lb_dict[[as.character(c_)]] = apply(CI_tstat_lb[[as.character(c_)]], mean, MARGIN=2, na.rm = TRUE)
    CI_tstat_ub_dict[[as.character(c_)]] = apply(CI_tstat_ub[[as.character(c_)]], mean, MARGIN=2, na.rm = TRUE)
    runtime_tstat_avg_dict[[as.character(c_)]] = mean(runtime_tstat[[as.character(c_)]])

    
  }
    
  hulc_nan_flag_dict = empty_list_c_0
  tstat_nan_flag_dict = empty_list_c_0
  OLS_nan_flag_dict = empty_list_c_0
  
  for (c_ in c_grid){

    notes_hulc_temp = unique(hulc_notes[[as.character(c_)]])
    notes_hulc_temp = notes_hulc_temp[notes_hulc_temp!=""]
    if (length(notes_hulc_temp)==0) hulc_nan_flag_dict[[as.character(c_)]] = "" else paste0("HulC NaNs produced in ",  length(notes_hulc_temp), "/", D, " coordinates")
    
    notes_tstat_temp = unique(tstat_notes[[as.character(c_)]])
    notes_tstat_temp = notes_tstat_temp[notes_tstat_temp!=""]
    if (length(notes_tstat_temp)==0) tstat_nan_flag_dict[[as.character(c_)]] = "" else paste0("tstat NaNs produced in ",  length(notes_tstat_temp), "/", D, " coordinates")
    
    notes_wald_temp = unique(Wald_notes[[as.character(c_)]])
    notes_wald_temp = notes_wald_temp[notes_wald_temp!=""]
    if (length(notes_wald_temp)==0) OLS_nan_flag_dict[[as.character(c_)]] = "" else paste0("Wald NaNs produced in CI (probably singular matrix) in ", length(notes_wald_temp), "/", D, " coordinates")
  }

  
  #Calculate coverages
  wald_coverage_rate_dict = empty_list_c_D
  hulc_coverage_rate_dict = empty_list_c_D
  tstat_coverage_rate_dict = empty_list_c_D
  
  wald_coverage_rate_without_nans_dict = empty_list_c_D
  hulc_coverage_rate_without_nans_dict = empty_list_c_D
  tstat_coverage_rate_without_nans_dict = empty_list_c_D
  
  for (c_ in c_grid){
    wald_coverage_rate_dict[[as.character(c_)]] = 100*apply(coverage_lm[[as.character(c_)]], mean, MARGIN=2, na.rm = TRUE)
    hulc_coverage_rate_dict[[as.character(c_)]] = 100*apply(coverage_hulc[[as.character(c_)]], mean, MARGIN=2, na.rm = TRUE)
    tstat_coverage_rate_dict[[as.character(c_)]] = 100*apply(coverage_tstat[[as.character(c_)]], mean, MARGIN=2, na.rm = TRUE)
    
    wald_coverage_rate_without_nans_dict[[as.character(c_)]] = 100*apply(coverage_lm_with_nans[[as.character(c_)]], mean, MARGIN=2, na.rm = TRUE)
    hulc_coverage_rate_without_nans_dict[[as.character(c_)]] = 100*apply(coverage_hulc_with_nans[[as.character(c_)]], mean, MARGIN=2, na.rm = TRUE)
    tstat_coverage_rate_without_nans_dict[[as.character(c_)]] = 100*apply(coverage_tstat_with_nans[[as.character(c_)]], mean, MARGIN=2, na.rm = TRUE)
  }
    
  ################
  # Width ratios
  ################
  
  hulc_tstat_width_ratio_dict = empty_list_1
  hulc_wald_width_ratio_dict = empty_list_1
  tstat_wald_width_ratio_dict = empty_list_1
  wald_tstat_width_ratio_dict = empty_list_1
  
  # Minimum widths
  width_wald_min_dict = empty_list_1
  width_hulc_min_dict = empty_list_1
  width_tstat_min_dict = empty_list_1
  
  # Maximum widths
  width_wald_max_dict = empty_list_1
  width_hulc_max_dict = empty_list_1
  width_tstat_max_dict = empty_list_1
  
  # Mean width ratios
  hulc_tstat_width_ratio_avg_dict = empty_list_c_D
  hulc_wald_width_ratio_avg_dict = empty_list_c_D
  tstat_wald_width_ratio_avg_dict = empty_list_c_D
  wald_tstat_width_ratio_avg_dict = empty_list_c_D
  
  # Median width ratios    
  hulc_tstat_width_ratio_median_dict = empty_list_c_D
  hulc_wald_width_ratio_median_dict = empty_list_c_D
  tstat_wald_width_ratio_median_dict = empty_list_c_D
  wald_tstat_width_ratio_median_dict = empty_list_c_D
  
  #Min width ratios    
  hulc_tstat_width_ratio_min_dict = empty_list_c_D
  hulc_wald_width_ratio_min_dict = empty_list_c_D
  tstat_wald_width_ratio_min_dict = empty_list_c_D
  wald_tstat_width_ratio_min_dict = empty_list_c_D
  
  #Max width ratios    
  hulc_tstat_width_ratio_max_dict = empty_list_c_D
  hulc_wald_width_ratio_max_dict = empty_list_c_D
  tstat_wald_width_ratio_max_dict = empty_list_c_D
  wald_tstat_width_ratio_max_dict = empty_list_c_D
  
  
  for (c_ in c_grid){
    
    #width ratios
    
    hulc_tstat_width_ratio_dict[[as.character(c_)]] = width_hulc[[as.character(c_)]]/width_tstat[[as.character(c_)]]
    hulc_wald_width_ratio_dict[[as.character(c_)]] = width_hulc[[as.character(c_)]]/width_lm[[as.character(c_)]]
    tstat_wald_width_ratio_dict[[as.character(c_)]] = width_tstat[[as.character(c_)]]/width_lm[[as.character(c_)]]
    wald_tstat_width_ratio_dict[[as.character(c_)]] = width_lm[[as.character(c_)]]/width_tstat[[as.character(c_)]]
    
    
    # Minimum widths
    if (sum(is.na(width_lm[[as.character(c_)]]))==S*D){ #Do this to avoid Inf/-Inf values
      width_wald_min_dict[[as.character(c_)]] = rep(NA, D)
    } else {
      width_wald_min_dict[[as.character(c_)]] = apply(width_lm[[as.character(c_)]], min, MARGIN = 2, na.rm = TRUE) 
    }
    
    if (sum(is.na(width_hulc[[as.character(c_)]]))==S*D){  #Do this to avoid Inf/-Inf values
      width_hulc_min_dict[[as.character(c_)]] = rep(NA, D)
      width_tstat_min_dict[[as.character(c_)]] = rep(NA, D)
    } else {
      width_hulc_min_dict[[as.character(c_)]] = apply(width_hulc[[as.character(c_)]], min, MARGIN = 2, na.rm = TRUE)
      width_tstat_min_dict[[as.character(c_)]] = apply(width_tstat[[as.character(c_)]], min, MARGIN = 2, na.rm = TRUE)
    }
    

    # Maximum widths
    if (sum(is.na(width_lm[[as.character(c_)]]))==S*D){ #Do this to avoid Inf/-Inf values
      width_wald_max_dict[[as.character(c_)]] = rep(NA, D)
    } else {
      width_wald_max_dict[[as.character(c_)]] = apply(width_lm[[as.character(c_)]], max, MARGIN = 2, na.rm = TRUE) 
    }
    
    if (sum(is.na(width_hulc[[as.character(c_)]]))==S*D){  #Do this to avoid Inf/-Inf values
      width_hulc_max_dict[[as.character(c_)]] = rep(NA, D)
      width_tstat_max_dict[[as.character(c_)]] = rep(NA, D)
    } else {
      width_hulc_max_dict[[as.character(c_)]] = apply(width_hulc[[as.character(c_)]], max, MARGIN = 2, na.rm = TRUE)
      width_tstat_max_dict[[as.character(c_)]] = apply(width_tstat[[as.character(c_)]], max, MARGIN = 2, na.rm = TRUE)
    }
    
    
    
    # Mean width ratios    
    if (sum(is.na(width_hulc[[as.character(c_)]]))==S*D){  #Do this to avoid Inf/-Inf values
      hulc_tstat_width_ratio_avg_dict[[as.character(c_)]] = rep(NA, D)
      hulc_wald_width_ratio_avg_dict[[as.character(c_)]] = rep(NA, D)
    } else {
      hulc_tstat_width_ratio_avg_dict[[as.character(c_)]] = apply(hulc_tstat_width_ratio_dict[[as.character(c_)]], mean, MARGIN = 2, na.rm = TRUE)
      hulc_wald_width_ratio_avg_dict[[as.character(c_)]] = apply(hulc_wald_width_ratio_dict[[as.character(c_)]], mean, MARGIN = 2, na.rm = TRUE)
    }
    
    if (sum(is.na(width_tstat[[as.character(c_)]]))==S*D | sum(is.na(width_lm[[as.character(c_)]]))==S*D){  #Do this to avoid Inf/-Inf values
      tstat_wald_width_ratio_avg_dict[[as.character(c_)]] = rep(NA, D)
      wald_tstat_width_ratio_avg_dict[[as.character(c_)]] = rep(NA, D)
    } else {
      tstat_wald_width_ratio_avg_dict[[as.character(c_)]] = apply(tstat_wald_width_ratio_dict[[as.character(c_)]], mean, MARGIN = 2, na.rm = TRUE)
      wald_tstat_width_ratio_avg_dict[[as.character(c_)]] = apply(wald_tstat_width_ratio_dict[[as.character(c_)]], mean, MARGIN = 2, na.rm = TRUE)
    }
    
    
    # Median width ratios    
    if (sum(is.na(width_hulc[[as.character(c_)]]))==S*D){  #Do this to avoid Inf/-Inf values
      hulc_tstat_width_ratio_median_dict[[as.character(c_)]] = rep(NA, D)
      hulc_wald_width_ratio_median_dict[[as.character(c_)]] = rep(NA, D)
    } else {
      hulc_tstat_width_ratio_median_dict[[as.character(c_)]] = apply(hulc_tstat_width_ratio_dict[[as.character(c_)]], median, MARGIN = 2, na.rm = TRUE)
      hulc_wald_width_ratio_median_dict[[as.character(c_)]] = apply(hulc_wald_width_ratio_dict[[as.character(c_)]], median, MARGIN = 2, na.rm = TRUE)
    }
    
    if (sum(is.na(width_tstat[[as.character(c_)]]))==S*D | sum(is.na(width_lm[[as.character(c_)]]))==S*D){  #Do this to avoid Inf/-Inf values
      tstat_wald_width_ratio_median_dict[[as.character(c_)]] = rep(NA, D)
      wald_tstat_width_ratio_median_dict[[as.character(c_)]] = rep(NA, D)
    } else {
      tstat_wald_width_ratio_median_dict[[as.character(c_)]] = apply(tstat_wald_width_ratio_dict[[as.character(c_)]], median, MARGIN = 2, na.rm = TRUE)
      wald_tstat_width_ratio_median_dict[[as.character(c_)]] = apply(wald_tstat_width_ratio_dict[[as.character(c_)]], median, MARGIN = 2, na.rm = TRUE)
    }
    

    
    #Min width ratios  
    if (sum(is.na(width_hulc[[as.character(c_)]]))==S*D){  #Do this to avoid Inf/-Inf values
      hulc_tstat_width_ratio_min_dict[[as.character(c_)]] = rep(NA, D)
    } else {
      hulc_tstat_width_ratio_min_dict[[as.character(c_)]] = apply(hulc_tstat_width_ratio_dict[[as.character(c_)]], min, MARGIN = 2, na.rm = TRUE)
    }
    
    if (sum(is.na(width_hulc[[as.character(c_)]]))==S*D | sum(is.na(width_lm[[as.character(c_)]]))==S*D){  #Do this to avoid Inf/-Inf values
      hulc_wald_width_ratio_min_dict[[as.character(c_)]] = rep(NA, D)
    } else {
      hulc_wald_width_ratio_min_dict[[as.character(c_)]] = apply(hulc_wald_width_ratio_dict[[as.character(c_)]], min, MARGIN = 2, na.rm = TRUE)
    }
    
    if (sum(is.na(width_tstat[[as.character(c_)]]))==S*D | sum(is.na(width_lm[[as.character(c_)]]))==S*D){  #Do this to avoid Inf/-Inf values
      tstat_wald_width_ratio_min_dict[[as.character(c_)]] = rep(NA, D)
      wald_tstat_width_ratio_min_dict[[as.character(c_)]] = rep(NA, D)
    } else {
      tstat_wald_width_ratio_min_dict[[as.character(c_)]] = apply(tstat_wald_width_ratio_dict[[as.character(c_)]], min, MARGIN = 2, na.rm = TRUE)
      wald_tstat_width_ratio_min_dict[[as.character(c_)]] = apply(wald_tstat_width_ratio_dict[[as.character(c_)]], min, MARGIN = 2, na.rm = TRUE)
    }
    
  
    
    
    
    
    
    
    #Max width ratios      
    if (sum(is.na(width_hulc[[as.character(c_)]]))==S*D){  #Do this to avoid Inf/-Inf values
      hulc_tstat_width_ratio_max_dict[[as.character(c_)]] = rep(NA, D)
    } else {
      hulc_tstat_width_ratio_max_dict[[as.character(c_)]] = apply(hulc_tstat_width_ratio_dict[[as.character(c_)]], max, MARGIN = 2, na.rm = TRUE)
    }
    
    if (sum(is.na(width_hulc[[as.character(c_)]]))==S*D | sum(is.na(width_lm[[as.character(c_)]]))==S*D){  #Do this to avoid Inf/-Inf values
      hulc_wald_width_ratio_max_dict[[as.character(c_)]] = rep(NA, D)
    } else {
      hulc_wald_width_ratio_max_dict[[as.character(c_)]] = apply(hulc_wald_width_ratio_dict[[as.character(c_)]], max, MARGIN = 2, na.rm = TRUE)
    }
    
    if (sum(is.na(width_tstat[[as.character(c_)]]))==S*D | sum(is.na(width_lm[[as.character(c_)]]))==S*D){  #Do this to avoid Inf/-Inf values
      tstat_wald_width_ratio_max_dict[[as.character(c_)]] = rep(NA, D)
      wald_tstat_width_ratio_max_dict[[as.character(c_)]] = rep(NA, D)
    } else {
      tstat_wald_width_ratio_max_dict[[as.character(c_)]] = apply(tstat_wald_width_ratio_dict[[as.character(c_)]], max, MARGIN = 2, na.rm = TRUE)
      wald_tstat_width_ratio_max_dict[[as.character(c_)]] = apply(wald_tstat_width_ratio_dict[[as.character(c_)]], max, MARGIN = 2, na.rm = TRUE)
    }
    
  }
    

  
  fields = data.frame(date=character(0), model_type=character(0), N=integer(0), S=integer(0), D=integer(0), method=character(0), c=character(0),
                      alpha_lr=numeric(0), epsilon=numeric(0), beta=numeric(0), sigma=numeric(0),
                      burn_in=logical(0),
                      burn_in_threshold=numeric(0), initializer=logical(0),
                      epochs_for_HulC=integer(0), fixed_step=logical(0), cov_type=character(0),
                      CI_type=character(0),
                      theta_k=integer(0),
                      coverage=numeric(0),
                      coverage_excluding_nans=numeric(0),
                      count_of_nans=integer(0),
                      width_avg=numeric(0), width_median=numeric(0), width_min=numeric(0), width_max=numeric(0),
                      CI_type_wald_width_ratio_avg=numeric(0), CI_type_wald_width_ratio_median=numeric(0),
                      CI_type_wald_width_ratio_min=numeric(0), CI_type_wald_width_ratio_max=numeric(0),
                      
                      CI_type_tstat_width_ratio_avg=numeric(0), CI_type_tstat_width_ratio_median=numeric(0),
                      CI_type_tstat_width_ratio_min=numeric(0), CI_type_tstat_width_ratio_max=numeric(0),
                      
                      avg_runtime=numeric(0),
                      notes = character(0))

  CI_types = c("wald", "hulc", "tstat")
  for (c_ in c_grid){
    for (CI_type in CI_types){
      if (CI_type=="wald"){
        
        coverage_rate_dict=wald_coverage_rate_dict
        coverage_rate_without_nans_dict=wald_coverage_rate_without_nans_dict
        num_NaNs_dict = num_NaNs_lm
        width_avg_dict = width_lm_avg_dict
        width_median_dict = width_lm_median_dict
        width_min_dict = width_wald_min_dict
        width_max_dict = width_wald_max_dict
        
        #CI_type_wald_width_ratio_avg_dict = rep(1, D)
        #CI_type_wald_width_ratio_median_dict = rep(1, D)
        #CI_type_wald_width_ratio_min_dict = rep(1, D)
        #CI_type_wald_width_ratio_max_dict = rep(1, D)
        
        CI_type_tstat_width_ratio_avg_dict = wald_tstat_width_ratio_avg_dict
        CI_type_tstat_width_ratio_median_dict = wald_tstat_width_ratio_median_dict
        CI_type_tstat_width_ratio_min_dict = wald_tstat_width_ratio_min_dict
        CI_type_tstat_width_ratio_max_dict = wald_tstat_width_ratio_max_dict
        
        runtime_dict=runtime_OLS_avg_dict
        notes_dict=OLS_nan_flag_dict
        
      }else if (CI_type=="hulc"){
        
        coverage_rate_dict=hulc_coverage_rate_dict
        coverage_rate_without_nans_dict=hulc_coverage_rate_without_nans_dict
        num_NaNs_dict = num_NaNs_hulc
        width_avg_dict = width_hulc_avg_dict
        width_median_dict = width_hulc_median_dict
        width_min_dict = width_hulc_min_dict
        width_max_dict = width_hulc_max_dict
        
        CI_type_wald_width_ratio_avg_dict = hulc_wald_width_ratio_avg_dict
        CI_type_wald_width_ratio_median_dict = hulc_wald_width_ratio_median_dict
        CI_type_wald_width_ratio_min_dict = hulc_wald_width_ratio_min_dict
        CI_type_wald_width_ratio_max_dict = hulc_wald_width_ratio_max_dict
        
        CI_type_tstat_width_ratio_avg_dict = hulc_tstat_width_ratio_avg_dict
        CI_type_tstat_width_ratio_median_dict = hulc_tstat_width_ratio_median_dict
        CI_type_tstat_width_ratio_min_dict = hulc_tstat_width_ratio_min_dict
        CI_type_tstat_width_ratio_max_dict = hulc_tstat_width_ratio_max_dict
        
        runtime_dict=runtime_hulc_avg_dict
        notes_dict=hulc_nan_flag_dict
        
      }else if (CI_type=="tstat"){
        coverage_rate_dict=tstat_coverage_rate_dict
        coverage_rate_without_nans_dict=tstat_coverage_rate_without_nans_dict
        num_NaNs_dict = num_NaNs_tstat
        width_avg_dict = width_tstat_avg_dict
        width_median_dict = width_tstat_median_dict
        width_min_dict = width_tstat_min_dict
        width_max_dict = width_tstat_max_dict
        
        CI_type_wald_width_ratio_avg_dict = tstat_wald_width_ratio_avg_dict
        CI_type_wald_width_ratio_median_dict = tstat_wald_width_ratio_median_dict
        CI_type_wald_width_ratio_min_dict = tstat_wald_width_ratio_min_dict
        CI_type_wald_width_ratio_max_dict = tstat_wald_width_ratio_max_dict
        
        #CI_type_tstat_width_ratio_avg = rep(1, D)
        #CI_type_tstat_width_ratio_median = rep(1, D)
        #CI_type_tstat_width_ratio_min = rep(1, D)
        #CI_type_tstat_width_ratio_max = rep(1, D)
        
        runtime_dict=runtime_tstat_avg_dict
        notes_dict=tstat_nan_flag_dict
      }
      
      
      fields_new = data.frame(date=rep(date, D), model_type=rep(model_type, D), N=rep(N, D), S=rep(S, D), D=rep(D, D), method=rep(method, D), c=rep(c_, D),
                          alpha_lr=rep(alpha_lr, D), epsilon=rep(epsilon, D), beta=rep(beta, D), sigma=rep(sigma, D),
                          burn_in=rep(burn_in, D),
                          burn_in_threshold=rep(burn_in_threshold, D), initializer=rep(initializer, D),
                          epochs_for_HulC=rep(epochs_for_HulC, D), fixed_step=rep(fixed_step, D), cov_type=rep(cov_type, D),
                          CI_type=rep(CI_type, D),
                          theta_k=1:D,
                          coverage= coverage_rate_dict[[as.character(c_)]],            
                          coverage_excluding_nans = coverage_rate_without_nans_dict[[as.character(c_)]],
                          count_of_nans=apply(num_NaNs_dict[[as.character(c_)]], sum, MARGIN=2),
                          
                          width_avg=width_avg_dict[[as.character(c_)]],
                          width_median=width_median_dict[[as.character(c_)]],
                          width_min=width_min_dict[[as.character(c_)]],
                          width_max=width_max_dict[[as.character(c_)]],
                          
                          CI_type_wald_width_ratio_avg=if(CI_type=="wald") rep(1, D) else CI_type_wald_width_ratio_avg_dict[[as.character(c_)]],
                          CI_type_wald_width_ratio_median=if(CI_type=="wald") rep(1, D) else CI_type_wald_width_ratio_median_dict[[as.character(c_)]],
                          CI_type_wald_width_ratio_min=if(CI_type=="wald") rep(1, D) else CI_type_wald_width_ratio_min_dict[[as.character(c_)]],
                          CI_type_wald_width_ratio_max=if(CI_type=="wald") rep(1, D) else CI_type_wald_width_ratio_max_dict[[as.character(c_)]],
                          
                          
                          CI_type_tstat_width_ratio_avg= if(CI_type=="tstat") rep(1, D) else CI_type_tstat_width_ratio_avg_dict[[as.character(c_)]],
                          CI_type_tstat_width_ratio_median= if(CI_type=="tstat") rep(1, D) else CI_type_tstat_width_ratio_median_dict[[as.character(c_)]],
                          CI_type_tstat_width_ratio_min= if(CI_type=="tstat") rep(1, D) else CI_type_tstat_width_ratio_min_dict[[as.character(c_)]],
                          CI_type_tstat_width_ratio_max= if(CI_type=="tstat") rep(1, D) else CI_type_tstat_width_ratio_max_dict[[as.character(c_)]],
                          
                          avg_runtime=rep(runtime_dict[[as.character(c_)]],D),
                          notes = rep(notes_dict[[as.character(c_)]],D) )
      
      fields = rbind(fields, fields_new)
      
    } #end of for (CI_type in CI_types) for fields dataframe
  }  #end of for (c_ in c_grid) for fields dataframe
  
  
  write.table(fields, csv_name, sep = ",", col.names = !file.exists(csv_name), append = T,
              na="", row.names=F)
    
}
    


run_this=F
if (run_this){
  run_sims(model_type = "OLS", S=3, N=10**3, D=100,
           method = "noisy-truncated-sgd", cov_type = "Toeplitz", alpha_level = .05,
           c_grid = c(0.01, 0.05, 0.1,  0.2,  0.5, 0.75, 1, 1.5, 2),
           #c_grid = c(2),
           alpha_lr = .505,
           epsilon = .8, sigma=1, beta=.25,
           initializer = T, epochs_for_HulC = 1,
           fixed_step = F, verbose=T, csv_name = "extra_sims.csv",
           use_all_wald_estimates=T)
  
}




