
#Adjust the working directory.
your_wd = ""
your_wd_plots = ""

packages_to_check <- c("tidyverse", "sgd", "latex2exp")
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



#Functions for HulC


sigmoid <- function(x) {
  return (1 / (1 + exp(-x)))
}


#Functions related to batch size
P_B <- function(B,D){
  #Returns the upper bound on the miscoverage probablility (equation 3 from HulC paper 9/23)
  #B = number of batches
  #D = delta; the median bias
  out = (0.5 - D)^B + (0.5 + D)^B
  return (out)
}


min_B <- function(alpha,D){
  #Returns the smallest integer B>=1 such that P_B(B, D) <= alpha, according to Algorithm 1 (HulC paper 9/23)
  B = 1
  while (TRUE) {
    p <- P_B(B, D)
    if (p <= alpha) {
      break  # Exit the loop when the condition is met
    }
    B <- B + 1
  }
  return(B)
}


B_star <- function(alpha,D){
  #Finds the batch size "B*" according to equation 4 (HulC paper 9/23)
  B = min_B(alpha, D)
  U = runif(1,0,1)
  numer = alpha - P_B(B, D)
  denom = P_B(B-1, D) - P_B(B,D)
  tau = numer/denom
  if (U <= tau){
    return(B-1)
  }
  else{
    return(B) 
  }
}




## Finding randomized B. This is the actually number of splits used for constructing the convex hull.
## For the purposes of SGD inference, we only need Delta = 0 and t = 0.
## If randomize = FALSE, then the method always uses 6 splits with Delta = 0, t = 0. 
## This can have coverage between 1 - alpha and 1 - alpha/2.
## The advantage would be that the finite sample median bias can be as large as 0.1.
#Source: https://github.com/Arun-Kuchibhotla/HulC/blob/main/R/HulC%20Inference%20for%20Online%20Algorithms.ipynb
find_randomize_B <- function(alpha, Delta = 0.0, t = 0.0, randomize = TRUE){
  if(Delta == 0.5 && t == 0){
    stop("Delta is 0.5 and t = 0. The estimator lies only on one side of the parameter!")
  }
  B_low <- max(ceiling(log((2 + 2*t)/alpha, base = 2 + 2*t)), ceiling(log((1 + t)/alpha, base = (2 + 2*t)/(1 + 2*Delta))))
  B_up <- ceiling(log((2 + 2*t)/alpha, base = (2 + 2*t)/(1 + 2*Delta)))
  Q <- function(B){
    ((1/2 - Delta)^B + (1/2 + Delta)^B)*(1 + t)^(-B + 1)
  }
  for(B in B_low:B_up){
    if(Q(B) <= alpha)
      break
  }
  B1 <- B
  if(randomize){
    B0 <- B1 - 1
    p1 <- Q(B1)
    p0 <- Q(B0)
    U <- runif(1)
    tau <- (alpha - Q(B1))/(Q(B0) - Q(B1))
    B <- B0*(U <= tau) + B1*(U > tau)
  }
  return(B)
}





## SGD with splitting.
## Except for the first two arguments, all the arguments of sgd_HulC are same as those for sgd function from library("sgd")
sgd_HulC <- function(alpha = 0.05, randomize = TRUE, formula, data, model, model.control = list(), sgd.control = list(...), ...){
  B <- find_randomize_B(alpha = alpha, randomize = randomize)
  nn <- nrow(data)
  batch_estimates <- NULL
  sgd_out <- vector("list", B) 
  for(idx in 1:B){
    data_idx <- data[seq(idx, nn, by = B),,drop = FALSE]
    sgd_out[[idx]] <- sgd::sgd(formula=formula, data=data_idx, model=model, model.control = model.control, sgd.control = sgd.control)
    batch_estimates <- rbind(batch_estimates, as.vector(sgd_out[[idx]]$coefficients))
  }
  ci_lwr <- apply(batch_estimates, 2, min)
  ci_upr <- apply(batch_estimates, 2, max)
  return(list(ci = cbind(ci_lwr, ci_upr), B = B, sgd_out = sgd_out, batch_estimates=batch_estimates))
}


#Loss functions

OLS_gradient = function(theta, X, Y){
  if (is.matrix(X)){
    return(as.numeric(t(X)%*%(X%*%theta-Y)))
  } else{
    return(as.numeric(X*as.numeric(X%*%theta-Y)))
  }
}

logistic_gradient = function(theta, X, Y){
  prod = X%*%theta
  if (is.matrix(X)){
    return(as.numeric(t(X)%*%(sigmoid(prod) - Y)))
  } else{
    return(as.numeric(X*as.numeric(sigmoid(prod) - Y)))
  }
}


root_sgd = function(data, grad_f, c_ = .01,  alpha_lr = .505, 
                    initial_theta = 0, plot_it = F, plot_theta_dim = 1,
                    true_theta = seq(0,1, length.out=dim(data)[2]),
                    verbose = F, epochs = 1, fixed_step = F,
                    return_trajectory_only=F){
  
  # Performs Root SGD  (Li et al 2021: https://arxiv.org/pdf/2008.12690)
  # This function outputs a list of 2 components:
  #          theta_root_SGD: the final iterate of theta using root SGD.
  #          theta_vanilla_SGD: the final iterate of theta using vanilla SGD and same step size hyperparameters.
  # data: A matrix or a dataframe [Y, X] where X consists of d numeric columns and Y is an array of real numbers. If the intercept (i.e., a column of 1's) is not included in the X columns, then this will be created.
  # grad_f: gradient function that should take 3 arguments: theta (numeric), X (matrix or numeric), Y (matrix or numeric) and outputs a numeric vector of same length as theta.
  # c_: The "c_" parameter in the eta_t() function used to calculate the step size: eta_t(c_, alpha_lr) = c_*t^(-alpha_lr)
  # alpha_lr: The "alpha" parameter in the eta_t() function used to calculate the step size: eta_t(c_, alpha_lr) = c_*t^(-alpha_lr). Note this value should be between 0.5 and 1, non-inclusive.
  # initial_theta: Starting value of theta. Must be either 1 dimension (in which case full vector is created, ex. a value of .5  results in a vector of all .5's) or a vector of same dimension as X input. Default is 0. 
  # plot: If TRUE, then a plot is produced that track progress of the theta estimate
  #       with respect to time steps t.
  # plot_theta_dim: The coordinate of theta to plot, ex., theta_to_plot=1 plots the intercept.
  # true_theta: an optional argument for plotting; default is to set theta between 0 and 1, equally spaced intervals.
  # epochs: Should be an integer. The number of epochs to run the algorithm.
  #         Default is 1.
  # fixed_step: If TRUE, means that the root-SGD algorithm is performed with constant step size = c_.
  #            If FALSE, then eta_t() function is used: : eta_t(c_, alpha_lr) = c_*t^(-alpha_lr)
  
  assertthat::assert_that(.5<alpha_lr & alpha_lr<1, msg="alpha_lr must be greater than 0.5 and less than 1.")
  assertthat::assert_that(c_>0, msg="c_ must be greater than 0.")
  
  
  if (is.data.frame(data)){
    Y = as.matrix(data[1])
    X = as.matrix(data[-1])
  }else if (is.matrix(data)){
    Y = as.matrix(data[,1])
    X = as.matrix(data[,-1])
  }
  
  n = dim(X)[1]
  
  #X needs first column to be 1's (for the intercept)
  if (sum(X[,1])!=n){
    X=cbind(rep(1,n), X)
    colnames(X)[1] = "X1"
  }
  
  d = dim(X)[2]
  
  if (length(initial_theta)==0){
    theta = rep(initial_theta, d) 
  } else {
    assertthat::assert_that(length(initial_theta)==d, msg = "Starting value initial_theta must be 1-dimensional or same dimension as data.")
    theta = initial_theta
  }
  
  
  #Starting values
  theta_t2 = rnorm(d) #two time steps back
  v = grad_f(theta, X[1,], Y[1])
  theta_t1 = theta
  theta = theta_t1-c_*v
  theta_SGD = theta
  
  
  if (plot_it | return_trajectory_only){
    theta_t = matrix(theta, ncol=d)
    v_t = matrix(v, ncol=d)
    theta_SGD_t = matrix(theta, ncol=d)
  }
  
  
  
  if (fixed_step){
    step = c_
  }
  
  T_ = 1 #Tracker that also uses epochs
  for (epoch in 1:epochs){
    #Shuffle data randomly at each epoch (to avoid cycles)
    indices_shuffled = sample(n)
    X = X[indices_shuffled,]
    Y = Y[indices_shuffled]
    
    
    for (tt in 2:n){
      T_ = T_ + 1
      
      #Define step size
      if (!fixed_step){
        step = c_*T_^(-alpha_lr)
      }
      
      if (plot_it | return_trajectory_only){
        theta_t = rbind(theta_t, theta)
        v_t = rbind(v_t, v)
        theta_SGD_t = rbind(theta_SGD_t, theta_SGD)
      }
      
      
      #Root-SGD updates
      v_new = grad_f(theta_t1, X[tt,], Y[tt]) + ((tt-1)/tt)*(v - grad_f(theta_t2, X[tt,], Y[tt]))
      theta = theta_t1 - step*v_new
      
      #Compare to vanilla SGD
      theta_SGD_new = theta_SGD - step*grad_f(theta_SGD, X[tt,], Y[tt])
      
      
      #Update variables
      v=v_new
      theta_t2 = theta_t1
      theta_t1 = theta
      theta_SGD = theta_SGD_new
      
    } #End of for (tt in 2:n)
    
    
  } #End of for (epoch in 1:epochs)
  
  
  if (plot_it){
    plot(x=1:T_, y= theta_t[,plot_theta_dim], type = "l",
         xlab = "time step t", ylab = expression(hat(theta)[k]), main = expression(hat(theta)[k]), col="blue",
         ylim=c(min(c(theta_t[,plot_theta_dim], theta_SGD_t[,plot_theta_dim], true_theta[plot_theta_dim])),
                max(c(theta_t[,plot_theta_dim], theta_SGD_t[,plot_theta_dim], true_theta[plot_theta_dim]))))
    lines(x=1:T_, y = theta_SGD_t[,plot_theta_dim], col = "gray")
    abline(h=true_theta[plot_theta_dim], col = "red")
    legend("bottomright", legend=c("Root-SGD", "Vanilla-SGD", "target"),
           col=c("blue", "gray", "red"), lty=1:1, cex=0.8)
    
  }
  if (return_trajectory_only){
    return(list(theta_t=theta_t, theta_SGD_t=theta_SGD_t))
  }
  
  out = list(theta_root_SGD = theta, theta_vanilla_SGD = theta_SGD)  
  return(out)
}



root_sgd_HulC <- function(data, grad_f, alpha = 0.05, randomize = TRUE, c_ = .01,  alpha_lr = .505, 
                          initial_theta = 0, plot_it = F, plot_theta_dim = 1,
                          true_theta = seq(0,1, length.out=dim(data)[2]),
                          verbose = F, epochs = 1, fixed_step = F){
  B <- find_randomize_B(alpha = alpha, randomize = randomize)
  nn <- nrow(data)
  batch_estimates <- NULL
  sgd_out <- vector("list", B) 
  
  for(idx in 1:B){
    data_idx <- data[seq(idx, nn, by = B),,drop = FALSE]
    out <- root_sgd(data_idx, grad_f, c_,  alpha_lr, 
                               initial_theta, plot_it, plot_theta_dim,
                               true_theta,
                               verbose, epochs, fixed_step)
    sgd_out[[idx]] = out$theta_root_SGD
    
    batch_estimates <- rbind(batch_estimates, as.vector(sgd_out[[idx]]))
  }
  ci_lwr <- apply(batch_estimates, 2, min)
  ci_upr <- apply(batch_estimates, 2, max)
  return(list(ci = cbind(ci_lwr, ci_upr), B = B, batch_estimates=batch_estimates))
}

gradient_truncation = function(gradient, epsilon){
  # See Alg 2 in Zhao et al 2021 (https://arxiv.org/abs/2103.00075) for reference
  #Input:
  #   gradient = numeric vector
  #   epsilon = value in interval (0,1)
  
  grad2 = gradient^2
  grad2_sorted = sort(grad2, decreasing=T)
  grad_sorted = gradient[order(grad2, decreasing = T)]
  grad2_cumsum = cumsum(grad2_sorted)
  threshold = (1-epsilon^2)*sum(grad2)
  
  out=abs(grad_sorted[grad2_cumsum>=threshold][1])
  return(out)
}



truncated_sgd = function(data, grad_f, epsilon=.8, 
                         c_ = .01,  alpha_lr = .505, 
                         initial_theta = 0, plot_it = F, plot_theta_dim = 1,
                         true_theta = seq(0,1, length.out=dim(data)[2]),
                         verbose = F, epochs = 1, fixed_step = F,
                         return_trajectory_only=F){
  
  # Performs Truncated SGD  (Zhao et al 2021: https://arxiv.org/abs/2103.00075)
  # This function outputs a list of 2 components:
  #          theta_trunc_SGD: the final iterate of theta using root SGD.
  #          theta_vanilla_SGD: the final iterate of theta using vanilla SGD and same step size hyperparameters.
  # data: A matrix or a dataframe [Y, X] where X consists of d numeric columns and Y is an array of real numbers. If the intercept (i.e., a column of 1's) is not included in the X columns, then this will be created.
  # grad_f: gradient function that should take 3 arguments: theta (numeric), X (matrix or numeric), Y (matrix or numeric) and outputs a numeric vector of same length as theta.
  # epsilon: value in interval (0,1)
  # sigma: noise parameter >0
  # c_: The "c_" parameter in the eta_t() function used to calculate the step size: eta_t(c_, alpha_lr) = c_*t^(-alpha_lr)
  # alpha_lr: The "alpha" parameter in the eta_t() function used to calculate the step size: eta_t(c_, alpha_lr) = c_*t^(-alpha_lr). Note this value should be between 0.5 and 1, non-inclusive.
  # initial_theta: Starting value of theta. Must be either 1 dimension (in which case full vector is created, ex. a value of .5  results in a vector of all .5's) or a vector of same dimension as X input. Default is 0. 
  # plot: If TRUE, then a plot is produced that track progress of the theta estimate
  #       with respect to time steps t.
  # plot_theta_dim: The coordinate of theta to plot, ex., theta_to_plot=1 plots the intercept.
  # true_theta: an optional argument for plotting; default is to set theta between 0 and 1, equally spaced intervals.
  # epochs: Should be an integer. The number of epochs to run the algorithm.
  #         Default is 1.
  # fixed_step: If TRUE, means that the root-SGD algorithm is performed with constant step size = c_.
  #            If FALSE, then eta_t() function is used: : eta_t(c_, alpha_lr) = c_*t^(-alpha_lr)
  
  assertthat::assert_that(.5<alpha_lr & alpha_lr<1, msg="alpha_lr must be greater than 0.5 and less than 1.")
  assertthat::assert_that(c_>0, msg="c_ must be greater than 0.")
  assertthat::assert_that(0<epsilon & epsilon<1, msg="epsilon must be between 0 and 1")
  
  if (is.data.frame(data)){
    Y = as.matrix(data[1])
    X = as.matrix(data[-1])
  }else if (is.matrix(data)){
    Y = as.matrix(data[,1])
    X = as.matrix(data[,-1])
  }
  
  
  n = dim(X)[1]
  
  #X needs first column to be 1's (for the intercept)
  if (sum(X[,1])!=n){
    X=cbind(rep(1,n), X)
    colnames(X)[1] = "X1"
  }
  
  d = dim(X)[2]
  
  if (length(initial_theta)==0){
    theta = rep(initial_theta, d) 
  } else {
    assertthat::assert_that(length(initial_theta)==d, msg = "Starting value initial_theta must be 1-dimensional or same dimension as data.")
    theta = initial_theta
  }
  
  
  #Starting values
  theta_t1 = theta
  theta_SGD = theta
  
  
  if (plot_it | return_trajectory_only){
    theta_t = matrix(theta, ncol=d)
    theta_SGD_t = matrix(theta, ncol=d)
  }
  
  
  
  if (fixed_step){
    step = c_
  }
  
  T_ = 1 #Tracker that also uses epochs
  for (epoch in 1:epochs){
    #Shuffle data randomly at each epoch (to avoid cycles)
    indices_shuffled = sample(n)
    X = X[indices_shuffled,]
    Y = Y[indices_shuffled]
    
    
    for (tt in 2:n){
      T_ = T_ + 1
      
      #Define step size
      if (!fixed_step){
        step = c_*T_^(-alpha_lr)
      }
      
      if (plot_it | return_trajectory_only){
        theta_t = rbind(theta_t, theta)
        theta_SGD_t = rbind(theta_SGD_t, theta_SGD)
      }
      
      #Calculate gradient
      grad_tt = grad_f(theta_t1, X[tt,], Y[tt])
      #Calculate cut threshold
      cut_threshold = gradient_truncation(grad_tt, epsilon)
      
      #Truncation
      grad_tilde_tt = grad_tt
      grad_tilde_tt[abs(grad_tt)<cut_threshold] = 0
      
      #Truncated-SGD update
      theta = theta_t1 - step*grad_tilde_tt
      
      #Compare to vanilla SGD
      theta_SGD_new = theta_SGD - step*grad_f(theta_SGD, X[tt,], Y[tt])
      
      
      #Update variables
      theta_t1 = theta
      theta_SGD = theta_SGD_new
      
    } #End of for (tt in 2:n)
    
    
  } #End of for (epoch in 1:epochs)
  
  
  if (plot_it){
    plot(x=1:T_, y= theta_t[,plot_theta_dim], type = "l",
         xlab = "time step t", ylab = expression(hat(theta)[k]), main = expression(hat(theta)[k]), col="purple",
         ylim=c(min(c(theta_t[,plot_theta_dim], theta_SGD_t[,plot_theta_dim], true_theta[plot_theta_dim])),
                max(c(theta_t[,plot_theta_dim], theta_SGD_t[,plot_theta_dim], true_theta[plot_theta_dim]))))
    lines(x=1:T_, y = theta_SGD_t[,plot_theta_dim], col = "gray")
    abline(h=true_theta[plot_theta_dim], col = "red")
    legend("bottomright", legend=c("Truncated-SGD", "Vanilla-SGD", "target"),
           col=c("purple", "gray", "red"), lty=1:1, cex=0.8)
  }
  
  if (return_trajectory_only){
    return(list(theta_t=theta_t, theta_SGD_t=theta_SGD_t))
  }
  
  out = list(theta_truncated_SGD = theta, theta_vanilla_SGD = theta_SGD)  
  return(out)
}



truncated_sgd_HulC <- function(data, grad_f, alpha = 0.05, randomize = TRUE,
                               epsilon=.8, 
                               c_ = .01,  alpha_lr = .505, 
                          initial_theta = 0, plot_it = F, plot_theta_dim = 1,
                          true_theta = seq(0,1, length.out=dim(data)[2]),
                          verbose = F, epochs = 1, fixed_step = F){
  B <- find_randomize_B(alpha = alpha, randomize = randomize)
  nn <- nrow(data)
  batch_estimates <- NULL
  sgd_out <- vector("list", B) 
  
  for(idx in 1:B){
    data_idx <- data[seq(idx, nn, by = B),,drop = FALSE]
    out <- truncated_sgd(data_idx, grad_f,epsilon,
                              c_,  alpha_lr,
                              initial_theta, plot_it, plot_theta_dim,
                              true_theta,
                              verbose, epochs, fixed_step)
    sgd_out[[idx]] = out$theta_truncated_SGD
    
    batch_estimates <- rbind(batch_estimates, as.vector(sgd_out[[idx]]))
  }
  ci_lwr <- apply(batch_estimates, 2, min)
  ci_upr <- apply(batch_estimates, 2, max)
  return(list(ci = cbind(ci_lwr, ci_upr), B = B, batch_estimates=batch_estimates))
}


noisy_truncated_sgd = function(data, grad_f, epsilon=.8, sigma=1, beta=0.25,
                               c_ = .01,  alpha_lr = .505, 
                               initial_theta = 0, plot_it = F, plot_theta_dim = 1,
                               true_theta = seq(0,1, length.out=dim(data)[2]),
                               verbose = F, epochs = 1, fixed_step = F,
                               return_trajectory_only=F){

  # Performs Noisy Truncated SGD  (Zhao et al 2021: https://arxiv.org/abs/2103.00075)
  # This function outputs a list of 2 components:
  #          theta_ntrunc_SGD: the final iterate of theta using root SGD.
  #          theta_vanilla_SGD: the final iterate of theta using vanilla SGD and same step size hyperparameters.
  # data: A matrix or a dataframe [Y, X] where X consists of d numeric columns and Y is an array of real numbers. If the intercept (i.e., a column of 1's) is not included in the X columns, then this will be created.
  # grad_f: gradient function that should take 3 arguments: theta (numeric), X (matrix or numeric), Y (matrix or numeric) and outputs a numeric vector of same length as theta.
  # epsilon: value in interval (0,1)
  # sigma: noise parameter >0
  # c_: The "c_" parameter in the eta_t() function used to calculate the step size: eta_t(c_, alpha_lr) = c_*t^(-alpha_lr)
  # alpha_lr: The "alpha" parameter in the eta_t() function used to calculate the step size: eta_t(c_, alpha_lr) = c_*t^(-alpha_lr). Note this value should be between 0.5 and 1, non-inclusive.
  # initial_theta: Starting value of theta. Must be either 1 dimension (in which case full vector is created, ex. a value of .5  results in a vector of all .5's) or a vector of same dimension as X input. Default is 0. 
  # plot: If TRUE, then a plot is produced that track progress of the theta estimate
  #       with respect to time steps t.
  # plot_theta_dim: The coordinate of theta to plot, ex., theta_to_plot=1 plots the intercept.
  # true_theta: an optional argument for plotting; default is to set theta between 0 and 1, equally spaced intervals.
  # epochs: Should be an integer. The number of epochs to run the algorithm.
  #         Default is 1.
  # fixed_step: If TRUE, means that the root-SGD algorithm is performed with constant step size = c_.
  #            If FALSE, then eta_t() function is used: : eta_t(c_, alpha_lr) = c_*t^(-alpha_lr)
  
  assertthat::assert_that(.5<alpha_lr & alpha_lr<1, msg="alpha_lr must be greater than 0.5 and less than 1.")
  assertthat::assert_that(c_>0, msg="c_ must be greater than 0.")
  assertthat::assert_that(0<epsilon & epsilon<1, msg="epsilon must be between 0 and 1")
  assertthat::assert_that(sigma>0, msg="sigma must be greater than 0.")
  assertthat::assert_that(0<beta & beta<0.5, msg="beta must be greater between 0 and .5")
  
  
  if (is.data.frame(data)){
    Y = as.matrix(data[1])
    X = as.matrix(data[-1])
  }else if (is.matrix(data)){
    Y = as.matrix(data[,1])
    X = as.matrix(data[,-1])
  }
  
  
  n = dim(X)[1]
  
  #X needs first column to be 1's (for the intercept)
  if (sum(X[,1])!=n){
    X=cbind(rep(1,n), X)
    colnames(X)[1] = "X1"
  }
  
  d = dim(X)[2]
  
  if (length(initial_theta)==0){
    theta = rep(initial_theta, d) 
  } else {
    assertthat::assert_that(length(initial_theta)==d, msg = "Starting value initial_theta must be 1-dimensional or same dimension as data.")
    theta = initial_theta
  }
  
  
  #Starting values
  theta_t1 = theta
  theta_SGD = theta
  
  
  if (plot_it | return_trajectory_only){
    theta_t = matrix(theta, ncol=d)
    theta_SGD_t = matrix(theta, ncol=d)
  }
  
  
  
  if (fixed_step){
    step = c_
  }
  
  T_ = 1 #Tracker that also uses epochs
  for (epoch in 1:epochs){
    #Shuffle data randomly at each epoch (to avoid cycles)
    indices_shuffled = sample(n)
    X = X[indices_shuffled,]
    Y = Y[indices_shuffled]
    
    
    for (tt in 2:n){
      T_ = T_ + 1
      
      #Define step size
      if (!fixed_step){
        step = c_*T_^(-alpha_lr)
      }
      
      if (plot_it | return_trajectory_only){
        theta_t = rbind(theta_t, theta)
        theta_SGD_t = rbind(theta_SGD_t, theta_SGD)
      }
      
      #Calculate gradient
      grad_tt = grad_f(theta_t1, X[tt,], Y[tt])
      #Calculate cut threshold
      cut_threshold = gradient_truncation(grad_tt, epsilon)
      
      #Truncation
      grad_tilde_tt = grad_tt
      grad_tilde_tt[abs(grad_tt)<cut_threshold] = 0
      
      #Set noisy term
      b_tt = rnorm(n=d, mean=0, sd=sigma)
      noisy_step = step^(.5+beta)
      
      #Noisy-truncated-SGD update
      theta = theta_t1 - step*grad_tilde_tt + noisy_step*b_tt
      
      #Compare to vanilla SGD
      theta_SGD_new = theta_SGD - step*grad_f(theta_SGD, X[tt,], Y[tt])
      
      
      #Update variables
      theta_t1 = theta
      theta_SGD = theta_SGD_new
      
    } #End of for (tt in 2:n)
    
    
  } #End of for (epoch in 1:epochs)
  
  
  if (plot_it){
    plot(x=1:T_, y= theta_t[,plot_theta_dim], type = "l",
         xlab = "time step t", ylab = expression(hat(theta)[k]), main = expression(hat(theta)[k]), col="purple",
         ylim=c(min(c(theta_t[,plot_theta_dim], theta_SGD_t[,plot_theta_dim], true_theta[plot_theta_dim])),
                max(c(theta_t[,plot_theta_dim], theta_SGD_t[,plot_theta_dim], true_theta[plot_theta_dim]))))
    lines(x=1:T_, y = theta_SGD_t[,plot_theta_dim], col = "gray")
    abline(h=true_theta[plot_theta_dim], col = "red")
    legend("bottomright", legend=c("Noisy-truncated-SGD", "Vanilla-SGD", "target"),
           col=c("purple", "gray", "red"), lty=1:1, cex=0.8)
    
  }
  if (return_trajectory_only){
    return(list(theta_t=theta_t, theta_SGD_t=theta_SGD_t))
  }
  
  out = list(theta_ntrunc_SGD = theta, theta_vanilla_SGD = theta_SGD)  
  return(out)
}





noisy_truncated_sgd_HulC <- function(data, grad_f, alpha = 0.05, randomize = TRUE,
                               epsilon=.8, sigma=1, beta=.25,
                               c_ = .01,  alpha_lr = .505, 
                               initial_theta = 0, plot_it = F, plot_theta_dim = 1,
                               true_theta = seq(0,1, length.out=dim(data)[2]),
                               verbose = F, epochs = 1, fixed_step = F){
  B <- find_randomize_B(alpha = alpha, randomize = randomize)
  nn <- nrow(data)
  batch_estimates <- NULL
  sgd_out <- vector("list", B) 
  
  for(idx in 1:B){
    data_idx <- data[seq(idx, nn, by = B),,drop = FALSE]
    out <- noisy_truncated_sgd(data_idx, grad_f, epsilon, sigma, beta,
                         c_,  alpha_lr,
                         initial_theta, plot_it, plot_theta_dim,
                         true_theta,
                         verbose, epochs, fixed_step)
    sgd_out[[idx]] = out$theta_ntrunc_SGD
    
    batch_estimates <- rbind(batch_estimates, as.vector(sgd_out[[idx]]))
  }
  ci_lwr <- apply(batch_estimates, 2, min)
  ci_upr <- apply(batch_estimates, 2, max)
  return(list(ci = cbind(ci_lwr, ci_upr), B = B, batch_estimates=batch_estimates))
}




compare_trajectories = F
if (compare_trajectories){
  setwd(your_wd)
  
  #Toggle these settings (only use 1 TRUE value at a time to run the SGD functions) to achieve plots p1, p2, p3, p4, p5:
  paper_settings_convergence1 = T
  paper_settings_convergence2 = F
  paper_settings_big_c = F
  paper_settings_small_c = F
  paper_settings_med_c = F
  if (paper_settings_convergence1){
    source("gen_data.R")
    N = 10**3
    D=5
    cov_type= "Toeplitz"
    seed=1
    set.seed(seed)
    XY = gen_normal0_data(n=N, d=D, cov_type = cov_type)
    XY_no_intercept = XY[names(XY)!="X1"]
    c_=.75
    k_to_plot=5
    true_theta = seq(0,1, length.out=D)
    true_theta = true_theta[k_to_plot]
    init_theta = rnorm(D, mean=0, sd=1e-5)
  } else if(paper_settings_convergence2){
    N = 10**3
    D=5
    cov_type= "Toeplitz"
    seed=4
    set.seed(seed)
    XY = gen_normal0_data(n=N, d=D, cov_type = cov_type)
    XY_no_intercept = XY[names(XY)!="X1"]
    c_=.75
    k_to_plot=5
    true_theta = seq(0,1, length.out=D)
    true_theta = true_theta[k_to_plot]
    init_theta = rnorm(D, mean=0, sd=1e-5)
  }else if(paper_settings_big_c){
    N = 10**4
    D=20
    cov_type= "Toeplitz"
    seed=1
    set.seed(seed)
    XY = gen_normal0_data(n=N, d=D, cov_type = cov_type)
    XY_no_intercept = XY[names(XY)!="X1"]
    c_=1.5
    k_to_plot=20
    true_theta = seq(0,1, length.out=D)
    true_theta = true_theta[k_to_plot]
    init_theta = rnorm(D, mean=0, sd=1e-5)
  }else if(paper_settings_small_c){
    N = 10**4
    D=20
    cov_type= "Toeplitz"
    seed=1
    set.seed(seed)
    XY = gen_normal0_data(n=N, d=D, cov_type = cov_type)
    XY_no_intercept = XY[names(XY)!="X1"]
    c_=.1
    k_to_plot=20
    true_theta = seq(0,1, length.out=D)
    true_theta = true_theta[k_to_plot]
    init_theta = rnorm(D, mean=0, sd=1e-5)
  }else if(paper_settings_med_c){
    N = 10**4
    D=20
    cov_type= "Toeplitz"
    seed=1
    set.seed(seed)
    XY = gen_normal0_data(n=N, d=D, cov_type = cov_type)
    XY_no_intercept = XY[names(XY)!="X1"]
    c_=1
    k_to_plot=20
    true_theta = seq(0,1, length.out=D)
    true_theta = true_theta[k_to_plot]
    init_theta = rnorm(D, mean=0, sd=1e-5)
  }
  
  
  
  set.seed(seed)
  out_isgd = sgd(Y ~ ., data=XY_no_intercept, model="lm",
            model.control=list(family=gaussian(link = "identity")),
            start=init_theta,
            sgd.control=list(method="implicit", lr = "one-dim",
                             lr.control=c(scale=1, gamma=1, alpha=c_, c=.505), 
                             size=N, #Get a full matrix of all theta values over the N estimates
                             npasses=1,
                             pass=T,
                             shuffle=T,
                             verbose=T))
  isgd_t = out_isgd$estimates[k_to_plot,]
  
  set.seed(seed)
  out_aisgd = sgd(Y ~ ., data=XY_no_intercept, model="lm",
                 model.control=list(family=gaussian(link = "identity")),
                 start=init_theta,
                 sgd.control=list(method="ai-sgd", lr = "one-dim",
                                  lr.control=c(scale=1, gamma=1, alpha=c_, c=.505), 
                                  size=N, #Get a full matrix of all theta values over the N estimates
                                  npasses=1,
                                  pass=T,
                                  shuffle=T,
                                  verbose=T))
  aisgd_t = out_aisgd$estimates[k_to_plot,]
  
  set.seed(seed)
  out_asgd = sgd(Y ~ ., data=XY_no_intercept, model="lm",
                  model.control=list(family=gaussian(link = "identity")),
                  start=init_theta,
                  sgd.control=list(method="asgd", lr = "one-dim",
                                   lr.control=c(scale=1, gamma=1, alpha=c_, c=.505), 
                                   size=N, #Get a full matrix of all theta values over the N estimates
                                   npasses=1,
                                   pass=T,
                                   shuffle=T,
                                   verbose=T))
  asgd_t = out_asgd$estimates[k_to_plot,]
  
  
  
  set.seed(seed)
  out_sgd = sgd(Y ~ ., data=XY_no_intercept, model="lm",
                 model.control=list(family=gaussian(link = "identity")),
                 start=init_theta,
                 sgd.control=list(method="sgd", lr = "one-dim",
                                  lr.control=c(scale=1, gamma=1, alpha=c_, c=.505), 
                                  size=N, #Get a full matrix of all theta values over the N estimates
                                  npasses=1,
                                  pass=T,
                                  shuffle=T,
                                  verbose=T))
  sgd_t = out_sgd$estimates[k_to_plot,]
  

  
  set.seed(seed)
  out_root_sgd = root_sgd(XY, OLS_gradient,
                c_ = c_,  alpha_lr = .505,
                initial_theta = init_theta, plot_it = F, plot_theta_dim = 1,
                true_theta = seq(0,1, length.out=dim(XY)[2]),
                verbose = F, epochs = 1, fixed_step = F,
                return_trajectory_only=T)
  rootsgd_t = out_root_sgd$theta_t[,k_to_plot]

  
  
  
  set.seed(seed)
  out_trunc_sgd = truncated_sgd(XY, OLS_gradient,
                      c_ = c_,  alpha_lr = .505,
                      initial_theta = init_theta, plot_it = F, plot_theta_dim = 1,
                      true_theta = seq(0,1, length.out=dim(XY)[2]),
                      verbose = F, epochs = 1, fixed_step = F,
                      return_trajectory_only=T)
  tsgd_t = out_trunc_sgd$theta_t[,k_to_plot]
  
  
  set.seed(seed)
  out_sgd_ntrunc_sgd = noisy_truncated_sgd(XY, OLS_gradient,
                                 c_ = c_,  alpha_lr = .505,
                                 initial_theta = init_theta, plot_it = F, plot_theta_dim = 1,
                                 true_theta = seq(0,1, length.out=dim(XY)[2]),
                                 verbose = F, epochs = 1, fixed_step = F,
                                return_trajectory_only=T)
  ntsgd_t = out_sgd_ntrunc_sgd$theta_t[,k_to_plot]
  
  
  small_line = .5
  big_line = 1
  
  lines_df = data.frame(t = 1:N, sgd_t=sgd_t, rootsgd_t=rootsgd_t, ntsgd_t=ntsgd_t, tsgd_t=tsgd_t, asgd_t=asgd_t, aisgd_t=aisgd_t, isgd_t=isgd_t,
                        target=true_theta) %>%
    pivot_longer(cols=-1, names_to="Method", values_to = "theta_t") %>%
    mutate(Method=ifelse(Method=="sgd_t", "Vanilla-SGD",
                         ifelse(Method=="rootsgd_t", "ROOT-SGD",
                                ifelse(Method=="ntsgd_t", "Noisy-truncated-SGD",
                                       ifelse(Method=="tsgd_t", "Truncated-SGD",
                                              ifelse(Method=="asgd_t", "Averaged-SGD",
                                                     ifelse(Method=="aisgd_t",
                                                            "Averaged-implicit-SGD",
                                                            ifelse(Method=="isgd_t", "Last-iterate-implicit-SGD",                                                          ifelse(Method=="target", "(target)", "EXTRA"))))))))) %>%
    
    mutate(Method = factor(Method, levels = c("Vanilla-SGD", "Averaged-SGD", "Last-iterate-implicit-SGD", "Averaged-implicit-SGD", "ROOT-SGD", "Truncated-SGD", "Noisy-truncated-SGD", "(target)"))) 

  if (paper_settings_convergence1){
    p1 = ggplot(lines_df, aes(x=t, y=theta_t, color=Method, linewidth = Method)) +
      geom_line() +
      theme_bw() +
      scale_color_manual(values = c("#ECE5A2", "#CC1616", "orange", "hotpink", "aquamarine", "#8414EA", "#5990FD", "black" )) +
      scale_linewidth_manual(values = c("(target)" = big_line, "Vanilla-SGD" = small_line, "ROOT-SGD" = small_line, "Noisy-truncated-SGD" = small_line, "Truncated-SGD" = small_line, "Averaged-SGD" = small_line, "Averaged-implicit-SGD"=small_line, "Last-iterate-implicit-SGD"=small_line)) +
      coord_cartesian(ylim = c(-1, 2)) +
      ylab(expression(hat(theta)[5]^"(t)")) + #Make sure k_to_plot is correct
      theme(axis.title.y = element_text(angle = 0, vjust = 0.5),
            panel.grid.minor = element_blank()) +
      theme(axis.text.x=element_blank(),
            axis.ticks.x = element_blank(),
            plot.title = element_text(size = 10)) +
      xlab("") +
      ggtitle(paste0("c = ", c_, ", seed = ", seed))
    p1
  }
  
  if (paper_settings_convergence2){
    p2 = ggplot(lines_df, aes(x=t, y=theta_t, color=Method, linewidth = Method)) +
      geom_line() +
      theme_bw() +
      scale_color_manual(values = c("#ECE5A2", "#CC1616", "orange", "hotpink", "aquamarine", "#8414EA", "#5990FD", "black" )) +
      scale_linewidth_manual(values = c("(target)" = big_line, "Vanilla-SGD" = small_line, "ROOT-SGD" = small_line, "Noisy-truncated-SGD" = small_line, "Truncated-SGD" = small_line, "Averaged-SGD" = small_line, "Averaged-implicit-SGD"=small_line, "Last-iterate-implicit-SGD"=small_line)) +
      coord_cartesian(ylim = c(-1, 2)) +
      ylab(expression(hat(theta)[5]^"(t)")) + #Make sure k_to_plot is correct
      xlab("time step t") +
      theme(axis.title.y = element_text(angle = 0, vjust = 0.5),
            panel.grid.minor = element_blank(),
            plot.title = element_text(size = 10))  +
      ggtitle(paste0("c = ", c_, ", seed = ", seed))
    p2
  }
  
  
  
  
  base_R_plot = F #This produces a plot in base R. Not as pretty.
  if (base_R_plot){
    plot(x=1:N, y= sgd_t, type = "l",
         xlab = "time step t", ylab = expression(hat(theta)[k]), main = expression(hat(theta)[k]), col="darkgray",
         ylim=c(-1, 4))
    lines(x=1:N, y = rootsgd_t, col = "hotpink")
    lines(x=1:N, y = ntsgd_t, col = "purple")
    lines(x=1:N, y = tsgd_t, col = "blue")
    lines(x=1:N, y = asgd_t, col = "red")
    lines(x=1:N, y = aisgd_t, col = "orange")
    lines(x=1:N, y = isgd_t, col = "aquamarine")
    abline(h=true_theta, col = "lightgray")
    legend( "topright", legend=c( "Vanilla-SGD", "Implicit", "Averaged implicit", "Averaged", "Root", "Truncated",
                                  "Noisy truncated", "target"),
            col=c("darkgray", "aquamarine", "orange", "red", "hotpink", "blue", "purple", "lightgray" ), lty=1:1, cex=0.5)
  }
  
  
  
  options(repr.plot.width=16, repr.plot.height=8)
  library(latex2exp)
  library(patchwork)
  (p1 / p2) + plot_layout(guides = "collect") & theme(legend.position = 'right', legend.text=element_text(size=8), legend.title=element_text(size = 10, face = "bold") )
  
  save_it=F
  if (save_it){
    setwd(your_wd_plots)
    ggsave(paste0("plots/Cleaned_R/other_methods_examples_D",D, ".pdf"), width = 25, height = 15, units = "cm")
    
  }
  
  if (paper_settings_big_c){
    p3 = ggplot(lines_df, aes(x=t, y=theta_t, color=Method, linewidth = Method)) +
      geom_line() +
      theme_bw() +
      scale_color_manual(values = c("#ECE5A2", "#CC1616", "orange", "hotpink", "aquamarine", "#8414EA", "#5990FD", "black" )) +
      scale_linewidth_manual(values = c("(target)" = big_line, "Vanilla-SGD" = small_line, "ROOT-SGD" = small_line, "Noisy-truncated-SGD" = small_line, "Truncated-SGD" = small_line, "Averaged-SGD" = small_line, "Averaged-implicit-SGD"=small_line, "Last-iterate-implicit-SGD"=small_line)) +
      coord_cartesian(ylim = c(-2, 10)) +
      ylab(expression(hat(theta)[20])) + #Make sure k_to_plot is correct
      xlab("time step t") +
      theme(axis.title.y = element_text(angle = 0, vjust = 0.5),
            panel.grid.minor = element_blank(),
            plot.title = element_text(size = 10),
            axis.text.x=element_blank(),
            axis.ticks.x = element_blank())  +
      ggtitle(paste0("c = ", c_, ", seed = ", seed)) +
      xlab("")
    p3
  }
  
  if (paper_settings_small_c){
    p4 = ggplot(lines_df, aes(x=t, y=theta_t, color=Method, linewidth = Method)) +
      geom_line() +
      theme_bw() +
      scale_color_manual(values = c("#ECE5A2", "#CC1616", "orange", "hotpink", "aquamarine", "#8414EA", "#5990FD", "black" )) +
      scale_linewidth_manual(values = c("(target)" = big_line, "Vanilla-SGD" = small_line, "ROOT-SGD" = small_line, "Noisy-truncated-SGD" = small_line, "Truncated-SGD" = small_line, "Averaged-SGD" = small_line, "Averaged-implicit-SGD"=small_line, "Last-iterate-implicit-SGD"=small_line)) +
      coord_cartesian(ylim = c(-2, 10)) +
      ylab(expression(hat(theta)[20])) + #Make sure k_to_plot is correct
      xlab("time step t") +
      theme(axis.title.y = element_text(angle = 0, vjust = 0.5),
            panel.grid.minor = element_blank(),
            plot.title = element_text(size = 10))  +
      ggtitle(paste0("c = ", c_, ", seed = ", seed))
    p4
  }
  
  if (paper_settings_med_c){
    p5 = ggplot(lines_df, aes(x=t, y=theta_t, color=Method, linewidth = Method)) +
      geom_line() +
      theme_bw() +
      scale_color_manual(values = c("#ECE5A2", "#CC1616", "orange", "hotpink", "aquamarine", "#8414EA", "#5990FD", "black" )) +
      scale_linewidth_manual(values = c("(target)" = big_line, "Vanilla-SGD" = small_line, "ROOT-SGD" = small_line, "Noisy-truncated-SGD" = small_line, "Truncated-SGD" = small_line, "Averaged-SGD" = small_line, "Averaged-implicit-SGD"=small_line, "Last-iterate-implicit-SGD"=small_line)) +
      coord_cartesian(ylim = c(-2, 10)) +
      ylab(expression(hat(theta)[20])) + #Make sure k_to_plot is correct
      xlab("time step t") +
      theme(axis.title.y = element_text(angle = 0, vjust = 0.5),
            panel.grid.minor = element_blank(),
            plot.title = element_text(size = 10),
            axis.text.x=element_blank(),
            axis.ticks.x = element_blank())  +
      ggtitle(paste0("c = ", c_, ", seed = ", seed)) +
      xlab("")
    p5
  }
  
  
  
  (p3 / p5 / p4) + plot_layout(guides = "collect") & theme(legend.position = 'right', legend.text=element_text(size=8), legend.title=element_text(size = 10, face = "bold") )
  
  save_it=F
  if (save_it){
    setwd(your_wd_plots)
    ggsave(paste0("plots/Cleaned_R/other_methods_examples_D",D, ".pdf"), width = 25, height = 15, units = "cm")
    
  }
  
  
}


