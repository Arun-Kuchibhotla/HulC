library(ggplot2)
library(stats)
library(ggpubr)
library(doParallel)
library(dplyr)

### isoreg

isoreg_pred = function(x0, data)
{
  f_hat = isoreg(data$x, data$y)
  f_hat_predictor = as.stepfun(f_hat) ## returns step prediction function
  return(f_hat_predictor(x0))
}

############################################################
############ Deng ##########################################

quant_dhz = function(data, n, alpha = 0.05, B = 10, data_driven, theta)
{
  # f0 = 2x if not data driven
  # B = repeated obs (recommended  10^6)
  # sigma = 1 
  
  temp_rep = numeric(B)
  
  if(data_driven == F){
    
    x_b = seq(from = -1 , to = 1, length.out = n)
    f_x_b = abs(x_b)^theta*sign(x_b)
    f_true_x0 = 0
    sig_f = 1
    
  }else{
    
    x_b = seq(from = min(data$x), to = max(data$x), length.out = n)
    data_iso = isoreg( x = data$x, y = data$y)
    
    if(data_iso$isOrd){
      f_smooth = loess(y~x, data = data.frame(x = data$x, y = data_iso$yf) )
    }else{
      f_smooth = loess(y~x, data = data.frame(x = data$x[data_iso$ord], 
                                              y = data_iso$yf) )
    }
    
    f_x_b = as.numeric( predict(f_smooth, newdata = data.frame( x = x_b)) ) 
    f_true_x0 = as.numeric( predict(f_smooth, newdata = data.frame(x = 0)) )
    
    ###diff estimator for variance
    var_f = 
      sum(( (2*data$y[2:(n-1)]) - data$y[1:(n-2)] - data$y[3:n] )^2)/(6*(n - 2))
    sig_f = sqrt(var_f)
  }
  
  # n such that x_n < 0 <= x_(n+1) 
  n_x0 = max( which(x_b < 0)) 
  
  print("starting bootstrap")
  
  pb = txtProgressBar(min = 0, style = 3, max = B, initial = 0) 
  
  #temp_nuv = numeric(B)
  for(b in 1:B)
  {
    y_b =  f_x_b + rnorm(n, sd = sig_f)
    
    f_iso_b = isoreg(x_b, y_b)
    f_hat_b = as.stepfun(f_iso_b)(0)    
    
    ### in case 0 is in first kink, then idx_x0_cross is 0
    #### in case 0 in in last kink, then idx_x0_cross is n
    
    if( min(f_iso_b$iKnots)< n_x0 & max(f_iso_b$iKnots) > n_x0 )
    {
      ## idx of knot where x0 is
      idx_x0_cross_b = max(which( f_iso_b$iKnots < n_x0 ))
      n_uv_hat_b = 
        f_iso_b$iKnots[idx_x0_cross_b + 1] - f_iso_b$iKnots[idx_x0_cross_b]
    }
    
    if( min(f_iso_b$iKnots) >= n_x0)
    {
      n_uv_hat_b = f_iso_b$iKnots[1] - 0
    }
    
    if( max(f_iso_b$iKnots) <= n_x0)
    {
      n_uv_hat_b = n - f_iso_b$iKnots[length(f_iso_b$iKnots)]
    }
    
    ## sqrt(n_uv)*|f_hat(x0) - f(x0)| / sigma 
    temp_rep[b] = sqrt(n_uv_hat_b)*abs(f_hat_b - f_true_x0)/sig_f 
    
    setTxtProgressBar(pb,b)
  }
  print("bootstrap over")
  
  return(quantile(temp_rep, prob = 1-alpha))
}

######################################################


CI_dhz = function(data, alpha = 0.05, x0 = 0, quant = NA, data_driven = F, theta = NA )
{
  n = nrow(data)
  if(is.na(quant)){ quant = quant_dhz(data, n, alpha, data_driven = data_driven, theta = theta) }
  
  f_iso = isoreg(data$x, data$y)
  f_hat_x0 = as.stepfun(f_iso)(x0)    
  
  n_x0 = max( which( sort(data$x) < x0 ))     
  
  ### in case 0 is in first kink, then idx_x0_cross is 0
  #### in case 0 in in last kink, then idx_x0_cross is n
  
  if( min(f_iso$iKnots)< n_x0 & max(f_iso$iKnots) > n_x0 )
  {
    idx_x0_cross = max(which( f_iso$iKnots < n_x0 )) ## idx of knot where x0 is
    n_uv_hat = f_iso$iKnots[idx_x0_cross + 1] - f_iso$iKnots[idx_x0_cross]
  }
  
  if( min(f_iso$iKnots) >= n_x0)
  {
    n_uv_hat = f_iso$iKnots[1] - 0
  }
  
  if( max(f_iso$iKnots) <= n_x0)
  {
    n_uv_hat = n - f_iso$iKnots[length(f_iso$iKnots)]
  }
  
  ###diff estimator for variance
  var_f_hat = 
    sum(( (2*data$y[2:(n-1)]) - data$y[1:(n-2)] - data$y[3:n] )^2)/(6*(n - 2))
  sig_f_hat = sqrt(var_f_hat)
  
  #print(n_uv_hat)
  #print(sig_f_hat)
  return( f_hat_x0 + (quant*sig_f_hat/sqrt(n_uv_hat))*c(-1,1)   )
}

############################################################
############ Subsampling ##########################################

subsample = function(data, b,N, tgrid, gamma = 0, x0)  
{
  ### b = subsample size
  ### N = number of subsamples
  ### tgrid = grid of values on which to evaluate the inverse cdf
  ### gamma:  root = n^gamma (theta.hat - theta)
  ### Note: set gamma = 0 to get the unnormalized root:  (theta.hat - theta)
  ###
  ### return: G^{-1}(t) for t in tgrid
  ###         where G(x) = P( n^gamma (theta.hat-theta) < = x)
  
  gridsize = 100
  n = nrow(data)
  theta.hat = isoreg_pred(x0 = x0, data = data)
  
  tmp = rep(0,N)
  
  pb = txtProgressBar(min = 0, style = 3, max = N, initial = 0) 
  for(i in 1:N){
    I = sample(1:n,size=b,replace=FALSE)
    tmp[i] = isoreg_pred(x0 = x0, data = data[I,,drop=FALSE])
    setTxtProgressBar(pb,i)
  }
  
  xx = b^gamma*(tmp - theta.hat)
  xgrid = seq(min(xx),max(xx),length=gridsize)
  G = rep(0,gridsize)
  for(i in 1:gridsize){
    G[i] = mean(b^gamma*(tmp - theta.hat) <= xgrid[i])
  }
  
  ### invert cdf
  out = rep(0,length(tgrid))
  for(i in 1:length(tgrid)){
    out[i] = min(xgrid[G >= tgrid[i]])
  }
  return(out)
}  

##############################

Estimate_Rate = function(data,beta,N,x0)
{
  ### 1 > beta_1 > ... > beta_I > 0
  n = nrow(data)
  I = length(beta)
  tgrid = c(.1,.6,.2,.7,.3,.8,.4,.9)  ##suggested grid
  J = 4
  y = matrix(0,I,J)
  even = c(2,4,6,8)
  odd  = c(1,3,5,7)
  b = rep(0,I)
  for(i in 1:I){
    b[i] = round(n^beta[i])
    tmp = subsample(data = data, b = b[i], N = N, tgrid = tgrid,x0 = x0)
    y[i,] = log(tmp[even] - tmp[odd])
  }
  y = apply(y,1,mean)
  ybar = mean(y)
  alpha = -sum((y-ybar)*(log(b)-mean(log(b))))/sum( (log(b)-mean(log(b)))^2)
  if(is.nan(alpha)){
    alpha <- 0
  }
  return(alpha)
}

##############################

CI_subsample = function(data,b,N=150, alpha = 0.05, beta=(9:1)/10, x0 = 0)
{
  ### 1 > beta_1 > ... > beta_I > 0
  n = nrow(data)
  theta.hat = isoreg_pred(x0 = 0, data = data)
  tgrid = c(alpha/2,1-alpha/2)
  gamma.hat = Estimate_Rate(data,beta,N,x0)
  tmp = subsample(data,b,N,tgrid,gamma=gamma.hat,x0)
  left  = theta.hat - tmp[2]/n^gamma.hat
  right = theta.hat - tmp[1]/n^gamma.hat
  return(c(left,right))
}

############################################################
############ HulC ##########################################


B_alpha_func = function(alpha = 0.05)
{
  B.alpha = ceiling(log(2/alpha, base = 2))
  
  ### rand_p is p such that alpha/2 = p*( 2^B ) + (1-p)*( 2^{B-1} )
  rand_p = 2 - (alpha/2)*2^(B.alpha)
  if( runif(1) > rand_p) {
    B.alpha = B.alpha - 1
  }
  return(B.alpha)
}

##############################

CI_hulc = function(data, alpha = 0.05, x0 = 0)
{
  B_alpha = B_alpha_func(alpha = alpha) 
  
  f_hat_intervals = rep(0, B_alpha)
  split_points = split( sample(nrow(data)), (1:nrow(data))%%B_alpha )
  
  
  for(b_idx in 1:B_alpha){
    f_hat_intervals[b_idx] = isoreg_pred(x0 = 0, data =
                                           data[split_points[[b_idx]],])
  }
  
  return(range(f_hat_intervals))
}

