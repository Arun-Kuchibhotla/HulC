library(ggplot2)
library(stats)
library(ggpubr)
library(parallel)
library(dplyr)
library(e1071)
library(fdrtool)
library(e1071)

start_time = Sys.time()


#### psi1 

psi1 = function(x)
{
  return(x)
}

cap_psi1 = function(x)
{
  return(x^2/2)
}

#### psi2 

psi2 = function(x)
{
  temp = numeric(length(x))
  for( i in 1:length(x))
  {
    if(x[i]>=0){
      temp[i] = x[i]/2
    }else{
      temp[i] = x[i]
    }
  }
  return(temp)
}

cap_psi2 = function(x)
{
  temp = numeric(length(x))
  for( i in 1:length(x))
  {
    if(x[i]>=0){
      temp[i] = x[i]^2/4
    }else{
      temp[i] = x[i]^2/2
    }
  }
  return(temp)
}

#### psi3 

psi3 = function(x)
{
  temp = numeric(length(x))
  for( i in 1:length(x))
  {
    if(x[i]>=0){
      temp[i] = x[i]^2
    }else{
      temp[i] = x[i]^3/3
    }
  }
  return(temp)
}

cap_psi3 = function(x)
{
  temp = numeric(length(x))
  for( i in 1:length(x))
  {
    if(x[i]>=0){
      temp[i] = x[i]^3/3
    }else{
      temp[i] = x[i]^4/12
    }
  }
  return(temp)
}

#### psi4 

psi4 = function(x)
{
  return( as.numeric(abs(x)>0.1)*sign(x) )
}

cap_psi4 = function(x)
{
  temp = numeric(length(x))
  for( i in 1:length(x))
  {
    if( x[i] >= 0.1){
      temp[i] = x[i]-0.1
    }else if( x[i] <= -0.1){
      temp[i] = -x[i] - 0.1
    }else{
      temp[i] = 0
    }
  }
  return(temp)
}

############

n_psi = 4

f_0n = function(x, n, s_n, psi_model)
{
  if(psi_model==1){
    temp =  sqrt(s_n/n)*psi1(s_n*x)
  }
  
  if(psi_model==2){
    temp =  sqrt(s_n/n)*psi2(s_n*x)
  }
  
  if(psi_model==3){
    temp =  sqrt(s_n/n)*psi3(s_n*x)
  }
  
  if(psi_model==4){
    temp =  sqrt(s_n/n)*psi4(s_n*x)
  }
  
  return( temp )
}

# X_i\sim U(-1,~1), \epsilon_i\sim\mathcal{N}(0,~1)
# X_i\sim U(-1,~1), \epsilon_i\sim t_5
# X_i\sim U(-1,~1), \epsilon_i\sim\chi_1^2(|X_i|)-\mathbb{E}[\chi_1^2(|X_i|)]
# X_i\sim U(-1,~1), \epsilon_i\sim\mathcal{N}(0,~X_i^2)
# X_i\sim 2\cdot\text{Beta}(2,~3)-1, \epsilon_i\sim t_5

data_gen_same_rate = function(n, s_n, err_model, psi_model)
{
  
  ## eps ~ gaussian N(0,1)
  if(err_model == 1){
    x = runif(n, min = -1, max = 1)
    y = f_0n(x, n = n, s_n = s_n, psi_model = psi_model) + rnorm(n)
  }
  
  ## eps ~ t_5
  if(err_model == 2){
    x = runif(n, min = -1, max = 1)
    y = f_0n(x, n = n, s_n = s_n, psi_model = psi_model) + rt(n, df = 5)
  }
  
  ## eps ~ centred chisq(|X|, df = 1)
  if(err_model == 3){
    x = runif(n, min = -1, max = 1)
    
    eps = rnorm(n, mean = abs(x) )
    eps = eps^2 - ( abs(x)^2 + rep(1,n)) 
    y = f_0n(x, n = n, s_n = s_n, psi_model = psi_model) + eps 
  }
  
  ## eps ~ gaussian N(0,X^2)
  if(err_model == 4){
    x = runif(n, min = -1, max = 1)
    y = f_0n(x, n = n, s_n = s_n, psi_model = psi_model) + rnorm(n, sd = abs(x))
  }
  
  
  ## x~ 2*beta(2,3) - 1 , eps ~ t_5
  if(err_model == 5){
    x = 2*rbeta(n, shape1 = 2, shape2 = 3) - 1
    y = f_0n(x, n = n, s_n = s_n, psi_model = psi_model) + rt(n, df = 5)
  }
  
  return(data.frame(x = x, y = y))
}

### isoreg

isoreg_pred = function(x0, data)
{
  f_hat = isoreg(data$x, data$y)
  f_hat_predictor = as.stepfun(f_hat) ## returns step prediction function
  return(f_hat_predictor(x0))
}
########

n_seq = floor(exp(seq(from  = 6.5, to = 10, length.out = 11 ))) 
n_iter = 500

# ("n","psi", "iter") 
data_save_1 = array( dim = c(length(n_seq), n_psi, n_iter))

for(n_idx in 1:length(n_seq) )
{
  n_temp = n_seq[n_idx]
  
  print(n_temp)
  
  ## s_n to infty, but s_n/n to 0  
  s_n_temp = (n_temp)^(1/3)
  
  for(psi_idx in 1:4)
  {
    for(iter_idx in 1:n_iter)
    {
      
      data_temp = data_gen_same_rate(n = n_temp, s_n = s_n_temp,
                                     err_model = 1, psi_model = psi_idx)
      data_save_1[n_idx, psi_idx, iter_idx] = isoreg_pred(x0 = 0,
                                                          data = data_temp)
    }
  }
}


## brownian motion settings
end = 15
freq = 1000
data_brownian_drift = matrix(nrow = n_psi, ncol = n_iter)

for(psi_idx in 1:4)
{
  for(iter_idx in 1:n_iter)
  {
    ### brownian motion setup
    y_brownian.right = rwiener(end = end, frequency = freq)
    y_brownian.left = rwiener(end = end, frequency = freq)
    y_brownian.left = y_brownian.left[(end*freq):1]
    
    t = seq(from = -end, to = end, by = 1/freq)
    y_brownian = c(y_brownian.left, 0, y_brownian.right)
    
    
    if(psi_idx==1)
    { y_brownian_drift = y_brownian +  cap_psi1(t/2) }
    if(psi_idx==2)
    { y_brownian_drift = y_brownian + cap_psi2(t/2)}
    if(psi_idx==3)
    { y_brownian_drift = y_brownian +  cap_psi3(t/2)}    
    if(psi_idx==4)
    { y_brownian_drift = y_brownian +  cap_psi4(t/2)}
    
    gcm = gcmlcm(t, y_brownian_drift, type = "gcm")
    idx_contain_0 = max( which(gcm$x.knots <= 0) )
    data_brownian_drift[psi_idx, iter_idx] = 
      gcm$slope.knots[idx_contain_0]
  }
}


###

df_data_save_1 = data.frame(n_idx = rep( 1:length(n_seq), each = n_psi*n_iter), 
                            psi_idx = rep( rep(1:n_psi, each = n_iter), 
                                           length(n_seq) ),
                            iter = rep(1:n_iter, n_psi*length(n_seq)),
                            diff_fhat_f = rep(NA, n_psi*length(n_seq)*n_iter),
                            brownian_drift = rep(NA,n_psi*length(n_seq)*n_iter), 
                            n = rep(NA, n_psi*length(n_seq)*n_iter),
                            s_n = rep(NA, n_psi*length(n_seq)*n_iter))

for( i in 1:nrow(df_data_save_1))
{
  n_temp =  n_seq[df_data_save_1$n_idx[i]] 
  
  df_data_save_1$diff_fhat_f[i] =   data_save_1[ df_data_save_1$n_idx[i],
                                                 df_data_save_1$psi_idx[i],
                                                 df_data_save_1$iter[i]]  
  df_data_save_1$brownian_drift[i] = 
    data_brownian_drift[ df_data_save_1$psi_idx[i], df_data_save_1$iter[i]]  
  
  df_data_save_1$n[i] = n_temp
  df_data_save_1$s_n[i] = n_temp^(1/3) 
}

write.csv(df_data_save_1, file = "./df_same_rate_log_1.csv")

print(Sys.time() - start_time)
