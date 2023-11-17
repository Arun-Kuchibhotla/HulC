library(fdrtool)
library(e1071)

library(ggplot2)
library(stats)
library(ggpubr)
library(parallel)
library(dplyr)

start_time = Sys.time()

## psi 

psi = function(x)
{
  return(abs(x)^(2)*sign(x))
}

cap_psi = function(x)
{
  return(abs(x)^(3)/3)
}


f_0n = function(x, n, s_n)
{
  # f(x) = f(x0) + sqrt(s_n/n)* psi( s_n(x- x0))
  # f(x0) = 0
  return( sqrt(s_n/n)*psi(s_n*x) )
}


# X_i\sim U(-1,~1), \epsilon_i\sim\mathcal{N}(0,~1)
# X_i\sim U(-1,~1), \epsilon_i\sim t_5
# X_i\sim U(-1,~1), \epsilon_i\sim\chi_1^2(|X_i|)-\mathbb{E}[\chi_1^2(|X_i|)]
# X_i\sim U(-1,~1), \epsilon_i\sim\mathcal{N}(0,~X_i^2)
# X_i\sim 2\cdot\text{Beta}(2,~3)-1, \epsilon_i\sim t_5

data_gen_same_psi = function(n, s_n, err_model)
{
  
  ## eps ~ gaussian N(0,1)
  if(err_model == 1){
    x = runif(n, min = -1, max = 1)
    y = f_0n(x, n = n, s_n = s_n) + rnorm(n)
  }
  
  ## eps ~ t_5
  if(err_model == 2){
    x = runif(n, min = -1, max = 1)
    y = f_0n(x, n = n, s_n = s_n) + rt(n, df = 5)
  }
  
  ## eps ~ centred chisq(|X|, df = 1)
  if(err_model == 3){
    x = runif(n, min = -1, max = 1)
    
    eps = rnorm(n, mean = abs(x) )
    eps = eps^2 - ( abs(x)^2 + rep(1,n)) 
    y = f_0n(x, n = n, s_n = s_n) + eps 
  }
  
  ## eps ~ gaussian N(0,X^2)
  if(err_model == 4){
    x = runif(n, min = -1, max = 1)
    y = f_0n(x, n = n, s_n = s_n) + rnorm(n, sd = abs(x))
  }
  
  
  ## x~ 2*beta(2,3) - 1 , eps ~ t_5
  if(err_model == 5){
    x = 2*rbeta(n, shape1 = 2, shape2 = 3) - 1
    y = f_0n(x, n = n, s_n = s_n) + rt(n, df = 5)
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


s_n_seq_func = function(n){ 
  return( c( n^(1/6), n^(2/6), n^(3/6),  n^(4/6), n^(5/6) ))
}

#n_iter = 10
n_iter = 500

# ("n","s_n", "iter") 
data_save_2 = array( dim = c(length(n_seq), 5, n_iter))

## brownian motion settings

end = 10
freq = 1000
data_brownian_drift = array( dim = c(length(n_seq), 5, n_iter))

for(n_idx in 1:length(n_seq) )
{
  n_temp = n_seq[n_idx]
  
  print(n_temp)
  
  ## s_n to infty, but s_n/n to 0  
  s_n_seq = s_n_seq_func(n_temp)
  
  for(s_n_idx in 1:length(s_n_seq))
  {
    
    for(iter_idx in 1:n_iter)
    {
      
      s_n_temp = s_n_seq[s_n_idx]
      
      ### to do
      data_temp = data_gen_same_psi(n = n_temp, s_n = s_n_temp, err_model = 1)
      data_save_2[n_idx, s_n_idx, iter_idx] = isoreg_pred(x0 = 0, data = data_temp)
    }
  }
}

#######

df_data_save_2 = data.frame(n_idx = rep( 1:length(n_seq), each = 5*n_iter), s_n_idx = rep( rep(1:5, each = n_iter), length(n_seq) ),
                            iter = rep(1:n_iter, 5*length(n_seq)) , 
                            diff_fhat_f = rep(NA, 5*length(n_seq)*n_iter),
                            brownian_drift = rep(NA, 5*length(n_seq)*n_iter),
                            n = rep(NA, 5*length(n_seq)*n_iter), s_n = rep(NA, 5*length(n_seq)*n_iter))

data_brownian_drift = numeric(n_iter)
for(iter_idx in 1:n_iter)
{
  ### brownian motion setup
  y_brownian.right = rwiener(end = end, frequency = freq)
  y_brownian.left = rwiener(end = end, frequency = freq)
  y_brownian.left = y_brownian.left[(end*freq):1]
  
  t = seq(from = -end, to = end, by = 1/freq)
  y_brownian = c(y_brownian.left, 0, y_brownian.right)
  y_brownian_drift = y_brownian +  cap_psi(t/2) 
  
  gcm = gcmlcm(t, y_brownian_drift, type = "gcm")
  idx_contain_0 = max( which(gcm$x.knots <= 0) )
  data_brownian_drift[iter_idx] = gcm$slope.knots[idx_contain_0]
}

for( i in 1:nrow(df_data_save_2))
{
  n_temp =  n_seq[df_data_save_2$n_idx[i]] 
  df_data_save_2$diff_fhat_f[i] =   data_save_2[ df_data_save_2$n_idx[i], df_data_save_2$s_n_idx[i], df_data_save_2$iter[i]]  
  df_data_save_2$brownian_drift[i] =   data_brownian_drift[df_data_save_2$iter[i]]  
  df_data_save_2$n[i] = n_temp
  df_data_save_2$s_n[i] = s_n_seq_func(n_temp)[df_data_save_2$s_n_idx[i]]  
}

write.csv(df_data_save_2, file = "/home/siddhaarth/iso_reg/log/df_same_psi_log.csv")

print(Sys.time() - start_time)


