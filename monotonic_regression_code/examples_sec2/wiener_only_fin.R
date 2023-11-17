library(fdrtool)
library(e1071)

library(ggplot2)
library(stats)
library(ggpubr)
library(parallel)
library(dplyr)

start_time = Sys.time()

isoreg_pred = function(x0, data)
{
  f_hat = isoreg(data$x, data$y)
  f_hat_predictor = as.stepfun(f_hat) ## returns step prediction function
  return(f_hat_predictor(x0))
}

#################################################################

psi_small_eg1 = function(x, theta = 2, A = 1)
{
  return( A*abs(x)^(theta)*sign(x) )
}

Psi_eg1 = function(x, theta = 2, A = 1)
{
  return( A*abs(x)^(theta+1)/(theta + 1) )
}


f_0n_eg1 = function(x,n,theta)
{
  s_n = n^(1/(2*theta + 1))
  return( sqrt(s_n/n)*psi_small_eg1(s_n*x, theta = theta) )  
}

#################################################################

#### psi2 
## same as example 2.1

f_0n_eg2 = function(x,n,theta)
{
  s_n = (sqrt(n)*log(n)/(2*theta + 1))^(2/(2*theta + 1)) 
  return( sqrt(s_n/n)*psi_small_eg1(s_n*x, theta = theta) )  
}

#################################################################

#### psi3 
## psi is x where x>0 and x/3 when x<0

psi_small_eg3 = function(x)
{
  temp = numeric(length(x))
  for( i in 1:length(x))
  {
    if(x[i]>=0){
      temp[i] = x[i]
    }else{
      temp[i] = (x[i]/3)
    }
  }
  return(temp)
}

Psi_eg3 = function(x)
{
  temp = numeric(length(x))
  for( i in 1:length(x))
  {
    if(x[i]>=0){
      temp[i] = x[i]^2/2
    }else{
      temp[i] = (x[i]^2/6)
    }
  }
  return(temp)
}

f_0n_eg3 = function(x,n)
{
  s_n = n^(1/3)
  return( sqrt(s_n/n)*psi_small_eg3(s_n*x) )
}

#################################################################

#### psi4 

## s_n = n^(1/5)
psi_small_eg4 = function(x)
{
  return(x)
}

Psi_eg4 = function(x)
{
  return( (x^2/2))
}

f_0n_eg4 = function(x,n)
{
  a_n1 = 1/(n^(1/5))
  a_n3 = 1
  return( (a_n1*x) + ((a_n3/6)*x^3) )
}

#################################################################

## test setting

end = 10
freq = 1000
n_iter = 10^5

n_psi = 4

val_brownian_drift = matrix(nrow = n_psi, ncol = n_iter )

for( psi_idx in 1:4)
{
  print(psi_idx)
  for( iter_idx in 1:n_iter){
    
    ### brownian motion setup
    y_brownian.right = rwiener(end = end, frequency = freq)
    y_brownian.left = rwiener(end = end, frequency = freq)
    y_brownian.left = y_brownian.left[(end*freq):1]
    
    t = seq(from = -end, to = end, by = 1/freq)
    y_brownian = c(y_brownian.left, 0, y_brownian.right)
    
    #### data setup + drift addition w.r.t to phi
    
    if(psi_idx==1)
    { 
      theta_1 = 2
      y_brownian_drift = y_brownian +  Psi_eg1(t/2, theta = theta_1) 
    }
    
    if(psi_idx==2)
    { 
      theta_2 = 2
      y_brownian_drift = y_brownian + Psi_eg1(t/2, theta = theta_2)
    }
    
    if(psi_idx==3)
    { 
      y_brownian_drift = y_brownian +  Psi_eg3(t/2) 
    }
    
    if(psi_idx==4)
    { 
      y_brownian_drift = y_brownian +  Psi_eg4(t/2) 
    }    
    
    
    gcm = gcmlcm(t, y_brownian_drift, type = "gcm")
    
    idx_contain_0 = max( which(gcm$x.knots <= 0) )
    val_brownian_drift[psi_idx, iter_idx] = gcm$slope.knots[idx_contain_0]
  }
}

df_wiener_only = expand.grid(iter = 1:n_iter, psi_idx = 1:n_psi) %>% mutate(brownian_drift = NA)

for(row_idx in 1:nrow(df_wiener_only))
{
  df_wiener_only$brownian_drift[row_idx] = val_brownian_drift[df_wiener_only$psi_idx[row_idx], df_wiener_only$iter[row_idx]]
}


write.csv(df_wiener_only, file = "./wiener_only_fin.csv")
print(Sys.time() - start_time)


