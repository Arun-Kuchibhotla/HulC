library(ggplot2)
library(stats)
library(ggpubr)
library(parallel)
library(dplyr)

start_time = Sys.time()
source("./methods.R")
#################################################################

## heteroscadistic error model
## sims with psi anti-symmetric
## varies with n
## 1 + x + x^3

## case 1 

f_0n = function(x, n, s_n, exmp)
{
  if(exmp==1){
    theta_1 = 1.5
    temp =  sqrt(s_n/n)*(abs(s_n*x))^theta_1*sign(x)
  }
  
  if(exmp==2){
    temp =  sqrt(s_n/n)*2*( (s_n*x) + sin(s_n*x) )
  }
  
  if(exmp==3){
    f_beta = function(t){ return( 5*(pbeta((t+1)/2, shape1 = 1.21, 
                                           shape2 = 2.45)
                                     - pbeta(0.5, shape1 = 1.21,
                                             shape2 = 2.45)) )} 
    temp =  sqrt(s_n/n)*f_beta(s_n*x)
  }
  
  if(exmp==4){
    temp =  x/n^(1/5) + x^3
  }
  
  if(exmp == 5){
    temp = sqrt(s_n/n)*( (s_n*x) + (s_n*x)^3  + (s_n*x)^5 )
  }
  
  if(exmp == 6){
    theta_6 = 1.5
    temp =  sqrt(s_n/n)*(abs(s_n*x))^theta_6*sign(x)
  }
  
  if(exmp == 7){
    theta_7 = 1.5
    temp =  sqrt(s_n/n)*(abs(s_n*x))^theta_7*sign(x)
  }
  return( temp )
}


data_gen_exmp = function(n, exmp )
{
  if(exmp == 1){
    ## eps ~ gaussian N(0,|X| + 1)
    ## psi = |x|^theta
    x = sort(runif(n, min = -1, max = 1))
    y = f_0n(x, n, n^{1/3}, 1) + rnorm(n, sd = 2*abs(x)+ 1 )  
  }
  
  if(exmp == 2){
    ## eps ~ gaussian N(0,1)
    x = sort(runif(n, min = -1, max = 1))
    y = f_0n(x, n, n^{1/3}, 2) + rnorm(n, sd = 1/2)  
  }
  
  if(exmp == 3){
    x = sort(runif(n, min = -1, max = 1))
    y = f_0n(x, n, n^{1/3}, 3) + rnorm(n, sd = 1/2 )  
  }
  
  if(exmp == 4){
    theta_4 = 1
    x = sort(runif(n, min = -1, max = 1))
    y = f_0n(x, n, n^{1/3}, 4) + rnorm(n, sd = 1/2 )  
  }
  
  if(exmp == 5){
    x = sort(runif(n, min = -1, max = 1))
    y =  f_0n(x, n, log(log(n)), 5) + rnorm(n, sd = 1 )  
  }
  
  if(exmp == 6)
  {
    x = sort(2*rbeta(n, shape1 = 2, shape2 = 3) - 1)
    y = f_0n(x, n, n^{1/3}, 6) + rnorm(n, sd = 1)
  }
  
  if(exmp == 7){
    ## eps ~ centred chisq(|X|, df = 1)
    ## psi = |x|^theta
    
    x = sort(runif(n, min = -1, max = 1))
    
    eps = (rnorm(n, mean = abs(x) ))
    eps = eps^2 - ( abs(x)^2 + rep(1,n))
    
    y = f_0n(x, n, n^{1/3}, 7) + eps/3
  }
  
  return(data.frame(x = x, y = y))
}

n_exmp = 7

n_seq = seq(from = 500, to = 5000, by = 500)
n_iter = 1000

cov_hulc = array(NA, dim = c(n_exmp, length(n_seq), n_iter ))
cov_dhz_data = array(NA, dim = c(n_exmp, length(n_seq), n_iter ))
cov_subsam = array(NA, dim = c(n_exmp, length(n_seq), n_iter ))

width_hulc = array(NA, dim = c(n_exmp, length(n_seq), n_iter ))
width_dhz_data = array(NA, dim = c(n_exmp, length(n_seq), n_iter ))
width_subsam = array(NA, dim = c(n_exmp, length(n_seq), n_iter ))


######

for(exmp_idx in 1:n_exmp)
{
  for( n_idx in 1:length(n_seq))
  {
    for(iter_idx in 1:n_iter)
    {
      data_temp =  data_gen_exmp(n_seq[n_idx], exmp = exmp_idx) 
      
      quant_dhz_data_val = quant_dhz(data = data_temp, n = n_seq[n_idx], 
                                     alpha = 0.05, data_driven = T, theta = NA)
      
      CI_temp_hulc = CI_hulc(data_temp)
      CI_temp_dhz_data = CI_dhz(data_temp,  alpha = 0.05, 
                                quant = quant_dhz_data_val)
      CI_temp_subsam = CI_subsample(data_temp, b = n_seq[n_idx]^(1/2))
      
      ## see if f(0) belongs to it
      
      cov_hulc[exmp_idx, n_idx, iter_idx] = 
        ifelse(CI_temp_hulc[1]<=0 & CI_temp_hulc[2]>=0, 1, 0)
      cov_dhz_data[exmp_idx, n_idx, iter_idx] = 
        ifelse(CI_temp_dhz_data[1]<=0 & CI_temp_dhz_data[2]>=0, 1, 0)
      cov_subsam[exmp_idx, n_idx, iter_idx] =
        ifelse(CI_temp_subsam[1]<=0 & CI_temp_subsam[2]>=0, 1, 0)
      
      ## width of CI
      width_hulc[exmp_idx, n_idx, iter_idx] = 
        CI_temp_hulc[2] - CI_temp_hulc[1]
      width_dhz_data[exmp_idx, n_idx, iter_idx] = 
        CI_temp_dhz_data[2] - CI_temp_dhz_data[1]
      width_subsam[exmp_idx, n_idx, iter_idx] =
        CI_temp_subsam[2] - CI_temp_subsam[1]
    }
  }
}




df_compare = expand.grid(exmp_idx = 1:n_exmp, 
                         method = c("hulc","dhz_data","subsam"), 
                         n_idx = 1:length(n_seq)) %>%
  mutate(coverage = NA, width = NA, n = NA)


for( row_idx in 1:nrow(df_compare))
{
  if(df_compare$method[row_idx] == "hulc")
  {
    df_compare$coverage[row_idx] =
      mean(cov_hulc[df_compare$exmp_idx[row_idx], 
                    df_compare$n_idx[row_idx], ])
    
    df_compare$width[row_idx] = 
      mean(width_hulc[df_compare$exmp_idx[row_idx], 
                      df_compare$n_idx[row_idx], ])
    
    df_compare$n[row_idx] = n_seq[df_compare$n_idx[row_idx]]
  }
  
  if(df_compare$method[row_idx] == "dhz_data")
  {
    df_compare$coverage[row_idx] = 
      mean(cov_dhz_data[df_compare$exmp_idx[row_idx],
                        df_compare$n_idx[row_idx], ])
    df_compare$width[row_idx] = 
      mean(width_dhz_data[df_compare$exmp_idx[row_idx], 
                          df_compare$n_idx[row_idx], ])
    df_compare$n[row_idx] = n_seq[df_compare$n_idx[row_idx]]
  }
  
  if(df_compare$method[row_idx] == "subsam")
  {
    df_compare$coverage[row_idx] = 
      mean(cov_subsam[df_compare$exmp_idx[row_idx], 
                      df_compare$n_idx[row_idx], ])
    df_compare$width[row_idx] =
      mean(width_subsam[df_compare$exmp_idx[row_idx], 
                        df_compare$n_idx[row_idx], ])
    df_compare$n[row_idx] = n_seq[df_compare$n_idx[row_idx]]
  }
}

write.csv(df_compare, file = "./df_compare_more_exmp_large_n_1.csv")

print(Sys.time() - start_time)





