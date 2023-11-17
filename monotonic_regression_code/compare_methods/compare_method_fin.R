library(ggplot2)
library(stats)
library(ggpubr)
library(parallel)
library(dplyr)

source("./methods.R")
#################################################################

f_0_hulc = function(x, theta)
{
  return(abs(x)^(theta)*sign(x) )
}

data_gen_hulc = function(n, theta)
{
  x = runif(n, min = -1, max = 1)
  y = f_0_hulc(x, theta = theta) + rnorm(n)
  
  return(data.frame(x = x, y = y))
}
  
#################################################################

theta_seq = seq(from = 0.2, to = 10, length.out = 15)
n_seq = c(50, 100, 250, 1000)
n_iter = 1000

cov_hulc = array(NA, dim = c(length(theta_seq), length(n_seq), n_iter ))
cov_dhz_orac = array(NA, dim = c(length(theta_seq), length(n_seq), n_iter ))
cov_dhz_data = array(NA, dim = c(length(theta_seq), length(n_seq), n_iter ))
cov_subsam = array(NA, dim = c(length(theta_seq), length(n_seq), n_iter ))

width_hulc = array(NA, dim = c(length(theta_seq), length(n_seq), n_iter ))
width_dhz_orac = array(NA, dim = c(length(theta_seq), length(n_seq), n_iter ))
width_dhz_data = array(NA, dim = c(length(theta_seq), length(n_seq), n_iter ))
width_subsam = array(NA, dim = c(length(theta_seq), length(n_seq), n_iter ))

######

for(theta_idx in 1:length(theta_seq))
{
  for( n_idx in 1:length(n_seq))
  {
    quant_dhz_orac_val = 
      quant_dhz(data = NA, n = n_seq[n_idx], alpha = 0.05, data_driven = F,
                theta = theta_seq[theta_idx])
    quant_dhz_data_val = 
      quant_dhz(data = data_gen_hulc(n_seq[n_idx], theta = theta_seq[theta_idx]),
                                   n = n_seq[n_idx], alpha = 0.05, 
                data_driven = T, theta = theta_seq[theta_idx])
    
    for(iter_idx in 1:n_iter)
    {
      data_temp =  data_gen_hulc(n_seq[n_idx], theta = theta_seq[theta_idx])  
      
      CI_temp_hulc = CI_hulc(data_temp)
      CI_temp_dhz_orac = CI_dhz(data_temp,  alpha = 0.05, 
                                quant = quant_dhz_orac_val)
      CI_temp_dhz_data = CI_dhz(data_temp,  alpha = 0.05, 
                                quant = quant_dhz_data_val)
      CI_temp_subsam = CI_subsample(data_temp, b = n_seq[n_idx]^(1/2))
      
      ## see if f(0) belongs to it
      
      cov_hulc[theta_idx, n_idx, iter_idx] =
        ifelse(CI_temp_hulc[1]<=0 & CI_temp_hulc[2]>=0, 1, 0)
      cov_dhz_orac[theta_idx, n_idx, iter_idx] = 
        ifelse(CI_temp_dhz_orac[1]<=0 & CI_temp_dhz_orac[2]>=0, 1, 0)
      cov_dhz_data[theta_idx, n_idx, iter_idx] = 
        ifelse(CI_temp_dhz_data[1]<=0 & CI_temp_dhz_data[2]>=0, 1, 0)
      cov_subsam[theta_idx, n_idx, iter_idx] = 
        ifelse(CI_temp_subsam[1]<=0 & CI_temp_subsam[2]>=0, 1, 0)
      
      ## width of CI
      width_hulc[theta_idx, n_idx, iter_idx] = 
        CI_temp_hulc[2] - CI_temp_hulc[1]
      width_dhz_orac[theta_idx, n_idx, iter_idx] = 
        CI_temp_dhz_orac[2] - CI_temp_dhz_orac[1]
      width_dhz_data[theta_idx, n_idx, iter_idx] = 
        CI_temp_dhz_data[2] - CI_temp_dhz_data[1]
      width_subsam[theta_idx, n_idx, iter_idx] = 
        CI_temp_subsam[2] - CI_temp_subsam[1]
    }
  }
}




df_compare = expand.grid(theta_idx = 1:length(theta_seq),
                         method = c("hulc","dhz_orac","dhz_data","subsam"), 
                         n_idx = 1:length(n_seq)) %>%
  mutate(coverage = NA, width = NA, theta = NA, n = NA)


for( row_idx in 1:nrow(df_compare))
{
  if(df_compare$method[row_idx] == "hulc")
  {
    df_compare$coverage[row_idx] = 
      mean(cov_hulc[df_compare$theta_idx[row_idx], df_compare$n_idx[row_idx], ])
    df_compare$width[row_idx] =
      mean(width_hulc[df_compare$theta_idx[row_idx], df_compare$n_idx[row_idx],])
    df_compare$theta[row_idx] = theta_seq[df_compare$theta_idx[row_idx]]
    df_compare$n[row_idx] = n_seq[df_compare$n_idx[row_idx]]
  }
  
  if(df_compare$method[row_idx] == "dhz_orac")
  {
    df_compare$coverage[row_idx] = 
      mean(cov_dhz_orac[df_compare$theta_idx[row_idx], 
                        df_compare$n_idx[row_idx],])
    df_compare$width[row_idx] =
      mean(width_dhz_orac[df_compare$theta_idx[row_idx], 
                          df_compare$n_idx[row_idx],])
    df_compare$theta[row_idx] = theta_seq[df_compare$theta_idx[row_idx]]
    df_compare$n[row_idx] = n_seq[df_compare$n_idx[row_idx]]
  }
  
  if(df_compare$method[row_idx] == "dhz_data")
  {
    df_compare$coverage[row_idx] =
      mean(cov_dhz_data[df_compare$theta_idx[row_idx], 
                        df_compare$n_idx[row_idx], ])
    df_compare$width[row_idx] =
      mean(width_dhz_data[df_compare$theta_idx[row_idx],
                          df_compare$n_idx[row_idx], ])
    df_compare$theta[row_idx] = theta_seq[df_compare$theta_idx[row_idx]]
    df_compare$n[row_idx] = n_seq[df_compare$n_idx[row_idx]]
  }
  
  if(df_compare$method[row_idx] == "subsam")
  {
    df_compare$coverage[row_idx] =
      mean(cov_subsam[df_compare$theta_idx[row_idx], 
                      df_compare$n_idx[row_idx], ])
    df_compare$width[row_idx] = 
      mean(width_subsam[df_compare$theta_idx[row_idx],
                        df_compare$n_idx[row_idx], ])
    df_compare$theta[row_idx] = theta_seq[df_compare$theta_idx[row_idx]]
    df_compare$n[row_idx] = n_seq[df_compare$n_idx[row_idx]]
  }
}

write.csv(df_compare, file = "./df_compare.csv")

