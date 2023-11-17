######################################################################################
library(dplyr)
library(ggplot2)
#########


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

df_wiener = read.csv("./wiener_only_fin.csv")[,-1]

par(mfrow = c(1,3))

n = 500 
end = 10
freq = 1000

#psi_idx_choice = 4
for(psi_idx_choice in 1:4)
{
  
  ### brownian motion setup
  y_brownian.right = rwiener(end = end, frequency = freq)
  y_brownian.left = rwiener(end = end, frequency = freq)
  y_brownian.left = y_brownian.left[(end*freq):1]
  
  t = seq(from = -end, to = end, by = 1/freq)
  y_brownian = c(y_brownian.left, 0, y_brownian.right)
  x = seq(from = -1, to = 1, length.out = n)
  epsilon = rnorm (n, mean = 0, sd = 1)
  
  if(psi_idx_choice==1)
  { 
    theta_1 = 2
    y = f_0n_eg1(x,n, theta = theta_1) + epsilon
    f_0n_x =  f_0n_eg1(x,n,theta = theta_1)
    y_brownian_drift = y_brownian +  Psi_eg1(t/2, theta = theta_1) 
  }
  
  if(psi_idx_choice==2)
  { 
    theta_2 = 2
    y = f_0n_eg1(x,n, theta = theta_2) + epsilon
    f_0n_x =  f_0n_eg2(x,n,theta = theta_2)
    y_brownian_drift = y_brownian + Psi_eg1(t/2, theta = theta_2)
  }
  
  if(psi_idx_choice==3)
  { 
    y = f_0n_eg3(x,n) + epsilon
    f_0n_x =  f_0n_eg3(x,n)
    y_brownian_drift = y_brownian +  Psi_eg3(t/2) 
  }    
  
  if(psi_idx_choice==4)
  { 
    y = f_0n_eg4(x,n) + epsilon
    f_0n_x =  f_0n_eg4(x,n)
    y_brownian_drift = y_brownian +  Psi_eg4(t/2) 
  }    
  
  
  df_xy = data.frame(x = x, y = y, f_0n_x = f_0n_x)
  p1 = ggplot(df_xy, 
              aes(x = x)) +
    geom_point(aes(y = y), alpha = .5) +
    geom_line(aes(y = f_0n_x), color = 'orange', linewidth = 1.5) +
    theme_bw() +
    theme(panel.grid = element_blank()) +
    theme( legend.text=element_text(size=14),
           legend.title =element_text(size=14),
           strip.text.x = element_text(size = 14),
           axis.text.x = element_text(size = 11),
           axis.text.y = element_text(size = 11),
           axis.title = element_text(size = 11))
  p1
  
  
  gcm = gcmlcm(t, y_brownian_drift, type = "gcm")
  idx_contain_0 = max( which(gcm$x.knots <= 0) )
  df_browninan = data.frame(t = t, y = y_brownian_drift) 
  df_gcm = data.frame(xknots = gcm$x.knots, yknots = gcm$y.knots)
  
  p2 = ggplot() +
    geom_line(data = df_browninan, aes(x = t, y = y)) + 
    geom_line(data = df_gcm, aes(x = xknots, y = yknots), color = "red" ) +
    geom_line(data = df_gcm[idx_contain_0 + (0:1),], aes(x = xknots, y = yknots), color = "green" ) +
    theme_bw() +
    theme(panel.grid = element_blank()) + ylim(NA,8) + 
    theme( legend.text=element_text(size=14),
           legend.title =element_text(size=14),
           strip.text.x = element_text(size = 14),
           axis.text.x = element_text(size = 11),
           axis.text.y = element_text(size = 11),
           axis.title = element_text(size = 11))
  
  p2  
  
  #######
  
  skew_bd = df_wiener %>% filter( psi_idx == psi_idx_choice) %>% .$brownian_drift %>% skewness() %>% round(4)
  
  p3 = ggplot(df_wiener %>% filter( psi_idx == psi_idx_choice), aes(x=brownian_drift)) + 
    geom_histogram(color="black", fill="grey" ,aes(y=..density..)) + 
    theme_bw() +
    labs(x = "Sample values")+
    theme(panel.grid = element_blank()) + 
    annotate("text", x = 0, y = 1, label = paste("Skewness:", skew_bd)) + 
    theme( legend.text=element_text(size=14),
           legend.title =element_text(size=14),
           strip.text.x = element_text(size = 14),
           axis.text.x = element_text(size = 11),
           axis.text.y = element_text(size = 11),
           axis.title = element_text(size = 11))
  p3
  
  p123 = ggarrange(p1, p2, p3, 
                   ncol=3, nrow=1, common.legend = TRUE, legend="bottom")
  
  p123
  ggsave(p123 , file = paste("./plots/psi_",
                             psi_idx_choice, ".pdf",sep = ""),
         width = 9, height = 3, dpi = 300, units = "in" )
}
