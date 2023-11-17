######################################################################################
library(dplyr)
library(ggplot2)

### s_n = (n^1/3)

df_same_rate = read.csv("./df_same_rate_log_1.csv")[,-1]

#######

n_idx_choice = 11

df_same_rate_11 = df_same_rate %>% filter(n_idx == n_idx_choice) %>%
  mutate(f_hat_scaled = diff_fhat_f*sqrt(n/s_n)*0.5) %>%
  mutate(f_hat_scaled_sort = NA, brownian_drift_sort = NA)

for(psi_idx_choice in 1:4 )
{
  temp_bd = df_same_rate_11 %>% filter( psi_idx == psi_idx_choice) %>% .$brownian_drift %>% sort()
  temp_f_hat = df_same_rate_11 %>% filter( psi_idx == psi_idx_choice) %>% .$f_hat_scaled %>% sort()
  df_same_rate_11$brownian_drift_sort[df_same_rate_11$psi_idx == psi_idx_choice] = temp_bd
  df_same_rate_11$f_hat_scaled_sort[df_same_rate_11$psi_idx == psi_idx_choice] = temp_f_hat
}


p3_same_rate = ggplot( data = df_same_rate_11,
                       aes(x = brownian_drift_sort, y = f_hat_scaled_sort, 
                           color = as.character(psi_idx))) +   
  geom_line(aes(group = psi_idx)) +
  geom_abline( slope = 1, intercept = 0, linetype="dashed", size=0.4) + 
  labs(colour = expression(paste(psi," choice"))  ,
       y = "Scaled estimates of f(0)", 
       x ="Samples from asymptotic distribution")  + 
  facet_wrap(~psi_idx) +
  theme_bw() +
  theme(panel.grid = element_blank()) + 
  theme(legend.position = "none") + 
  theme( legend.text=element_text(size=15),
        strip.text.x = element_text(size = 15),
        axis.text.x = element_text(size = 13),
        axis.text.y = element_text(size = 13),
        axis.title = element_text(size = 14))



p3_same_rate 

ggsave(p3_same_rate , file = "./plots/diff_psi_same_rate_3_log_1.pdf", 
       width = 7, height = 7, dpi = 300, units = "in" )

############################################################


df_aggr_same_rate =  df_same_rate %>% group_by(n_idx,psi_idx) %>% 
  summarise(n = mean(n), s_n = mean(s_n), meanMSE = mean((diff_fhat_f)^2))

df_text_full = data.frame(x = rep(8,4), y = c(-4.9,-5.78,-4.7,-6.5), 
                          psi_idx = 1:4 ) 

for(psi_idx_choice in 1:4)
{
  y_mse = df_aggr_same_rate %>% filter(psi_idx == psi_idx_choice) %>% 
    .$meanMSE %>% sqrt() %>% log()
  x_mse = df_aggr_same_rate %>% filter(psi_idx == psi_idx_choice) %>%
    .$n %>% sqrt() %>% log()
  df_text_full$slope[psi_idx_choice] =
    paste("slope = ", coef(lm(y_mse ~ x_mse))[2] %>% 
                                               round(3), sep = "")
}


p1_same_rate = ggplot( data = df_aggr_same_rate) +  
  geom_line(aes(x = log(n), y = log((meanMSE)), group = psi_idx,
                color = as.character(psi_idx))) +
  coord_fixed(ratio = 1, xlim = NULL, ylim = NULL, expand = TRUE,
              clip = "on") + 
  labs(colour = expression(paste(psi," choice")) , 
       y = "log(MSE)", x = "log(n)")  +
  geom_text(data = df_text_full, aes(x = x, y = y, label = slope, 
                                     color = as.character(psi_idx)),
            show.legend = FALSE, size = 5) +
  theme_bw() +
  theme(legend.position = "bottom", panel.grid = element_blank()) + 
  theme( legend.text=element_text(size=14),
         legend.title =element_text(size=14),
         strip.text.x = element_text(size = 14),
         axis.text.x = element_text(size = 13),
         axis.text.y = element_text(size = 13),
         axis.title = element_text(size = 14))


p1_same_rate

ggsave(p1_same_rate, file = "./plots/diff_psi_same_rate_1_log_1.pdf",  
       width = 7, height = 6, dpi = 300, units = "in" )

############################################################
