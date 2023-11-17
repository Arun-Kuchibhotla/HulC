######################################################################################
library(dplyr)
library(RColorBrewer)
library(ggplot2)

df_same_psi = read.csv("./df_same_psi_log.csv")[,-1]
df_aggr_same_psi =  df_same_psi %>% 
  group_by(n_idx,s_n_idx) %>% 
  summarise(n = mean(n), s_n = mean(s_n), meanMSE = mean((diff_fhat_f)^2))
df_aggr_same_psi = df_aggr_same_psi %>% 
  mutate(s_n_power = paste(s_n_idx,"/6",sep = ""))


########################################################################

df_text_full = data.frame(x = rep(8.5,5),
                          y = c(-3,-2.25,-1.6,-0.75,0), 
                          s_n_idx = 1:5 ) %>%
  mutate(slope = NA) %>% 
  mutate(s_n_power = paste(s_n_idx,"/6",sep = ""))

for(s_n_idx_choice in 1:5)
{
  y_mse = df_aggr_same_psi %>% filter(s_n_idx == s_n_idx_choice) %>% 
    .$meanMSE %>% sqrt() %>% log()
  x_mse = df_aggr_same_psi %>% filter(s_n_idx == s_n_idx_choice) %>% 
    .$n %>% sqrt() %>% log()
  df_text_full$slope[s_n_idx_choice] = 
    paste("slope = ", coef(lm(y_mse ~ x_mse))[2] %>% round(3), sep = "")
}

p1_same_psi = ggplot( data = df_aggr_same_psi, aes(x = log(n), y = log(sqrt(meanMSE)), color = s_n_power)) +   geom_line(aes(group = s_n_idx)) +
  labs(colour = expression(alpha) , y = "log (MSE)", x = "log(n)")  + 
  scale_color_hue(h = c(180, 300))+
  geom_text(data = df_text_full, aes(x = x, y = y, label = slope), show.legend =  F, size = 5) + 
  theme_bw() +
  theme(panel.grid = element_blank())+ theme(legend.position = "bottom")  + 
  theme( legend.text=element_text(size=14),
         legend.title =element_text(size=14),
         strip.text.x = element_text(size = 14),
         axis.text.x = element_text(size = 13),
         axis.text.y = element_text(size = 13),
         axis.title = element_text(size = 14))



p1_same_psi

ggsave(p1_same_psi , file = "./plots/diff_rates_same_psi_1.pdf", 
       width = 7, height = 6, dpi = 300, units = "in" )


p2_same_psi = ggplot( data = df_aggr_same_psi, aes(x = log(n/s_n), y = log(sqrt(meanMSE)), color = s_n_power)) +   geom_line(aes(group = s_n_idx)) +
  labs(colour = expression(alpha) , y = "log (MSE)", x =expression('log(n/s'[n]*')'))  +
  scale_color_hue(h = c(180, 300)) + 
  theme_bw() +
  theme(panel.grid = element_blank()) + theme(legend.position = "bottom") + 
  theme( legend.text=element_text(size=14),
         legend.title =element_text(size=14),
         strip.text.x = element_text(size = 14),
         axis.text.x = element_text(size = 13),
         axis.text.y = element_text(size = 13),
         axis.title = element_text(size = 14))


p2_same_psi

ggsave(p2_same_psi , file = "../plots/diff_rates_same_psi_2.pdf", 
       width = 7, height = 6, dpi = 300, units = "in" )


p12_same_psi = ggarrange(p1_same_psi, p2_same_psi, 
                         ncol=2, nrow=1, common.legend = TRUE, legend="bottom")
p12_same_psi

ggsave(p12_same_psi , file = "./plots/diff_rates_same_psi_12.pdf",
       width = 10, height = 5, dpi = 300, units = "in" )

#######

n_idx_choice = 11
df_same_psi_11 = df_same_psi %>% filter(n_idx == n_idx_choice) %>%
  mutate(f_hat_scaled = diff_fhat_f*sqrt(n/s_n)*0.5) %>%
  mutate(f_hat_scaled_sort = NA, brownian_drift_sort = NA)

for(s_n_idx_choice in 1:5 )
{
  temp_bd = df_same_psi_11 %>% filter( s_n_idx == 1) %>% 
    .$brownian_drift %>% sort()
  temp_f_hat = df_same_psi_11 %>% filter( s_n_idx == s_n_idx_choice) %>% 
    .$f_hat_scaled %>% sort()
  df_same_psi_11$brownian_drift_sort[df_same_psi_11$s_n_idx == s_n_idx_choice] = temp_bd
  df_same_psi_11$f_hat_scaled_sort[df_same_psi_11$s_n_idx == s_n_idx_choice] = temp_f_hat
}

df_same_psi_11 = df_same_psi_11 %>% mutate(s_n_power = paste(s_n_idx,"/6",sep = ""))

p3_same_psi = ggplot( data = df_same_psi_11, aes(x = brownian_drift_sort, y = f_hat_scaled_sort, color = s_n_power)) +  
  geom_line(aes(group = s_n_idx)) +
  geom_abline( slope = 1, intercept = 0, linetype="dashed", size=0.4) + 
  labs(colour = expression(alpha) , y = "Scaled estimates of f(0)", x ="Samples from asymptotic distribution")  +
  scale_color_hue(h = c(180, 300)) + 
  theme_bw() +
  theme(panel.grid = element_blank()) + theme(legend.position = "bottom") + 
  theme( legend.text=element_text(size=14),
         legend.title =element_text(size=14),
         strip.text.x = element_text(size = 14),
         axis.text.x = element_text(size = 13),
         axis.text.y = element_text(size = 13),
         axis.title = element_text(size = 14))


p3_same_psi 

ggsave(p3_same_psi , file = "./plots/diff_rates_same_psi_3.pdf",  
       width = 5, height = 5, dpi = 300, units = "in" )



