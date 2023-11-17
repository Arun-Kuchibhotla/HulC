######################################################################################
library(dplyr)
library(ggplot2)
library(ggpubr)

df_compare = read.csv("./df_compare.csv")[,-1] 

df_compare$method[df_compare$method=="hulc"] = "HulC" 
df_compare$method[df_compare$method=="dhz_data"] = "DHZ pivot (data-driven)"
df_compare$method[df_compare$method=="dhz_orac"] = "DHZ pivot (oracle)"
df_compare$method[df_compare$method=="subsam"] = "Subsampling (1/2)"



p1_compare = df_compare %>% 
  ggplot(mapping = aes(x = theta, y = coverage, color = method)) +
  geom_line(aes(group = method), alpha = 1, linewidth = 1.2) +
  geom_hline(yintercept = 0.95,linetype = 2) + 
  facet_grid(~ n) + labs(colour = "Method", y = "Coverage", 
                         x = expression(theta) ) + 
  theme_bw() +
  theme(legend.position = "bottom", legend.text=element_text(size=15))  + 
  theme(strip.text.x = element_text(size = 15),
        axis.text.x = element_text(size = 11),
        axis.text.y = element_text(size = 11))

p2_compare = df_compare %>% 
  ggplot(mapping = aes(x = theta, y = width, color = method)) +
  geom_line(aes(group = method), alpha = 1, linewidth = 1.2) +
  facet_grid(~ n) + 
  labs(colour = "Method", y = "Width", x = expression(theta) ) + 
  theme_bw() +
  theme(legend.position = "bottom", legend.text=element_text(size=15)) + 
  theme(strip.text.x = element_text(size = 15),
        axis.text.x = element_text(size = 11),
        axis.text.y = element_text(size = 11))


p21 = ggarrange(p1_compare + theme(text = element_text(size = 14)), 
                p2_compare + theme(text = element_text(size = 14)) , 
                ncol=1, nrow=2, common.legend = TRUE, legend="bottom")

p21


ggsave(p21 , file = "./plots/compare.pdf",  
       width = 11, height = 6, dpi = 300, units = "in" )
