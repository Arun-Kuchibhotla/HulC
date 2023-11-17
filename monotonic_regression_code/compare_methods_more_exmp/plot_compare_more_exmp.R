######################################################################################
library(dplyr)
library(ggplot2)
library(ggpubr)

df_compare = read.csv("./df_compare_more_exmp_large_n_1.csv")[,-1] 

df_compare$method[df_compare$method=="hulc"] = "HulC" 
df_compare$method[df_compare$method=="dhz_data"] = "DHZ pivot (data-driven)"
df_compare$method[df_compare$method=="subsam"] = "Subsampling (1/2)"

p1_compare = df_compare %>% 
  ggplot(mapping = aes(x = n/100, y = coverage, color = method)) +
  geom_line(aes(group = method), alpha = 1, linewidth = 1) +
  geom_hline(yintercept = 0.95,linetype = 2) + 
  facet_grid(~ exmp_idx) + 
  labs(colour = "Method", y = "Coverage", x = "n (in the scale of 100)" ) + 
  theme_bw() +
  theme(legend.position = "bottom", legend.text=element_text(size=20)) + 
  theme(strip.text.x = element_text(size = 18),
        axis.text.x = element_text(size = 18),
        axis.text.y = element_text(size = 18)) + 
  scale_x_continuous(breaks=seq(5, 50, 15))

p2_compare = df_compare %>% 
  ggplot(mapping = aes(x = n/100, y = width, color = method)) +
  geom_line(aes(group = method), alpha = 1, linewidth = 1) +
  facet_grid(~ exmp_idx) + 
  labs(colour = "Method", y = "Width", x = "n (in the scale of 100)" ) + 
  theme_bw() +
  theme(legend.position = "bottom", legend.text=element_text(size=20)) + 
  theme(strip.text.x = element_text(size = 18),
        axis.text.x = element_text(size = 18),
        axis.text.y = element_text(size = 18)) + 
  scale_x_continuous(breaks=seq(5, 50, 15))


p21 = ggarrange(p1_compare + theme(text = element_text(size = 18)), 
                p2_compare + theme(text = element_text(size = 18)) , 
                ncol=1, nrow=2, common.legend = TRUE, legend="bottom")

p21 = annotate_figure(p21,
                      top = text_grob("Comparison of inference methods",
                                      size = 17))
p21


ggsave(p21 , file = "./plots/compare_more_exmp.pdf", 
       width = 13, height = 6, dpi = 300, units = "in" )
