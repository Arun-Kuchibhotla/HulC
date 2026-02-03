
#Change the following paths to your own local or remote working directories.
local=T
if (local){
  setwd("C:/Users/selin/Dropbox/Hulc_simulations/hulc_simulations/HulC_R")
} else{
  setwd("/home/shcarter/Hulc/simulations/R_sims")
}


source("additional_sims.R")


#It is not recommended to run the loop over all off the following vectors; rather, break them into pieces. For example, set method_vector = c("ai-sgd", "implicit") for a single session.
method_vector = c("ai-sgd", "implicit", "sgd",  "root-sgd", "truncated-sgd", "noisy-truncated-SGD")
N_vector = c(10**3, 10**4, 5*10**4, 10**5)
D_vector= c(5, 20, 100)
cov_type_vector = c("I", "EquiCorr", "Toeplitz")


for (method in method_vector){
  for (N in N_vector){
    for (D in D_vector){
      for (cov_type in cov_type_vector){
        run_sims(model_type = "OLS", S=200, N=N, D=D, 
                 method = method, cov_type = cov_type, alpha_level = .05, 
                 c_grid = c(0.01, 0.05, 0.1,  0.2,  0.5, 0.75, 1, 1.5, 2),
                 #c_grid =  0.1* c(0.01, 0.05, 0.1,  0.2,  0.5, 0.75, 1, 1.5, 2),
                 alpha_lr = .505,
                 epsilon = .8, sigma=1, beta=.25,
                 initializer = T, epochs_for_HulC = 1,
                 fixed_step = F, verbose=T, csv_name = "extra_sims.csv",
                 use_all_wald_estimates=F)
        
        
        
      }
    }
  }
}


