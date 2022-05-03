source("auxiliary_functions.R")
### HulC from a given confidence interval

## ci_to_est_to_hulc function takes data (given data) as input,
## 		ci is a function input that takes data and miscoverage error as input and
## 			outputs an approximate confidence interval as a vector,
## 		alpha is a scalar input that represents the nomianl miscoverage error needed.	

ci_to_est_to_hulc <- function(data, ci, alpha = 0.05, gamma = alpha, randomize = TRUE){
	data <- as.matrix(data)
	nn <- nrow(data)
	data <- data[sample(nn),,drop=FALSE]
	Delta <- gamma/2
	B1 <- solve_for_B(alpha = alpha, Delta = Delta, t = 0)
	B <- B1
	if(randomize){
		p1 <- (1/2 + Delta)^B1 + (1/2 - Delta)^B1
		B0 <- B1 - 1
		p0 <- (1/2 + Delta)^B0 + (1/2 - Delta)^B0
		U <- runif(1)
		tau <- (alpha - p1)/(p0 - p1)
		B <- B0*(U <= tau)+ B1*(U > tau)
	}
	if(B > nn){
	  print(paste0("Delta = ", Delta, ", No. of splits = ", B, ", Sample size = ", nn))
		stop("Error: not enough samples for splitting!")
	}
	ci_est <- rep(0, B)
	ci_list <- vector("list", B)
	TMP <- split(1:nn, sort((1:nn)%%B))
	for(idx in 1:B){
		ci_list[[idx]] <- ci(data[TMP[[idx]],,drop=FALSE], gamma)
		tmp_runif <- runif(1)
		ci_est[idx] <- ci_list[[idx]][1]*(tmp_runif < 1/2) + ci_list[[idx]][2]*(tmp_runif >= 1/2)
	}
	CI <- range(ci_est)
	names(CI) <- c("lwr", "upr")
	ret <- list(CI = CI, given_cis = ci_list, B = B)
	return(ret)	
}

ci_to_hulc <- function(data, ci, alpha = 0.05, gamma = alpha, randomize = TRUE){
	data <- as.matrix(data)
	nn <- nrow(data)
	data <- data[sample(nn),,drop=FALSE]
	B1 <- ceiling(log(alpha)/log(gamma))
	B <- B1
	if(randomize){
		p1 <- gamma^B1
		B0 <- B1 - 1
		p0 <- gamma^B0
		U <- runif(1)
		tau <- (alpha - p1)/(p0 - p1)
		B <- B0*(U <= tau)+ B1*(U > tau)
	}
	if(gamma < alpha){
		B <- 1
	}
	if(B > nn){
	  print(paste0("Delta = ", Delta, ", No. of splits = ", B, ", Sample size = ", nn))
		stop("Error: not enough samples for splitting!")
	}
	ci_est <- NULL
	ci_list <- vector("list", B)
	TMP <- split(1:nn, sort((1:nn)%%B))
	for(idx in 1:B){
		ci_list[[idx]] <- ci(data[TMP[[idx]],,drop=FALSE], gamma)
		ci_est <- c(ci_est, ci_list[[idx]])
	}
	CI <- range(ci_est)
	names(CI) <- c("lwr", "upr")
	ret <- list(CI = CI, given_cis = ci_list, B = B)
	return(ret)		
}

# ### Example using Wald intervals
# ci <- function(x, gamma = 0.05){
# 	a <- mean(x)
# 	b <- sd(x)/sqrt(length(x))
# 	quant <- -qnorm(gamma/2)
# 	return(c(a - quant*b, a + quant*b))
# }

# x <- rnorm(1000)
# ci_to_est_to_hulc(x, ci, alpha = 0.05, gamma = alpha)
# ci_to_est_to_hulc(x, ci, alpha = 0.05, gamma = 1/sqrt(length(x)))
# ### Comparing to the wald interval on full data
# ci(x, gamma = 0.05)

# ci_to_hulc(x, ci, alpha = 0.05, gamma = 2*alpha)
# ci_to_hulc(x, ci, alpha = 0.05, gamma = 10*alpha)
# ### Comparing to the wald interval on full data
# ci(x, gamma = 0.05)