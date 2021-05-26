source("auxiliary_functions.R")
## HulC1d() uses asymptotic median bias value to construct
## 		convex hull confidence interval for a univariate 
##		parameter. This is Algorithm 1 of the paper.
## data is a data frame.
## estimate is a function that takes a data frame as input 
## 		and returns a one-dimensional estimate.
## alpha is the level.
## Delta is the median bias of the estimate(). 
## randomize is a logical. If TRUE then the number of splits
## 		is randomized. If FALSE, then the larger number of
##		splits is used.
HulC1d <- function(data, estimate, alpha = 0.05, Delta = 0, randomize = TRUE){
	data <- as.matrix(data)
	nn <- nrow(data)
	data <- data[sample(nn),,drop=FALSE]
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
	TMP <- split(1:nn, sort((1:nn)%%B))
	for(idx in 1:B){
		ci_est[idx] <- estimate(data[TMP[[idx]],,drop=FALSE])
	}
	CI <- range(ci_est)
	names(CI) <- c("lwr", "upr")
	ret <- list(CI = CI, median.bias = Delta, B = B)
	return(ret)
}

## HulC() uses asymptotic median bias value to construct
## 		rectangular hull confidence region. This is Algorithm 1
## 		of the paper with union bound.
## data is a data frame.
## estimate is a function that takes a data frame as input 
## 		and returns a one-dimensional estimate. If multivariate,
##  	union will be used to obtain the confidence region.
## alpha is the level.
## Delta is the median bias of the estimate(). It can be a 
## 		vector. If a scalar is given, then it will repeated to
## 		form a vector of same length as dim.
## dim is the dimension of the output of estimate().
## randomize is a logical. If TRUE then the number of splits
## 		is randomized. If FALSE, then the larger number of
##		splits is used.
HulC <- function(data, estimate, alpha = 0.05, Delta = 0, dim = 1, randomize = TRUE){
	data <- as.matrix(data)
  if(length(Delta) == 1){
    Delta <- Delta*rep(1, dim)
  }
	CI <- matrix(0, nrow = dim, ncol = 2)
	B <- rep(0, dim)
	colnames(CI) <- c("lwr", "upr")
	for(idx in 1:dim){
		foo <- function(dat){
			estimate(dat)[idx]
		}
		tryCatch(
			hulc_idx <- HulC1d(data, foo, alpha = alpha/dim, Delta = Delta[idx], randomize = randomize),
      error = function(e){
        hulc_idx <- list(CI = c(NA, NA), B = NA)
      }
    )
		CI[idx,] <- hulc_idx$CI
		B[idx] <- hulc_idx$B
	}
	ret <- list(CI = CI, median.bias = Delta, B = B)
	return(ret)
}

## Adaptive_HulC() estimates the median bias of the estimator
##		and construct the rectangular confidence region. This is 
## 		Algorithm 2 of the paper with union bound.
## data is a data frame.
## estimate is a function that takes a data frame as input 
## 		and returns a one-dimensional estimate. If multivariate,
##  	union will be used to obtain the confidence region.
## alpha is the level.
## dim is the dimension of the output of estimate().
## subsamp_exp is the exponent of sample size 
## nsub is the number of subsamples.
## randomize is a logical. If TRUE then the number of splits
## 		is randomized. If FALSE, then the larger number of
##		splits is used.
adaptive_HulC <- function(data, estimate, alpha = 0.05, dim = 1, subsamp_exp = 2/3, nsub = 1000, randomize = TRUE){
	data <- as.matrix(data)
	CI <- matrix(0, nrow = dim, ncol = 2)
	colnames(CI) <- c("lwr", "upr")
	B <- rep(0, dim)
	Delta <- rep(0, dim)
	for(idx in 1:dim){
		foo <- function(dat){
			estimate(dat)[idx]
		}
		Delta[idx] <- subsamp_median_bias(data, foo, subsamp_exp = subsamp_exp, nsub = nsub)
		tryCatch(
			hulc_idx <- HulC1d(data, foo, alpha = alpha/dim, Delta = Delta[idx], randomize = randomize),
      error = function(e){
        hulc_idx <- list(CI = c(NA, NA), B = NA)
      }
    )
		CI[idx,] <- hulc_idx$CI
		B[idx] <- hulc_idx$B		
	}
	ret <- list(CI = CI, median.bias = Delta, B = B)
	return(ret)
}

## unimodal_HulC1d() uses asymptotic median bias and unimodality to
## 		construct an inflated convex hull confidence interval for a
##		univariate parameter. This is Algorithm 3 of the paper.
## data is a data frame.
## estimate is a function that takes a data frame as input and returns
## 		a one-dimensional estimate.
## alpha is the level.
## Delta is the asymptotic median bias of the estimate().
## t is the inflation factor.
## randomize is a logical. If TRUE then the number of splits
## 		is randomized. If FALSE, then the larger number of
##		splits is used.
unimodal_HulC1d <- function(data, estimate, alpha = 0.05, Delta = 1/2, t = 0.1, randomize = TRUE){
	data <- as.matrix(data)
	nn <- nrow(data)
	data <- data[sample(nn),,drop=FALSE]
	B1 <- solve_for_B(alpha = alpha, Delta = Delta, t = t)
	B <- B1
	if(randomize){
		p1 <- ((1/2 - Delta)^B1 + (1/2 + Delta)^B1)*(1 + t)^(-B1 + 1)
		B0 <- B1 - 1
		p0 <- ((1/2 - Delta)^B0 + (1/2 + Delta)^B0)*(1 + t)^(-B0 + 1)
		U <- runif(1)
		tau <- (alpha - p1)/(p0 - p1)
		B <- B0*(U <= tau)+ B1*(U > tau)
	}
	if(B > nn){
	  print(paste0("Delta = ", Delta, ", No. of splits = ", B, ", Sample size = ", nn))
	  stop("Error: not enough samples for splitting!")
	}
	ci_est <- rep(0, B)
	TMP <- split(1:nn, sort((1:nn)%%B))
	for(idx in 1:B){
		ci_est[idx] <- estimate(data[TMP[[idx]],,drop=FALSE])
	}
	CI <- range(ci_est)
	CI <- CI + t*diff(CI)*c(-1, 1)
	names(CI) <- c("lwr", "upr")
	ret <- list(CI = CI, median.bias = Delta, B = B)
	return(ret)
}

## unimodal_HulC() uses asymptotic median bias and unimodality to
## 		construct an inflated rectangular hull confidence region for a
##		multivariate parameter. This is Algorithm 3 of the paper with
##		union bound.
## data is a data frame.
## estimate is a function that takes a data frame as input and returns
## 		a one-dimensional estimate. If multivariate, union bound is used.
## alpha is the level.
## Delta is the asymptotic median bias of the estimate(). It is allowed
##		to be a vector. If a scalar is given, then it will repeated to
## 		form a vector of same length as dim.
## t is the inflation factor. 
## randomize is a logical. If TRUE then the number of splits
## 		is randomized. If FALSE, then the larger number of
##		splits is used.
unimodal_HulC <- function(data, estimate, alpha = 0.05, Delta = 1/2, t = 0.1, dim = 1, randomize = TRUE){
	data <- as.matrix(data)
	if(length(Delta) == 1){
	  Delta <- Delta*rep(1, dim)
	}
	CI <- matrix(0, nrow = dim, ncol = 2)
	colnames(CI) <- c("lwr", "upr")
	B <- rep(0, dim)
	for(idx in 1:dim){
		foo <- function(dat){
			estimate(dat)[idx]
		}
		tryCatch(
			hulc_idx <- unimodal_HulC1d(data, foo, alpha = alpha/dim, Delta = Delta[idx], t = t, randomize = randomize),      
			error = function(e){
        hulc_idx <- list(CI = c(NA, NA), B = NA)
      }
    )
		CI[idx,] <- hulc_idx$CI
		B[idx] <- hulc_idx$B
	}
	ret <- list(CI = CI, median.bias = Delta, B = B)
	return(ret)
}

## adaptive_unimodal_HulC() uses estimated median bias and unimodality to
## 		construct an inflated rectangular hull confidence region for a
##		multivariate parameter. This is Algorithm 3 of the paper with
##		subsample estimate of median bias and union bound.
## data is a data frame.
## estimate is a function that takes a data frame as input and returns
## 		a one-dimensional estimate. If multivariate, union bound is used.
## alpha is the level.
## t is the inflation factor.
## dim is the dimension of the output of estimate().
## subsamp_exp is the exponent of sample size 
## nsub is the number of subsamples.
## randomize is a logical. If TRUE then the number of splits
## 		is randomized. If FALSE, then the larger number of
##		splits is used.
adaptive_unimodal_HulC <- function(data, estimate, alpha = 0.05, t = 0.1, dim = 1, subsamp_exp = 2/3, nsub = 1000, randomize = TRUE){
	data <- as.matrix(data)
	CI <- matrix(0, nrow = dim, ncol = 2)
	colnames(CI) <- c("lwr", "upr")
	Delta <- rep(0, dim)
	B <- rep(0, dim)
	for(idx in 1:dim){
		foo <- function(dat){
			estimate(dat)[idx]
		}
		Delta[idx] <- subsamp_median_bias(data, foo, subsamp_exp = subsamp_exp, nsub = nsub)
		tryCatch(
			hulc_idx <- unimodal_HulC1d(data, foo, alpha = alpha/dim, Delta = Delta[idx], t = t, randomize = randomize),
      error = function(e){
        hulc_idx <- list(CI = c(NA, NA), B = NA)
      }
    )		
		CI[idx,] <- hulc_idx$CI
		B[idx] <- hulc_idx$B		
	}
	ret <- list(CI = CI, median.bias = Delta, B = B)
	return(ret)
}