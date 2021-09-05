## data is a data frame.
## estimate is a function that takes a data frame as input 
## 	and returns a one-dimensional estimate. If multi-dimensional
## 	a union bound is used to construct the confidence region.
## alpha is the level.
## asymmetry is how far off are the probabilities from 1/2.
## asymmetry = 0 means the estimator is symmetric around the target.
convex_hull <- function(data, estimate, alpha = 0.05, asymmetry = 0, dim = 1){
  data <- as.matrix(data)
	nn <- nrow(data)
	data <- data[sample(nn),,drop=FALSE]
	B1 <- solve_for_B(alpha = alpha, asymmetry = asymmetry, dim = dim)
	p1 <- (1/2 + asymmetry)^B1 + (1/2 - asymmetry)^B1
	B0 <- B1 - 1
	p0 <- (1/2 + asymmetry)^B0 + (1/2 - asymmetry)^B0
	U <- runif(1)
	tau <- (alpha - p1)/(p0 - p1)
	B <- B0*(U <= tau)+ B1*(U > tau)
	if(B > nn){
		stop("Error: not enough samples for splitting!")
	}	
	ci_est <- matrix(0, nrow = B, ncol = dim)
	TMP <- split(1:nn, sort((1:nn)%%B))
	for(idx in 1:B){
		ci_est[idx,] <- estimate(data[TMP[[idx]],,drop=FALSE])
	}
	CI <- cbind(apply(ci_est, 2, min), apply(ci_est, 2, max))
	return(CI)
}

solve_for_B <- function(alpha = 0.05, asymmetry = 0, dim = 1){
	B_low <- floor(log(2*dim/alpha, base = 2))
	B_up <- ceiling(log(2*dim/alpha, base = 2/(1 + 2*asymmetry)))
	for(B in B_low:B_up){
		if((1/2 + asymmetry)^B + (1/2 - asymmetry)^B <= alpha)
			break
	}
	return(B)
}

## For the following function, asymmetry can be a vector.
## If one number is given for asymmetry, then it will be made
## into a vector with all entries the same.
convex_hull_subsamp <- function(data, estimate, alpha = 0.05, asymmetry = NULL, dim = 1, subsamp = TRUE, subsamp_exp = 2/3, nsub = 1000){
	## If neither asymmetry nor subsampling is given, then
	## through an error.
	if(is.null(asymmetry) && isFALSE(subsamp)){
		stop("Input Error: either 'asymmetry' should be non-null\\ or 'subsamp' should be TRUE!")
	}
	## If both asymmetry and subsampling are given, then
	## asymmetry is used and subsampling is set FALSE.
	if(!is.null(asymmetry) && isTRUE(subsamp)){
		subsamp <- FALSE
		if(!is.vector(asymmetry)){
			asymmetry <- asymmetry*rep(1, dim)
		}
	}
	data <- as.matrix(data)
	nn <- nrow(data)
	data <- data[sample(nn),,drop=FALSE]
	CI <- matrix(0, nrow = dim, ncol = 2)
	if(isTRUE(subsamp)){
		subsamp_size <- round(nn^subsamp_exp)
		nsub <- min(nsub, choose(nn, subsamp_size))
		fulldata_estimate <- estimate(data)
		asymmetry <- rep(0, dim)
		for(b in 1:nsub){
			TMP <- estimate(data[sample(nn, subsamp_size, replace = FALSE),])
			asymmetry <- asymmetry + (TMP - fulldata_estimate <= 0)/nsub
		}
		asymmetry <- abs(asymmetry - 1/2)
	}
	# print(asymmetry)
	for(idx in 1:dim){
		foo <- function(dat){ estimate(dat)[idx] }
		CI[idx,] <- convex_hull(data, foo, alpha = 0.05/dim, asymmetry = asymmetry[idx], dim = 1)
	}
	return(list(ci = CI, asymmetry = asymmetry))
}

## data is a data frame.
## estimate is a function that takes a data frame as input 
## 	and returns a one-dimensional estimate. If multi-dimensional
## 	a union bound is used to construct the confidence region.
## alpha is the level.
## symmetry is either TRUE or FALSE. There is no relaxation.
## dim is the dimension of tha estimator.
## t is the constant by which the inflation of convex hull by range happens.
unimodal_conf <- function(data, estimate, alpha = 0.05, symmetry = FALSE, dim = 1, t = 0.5){
	data <- as.matrix(data)
	nn <- nrow(data)
	data <- data[sample(nn),,drop=FALSE]
	c <- symmetry + 1
	B1 <- ceiling(log(dim/alpha)/log(c + c*t) + 1)
	p1 <- 1/(c + c*t)^{-B1 + 1}
	B0 <- B1 - 1
	p0 <- 1/(c + c*t)^{-B0 + 1}
	U <- runif(1)
	tau <- (alpha - p1)/(p0 - p1)
	B <- B0*(U <= tau) + B1*(U > tau)
	ci_est <- matrix(0, nrow = B, ncol = dim)
	TMP <- split(1:nn, sort((1:nn)%%B))
	for(idx in 1:B){
		ci_est[idx,] <- estimate(data[TMP[[idx]],])
	}
	ci_max <- apply(ci_est, 2, max)
	ci_min <- apply(ci_est, 2, min)
	CI <- cbind(ci_min - t*(ci_max - ci_min), ci_max + t*(ci_max - ci_min))
	return(CI)
}

## Example CI for N(0, 1) problem
# convex_hull(rnorm(100), mean)
# convex_hull_subsamp(rnorm(100), mean)
# unimodal_conf(rnorm(100), mean, symmetry = TRUE)

## Example experiment for N(0, 1) problem
# nrep <- 200
# nsamp <- 100
# intervals_hull <- intervals_subsamp <- intervals_mode <- matrix(0, nrow = nrep, ncol = 2)
# for(idx in 1:nrep){
#   intervals_hull[idx,] <- convex_hull(rnorm(nsamp), mean)
#   intervals_subsamp[idx,] <- convex_hull_subsamp(rnorm(nsamp), mean)$ci
#   intervals_mode[idx,] <- unimodal_conf(rnorm(nsamp), mean, symmetry = TRUE)
# }
# mean(intervals_hull[,1]*intervals_hull[,2] <= 0)
# mean(intervals_subsamp[,1]*intervals_subsamp[,2] <= 0)
# mean(intervals_mode[,1]*intervals_mode[,2] <= 0)

### Example of monotone regression
# par(oma = c(1.1, 1, 0, 0))
# par(mfrow = c(2,2), mgp=c(2,0.5,0), mar = c(4, 5, 4, 4), mai = c(1, 0.4, 0.3, 0.1))
# nsamp_vec <- c(250, 500, 750, 1000)
# for(nsamp in nsamp_vec){
#   f0 <- function(x){
#     -1*(x<=.5)+ ((x-.5)/.5)^2*(x>.5)+1
#   }
#   x <- sort(runif(nsamp))
#   y <- f0(x) + rnorm(nsamp, 0, 0.1)
#   ngrid <- 25
#   grid <- seq(nsamp^(-1/2), 1 - nsamp^(-1/2), length = ngrid)
#   estimate <- function(data){
#     ## data has first column x, second column y
#     tmp <- isoreg(data[,1], data[,2])
#     foo <- approxfun(sort(tmp$x), tmp$yf, method="constant", rule = 2)
#     return(foo(grid))
#   }
#   # data <- cbind(x, y)
#   # alpha <- 0.05
#   # symmetry <- FALSE
#   # dim <- ngrid
#   # t <- 0.5
#   CI_mode <- unimodal_conf(cbind(x, y), estimate, alpha = 0.05, dim = ngrid, t = 0.5)
#   # TMP <- convex_hull_subsamp(cbind(x, y), estimate, alpha = 0.05, dim = ngrid)
#   # CI_subsamp <- TMP$ci
#   plot(x, y, pch = '.', ylim = range(CI_mode), col = "gray")
#   lines(x, f0(x), col = "red", lwd = 2)
#   grid_new <- c(0, grid, 1)
#   ci_low <- cummax(c(-1e05, CI_mode[,1], 1e05))
#   ci_up <- rev(cummin(rev(c(-1e05, CI_mode[,2], 1e05))))
#   lines(grid_new, ci_low, lty = 2, col = "blue")
#   lines(grid_new, ci_up, lty = 2, col = "blue")
#   points(grid, CI_mode[,1], pch = "*", col = "blue")
#   points(grid, CI_mode[,2], pch = "*", col = "blue")
#   title(paste("Sample size", nsamp, "Mode CI"))
#   
#   # plot(x, y, pch = '.', ylim = range(CI_subsamp), col = "gray")
#   # lines(x, f0(x), col = "red", lwd = 2)
#   # grid_new <- c(0, grid, 1)
#   # ci_low <- cummax(c(-1e05, CI_subsamp[,1], 1e05))
#   # ci_up <- rev(cummin(rev(c(-1e05, CI_subsamp[,2], 1e05))))
#   # lines(grid_new, ci_low, lty = 2, col = "blue")
#   # lines(grid_new, ci_up, lty = 2, col = "blue")
#   # points(grid, CI_subsamp[,1], pch = "*", col = "blue")
#   # points(grid, CI_subsamp[,2], pch = "*", col = "blue")
#   # title(paste("Sample size", nsamp, "Subsamp CI"))
# }
