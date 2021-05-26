## For non-negative Delta and t, set
## 	Q(B; Delta, t) = [(1/2 - Delta)^B + (1/2 + Delta)^B]*(1 + t)^{-B+1}
## The following function finds the smallest B for a given t such that
##	Q(B; Delta, t) <= alpha.
solve_for_B <- function(alpha, Delta, t){
  if(Delta == 0.5 && t == 0){
    stop("Delta is 0.5 and t = 0. The estimator lies only on one side of the parameter!")
  }
	B_low <- max(floor(log((2 + 2*t)/alpha, base = 2 + 2*t)), floor(log((1 + t)/alpha, base = (2 + 2*t)/(1 + 2*Delta))))
	B_up <- ceiling(log((2 + 2*t)/alpha, base = (2 + 2*t)/(1 + 2*Delta)))
	Q <- function(B){
		((1/2 - Delta)^B + (1/2 + Delta)^B)*(1 + t)^(-B + 1)
	}
	for(B in B_low:B_up){
		if(Q(B) <= alpha)
			break
	}
	return(B)
}

## For any estimation function estimate() that returns a
## univariate estimator, subsamp_median_bias() provides an
## estimate of the median bias using subsampling.
## The subsample size used is (sample size)^{subsamp_exp}.
## The input data is a data frame or a matrix.
## nsub is the number of subsamples
subsamp_median_bias <- function(data, estimate, subsamp_exp = 2/3, nsub = 1000){
	data <- as.matrix(data)
	nn <- nrow(data)
	subsamp_size <- round(nn^subsamp_exp)
	nsub <- min(nsub, choose(nn, subsamp_size))
	fulldata_estimate <- estimate(data)
	Delta <- 0
	for(b in 1:nsub){
		TMP <- estimate(data[sample(nn, subsamp_size, replace = FALSE),,drop=FALSE])
		Delta <- Delta + (TMP - fulldata_estimate <= 0)/nsub
	}
	Delta <- abs(Delta - 1/2)
	return(Delta)
}