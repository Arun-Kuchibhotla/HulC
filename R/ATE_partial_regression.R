library(mvtnorm)
library(grf)
library(sandwich)

Forest = function(X,Y,ntrees=1000,Xtarget){
	### fit Y = mu(X,A)
	### train on (X,Y); get fitted values at Xtarget

	X = as.matrix(X)
	Xtarget = as.matrix(Xtarget)
	tmp = regression_forest(X,Y,honesty = FALSE,num.trees=ntrees)
	out = predict(tmp,Xtarget)[,1]
	names(out) = NULL
	return(out)
}

## dat is a matrix with first column the response (Y),
## 	second column the treatment (A),
## 	the remaining columns the covariates (A).
ATE <- function(dat, ntrees = 1000, split = 1/2){
	dat <- as.matrix(dat)
	nn <- nrow(dat)
	I1 <- sample(nn, round(nn*split))
	dat1 <- dat[I1,]
	dat2 <- dat[-I1,]
	
	nn2 <- nrow(dat2)
    mu <- Forest(dat1[,-c(1,2)],dat1[,1],ntrees=ntrees,Xtarget=dat2[,-c(1,2)])  ## regress Y on X
    nu <- Forest(dat1[,-c(1,2)],dat1[,2],ntrees=ntrees,Xtarget=dat2[,-c(1,2)])  ## regress A on X
    resY1 <- dat2[,1] - mu
    resA1 <- dat2[,2] - nu

	nn1 <- nrow(dat1)
    mu <- Forest(dat2[,-c(1,2)],dat2[,1],ntrees=ntrees,Xtarget=dat1[,-c(1,2)])  ## regress Y on X
    nu <- Forest(dat2[,-c(1,2)],dat2[,2],ntrees=ntrees,Xtarget=dat1[,-c(1,2)])  ## regress A on X
    resY2 <- dat1[,1] - mu
    resA2 <- dat1[,2] - nu

    tmp1 <- lm(resY1 ~ 0 + resA1)
    psi1 <- tmp1$coef	
    se1 <- sqrt(vcovHC(tmp1, type = "HC")[1,1])
    tmp2 <- lm(resY2 ~ 0 + resA2)
    psi2 <- tmp2$coef
    se2 <- sqrt(vcovHC(tmp2, type = "HC")[1,1])
    return(list(psi = (nn2*psi1 + nn1*psi2)/(nn1 + nn2), se = sqrt(nn2*nn2*se1*se1 + nn1*nn1*se2*se2)/(nn1 + nn2)))
}

ATE_est <- function(dat, ntrees = 1000, split = 1/2){
	return(ATE(dat, ntrees = ntrees, split = split)$psi)
}