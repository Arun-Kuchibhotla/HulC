
packages_to_check <- c("mvtnorm")
installed_pkgs <- rownames(installed.packages())

for (pkg in packages_to_check) {
  if (!(pkg %in% installed_pkgs)) {
    message(paste("Package", pkg, "not found. Installing..."))
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }  else {
    library(pkg, character.only = TRUE)
  }
}


gen_normal0_data = function(n=1000, d=5, start=0, stop=1, cov_type="I"){
  
  # This function returns an n-sized sample (X,y): 1 column of 1's (for intercept) and
  #   d-1 covariates (X matrix~ n x d) and 1 outcome (y),
  # where y is a linear function of x plus an error term. 
  #According to the Chen et al paper (2016), the linear regression parameters are equally spaced
  # between 0 and 1, that is: theta = seq(start, stop, num=d)
  
  # cov_type="I" means x's are drawn independently from N(0,1) distribution.
  # cov_type="Toeplitz" means Sigma(i,j) = 0.5^{|i-j|} (using Chen simulation as reference)
  # cov_type = "EquiCorr" means Sigma(i,j) = 0.2 for i != j, and 1 if i = j
  
  if (cov_type=="I"){
    #Create X matrix
    X = matrix(rnorm(n*d, mean=0, sd = 1),nrow=n)
    X[,1] = 1
  }
  
  if (cov_type=="Toeplitz"){
    Sigma = matrix(0,nrow=d-1, ncol=d-1)
    r= 0.5
    for (row in 1:(d-1)){
      for (col in 1:(d-1)){
        Sigma[row,col] = r**abs(row-col)
        }
    }
    X = mvtnorm::rmvnorm(n=n, mean=rep(0, nrow(Sigma)), sigma=Sigma)
    X = cbind(rep(1,nrow(X)), X)
  }
  
  if (cov_type=="EquiCorr"){
    r = 0.2
    Sigma = matrix(r, nrow=d-1, ncol=d-1)
    diag(Sigma) = 1
    X = mvtnorm::rmvnorm(n=n, mean=rep(0, nrow(Sigma)), sigma=Sigma)
    X = cbind(rep(1,nrow(X)), X)
  }

  #Create the theta vector
  theta = seq(from=start, to=stop, length=d)
  
  #Generate Y
  Y = X%*%theta + rnorm(n=n, mean=0, sd=1)
  
  plot_it = F
  if (plot_it){
   # Create a scatter plot
    plot(X[,2], Y, xlab = "X_2")
  }
  
  
  out = data.frame(Y, X)
  
  return(out)
  
  
}




sigmoid = function(s){
  #s is a scalar
  denom = 1 + exp(-s)
  
  return(1/denom)
}



gen_normal0_logistic_data = function(n, d, start=0, stop=1, ytype="neg11", cov_type="I"){
  
  # This function returns an n-sized sample (X,y): 1 column of 1's (for intercept) and
  #   d-1 covariates (X matrix~ n x d) and 1 outcome (y),
  # where y is a linear function of x plus an error term. 
  #According to the Chen et al paper (2016), the linear regression parameters are equally spaced
  # between 0 and 1, that is: theta = seq(start, stop, num=d)
  # # ytype = "neg11" means y_i is binary {-1, 1}. type = "01" means y_i is binary {0, 1} 
  #
  # cov_type="I" means x's are drawn independently from N(0,1) distribution.
  # cov_type="Toeplitz" means Sigma(i,j) = 0.5^{|i-j|} (using Chen simulation as reference)
  # cov_type = "EquiCorr" means Sigma(i,j) = 0.2 for i != j, and 1 if i = j
  
  if (cov_type=="I"){
    #Create X matrix
    X = matrix(rnorm(n*d, mean=0, sd = 1),nrow=n)
    X[,1] = 1
  }
  
  if (cov_type=="Toeplitz"){
    Sigma = matrix(0,nrow=d-1, ncol=d-1)
    r= 0.5
    for (row in 1:(d-1)){
      for (col in 1:(d-1)){
        Sigma[row,col] = r**abs(row-col)
      }
    }
    X = mvtnorm::rmvnorm(n=n, mean=rep(0, nrow(Sigma)), sigma=Sigma)
    X = cbind(rep(1,nrow(X)), X)
  }
  
  if (cov_type=="EquiCorr"){
    r = 0.2
    Sigma = matrix(r, nrow=d-1, ncol=d-1)
    diag(Sigma) = 1
    X = mvtnorm::rmvnorm(n=n, mean=rep(0, nrow(Sigma)), sigma=Sigma)
    X = cbind(rep(1,nrow(X)), X)
  }
  
  
  #Create the theta vector
  theta = seq(from=start, to=stop, length=d)
  
  #Generate Y 
  mu = sigmoid(X%*%theta)
  Y = rbinom(n=n, size=1, prob=mu)
  if (ytype=="neg11"){
    Y[Y==0] = -1
  }

  
  plot_it = F
  if (plot_it){
    # Create a scatter plot
    plot(X[,2], Y, xlab = "X_2", main = 'Scatter Plot (Logistic Regression)')
  }


  out = data.frame(Y, X)
  
  return(out)

}



