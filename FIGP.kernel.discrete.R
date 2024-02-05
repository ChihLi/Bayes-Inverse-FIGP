FIGP.kernel.discrete <- function(theta, nu, G=NULL, gN=NULL, XN, kernel){
  
  n <- length(G)
  rnd <- nrow(XN)
  
  if(is.null(G)){
    if(kernel=="linear"){
      R <- sqrt(distance(t(t(XN)/theta)))
      Phi <- matern.kernel(R, nu=nu)
      a <- matrix(0,ncol=1,nrow=rnd)
      a[,1] <- gN
      K <- (t(a) %*% Phi %*% a) / rnd 
    }else{
      a <- matrix(0,ncol=1,nrow=rnd)
      a[,1] <- gN
      R <- sqrt(distance(t(a),t(a))/rnd)
      K <- matern.kernel(R/theta, nu=nu)
    }
  }else{
    A <- matrix(0,ncol=n,nrow=rnd)
    for(i in 1:n)  A[,i] <- apply(XN, 1, G[[i]])
    
    if(kernel=="linear"){
      R <- sqrt(distance(t(t(XN)/theta)))
      Phi <- matern.kernel(R, nu=nu)
      if(is.null(gN)){
        K <- (t(A) %*% Phi %*% A) / rnd # should be /rnd^2 but the values become too small, but it doesn't hurt without it because of scale parameter
        K <- (K+t(K))/2
      }else{
        a <- matrix(0,ncol=1,nrow=rnd)
        a[,1] <- gN
        K <- (t(a) %*% Phi %*% A) / rnd 
      }
    }else{
      if(is.null(gN)){
        R <- sqrt(distance(t(A))/rnd)
        K <- matern.kernel(R/theta, nu=nu)
      }else{
        a <- matrix(0,ncol=1,nrow=rnd)
        a[,1] <- gN
        R <- sqrt(distance(t(a),t(A))/rnd)
        K <- matern.kernel(R/theta, nu=nu)
      }
    }
  }
  
  return(K)
}
