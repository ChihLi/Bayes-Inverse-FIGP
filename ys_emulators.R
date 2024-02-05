##### predict the scores using FIGP model with the discrete input XN
pred.FIGP.XN <- function(fit, gN, XN){
  
  Ki <- fit$Ki
  theta <- fit$theta
  nu <- fit$nu
  nug <- fit$nug
  G <- fit$G
  y <- fit$y
  d <- fit$d
  mu.hat <- fit$mu.hat
  kernel <- fit$kernel
  
  n <- length(y)
  
  rnd <- nrow(XN)
  A <- matrix(0,ncol=n,nrow=rnd)
  for(i in 1:n)  A[,i] <- apply(XN, 1, G[[i]])
  
  K <- FIGP.kernel.discrete(theta, nu, G, gN=NULL, XN, kernel)
  Ki <- solve(K+diag(nug,n))
  KX <- FIGP.kernel.discrete(theta, nu, G, gN=gN, XN, kernel)
  KXX <- FIGP.kernel.discrete(theta, nu, G=NULL, gN=gN, XN, kernel) 
  
  # if(kernel == "linear"){
  #   ## compute linear kernel
  #   R <- sqrt(distance(t(t(XN)/theta)))
  #   Phi <- matern.kernel(R, nu=nu)
  #   K <- (t(A) %*% Phi %*% A) / rnd # should be /rnd^2 but the values become too small, but it doesn't hurt without it because of scale parameter
  #   K <- (K+t(K))/2
  #   Ki <- solve(K+diag(nug,n))
  #   
  #   a <- matrix(0,ncol=1,nrow=rnd)
  #   a[,1] <- gN
  #   KX <- (t(a) %*% Phi %*% A) / rnd 
  #   KXX <- (t(a) %*% Phi %*% a) / rnd 
  # }else{
  #   ## compute nonlinear kernel
  #   R <- sqrt(distance(t(A))/rnd)
  #   K <- matern.kernel(R/theta, nu=nu)
  #   Ki <- solve(K+diag(nug,n))
  #   
  #   a <- matrix(0,ncol=1,nrow=rnd)
  #   a[,1] <- gN
  #   R <- sqrt(distance(t(a),t(A))/rnd)
  #   KX <- matern.kernel(R/theta, nu=nu)
  #   R <- sqrt(distance(t(a),t(a))/rnd)
  #   KXX <- matern.kernel(R/theta, nu=nu)
  # }
  
  mup2 <- drop(mu.hat + KX %*% Ki %*% (y - mu.hat))
  tau2hat <- drop(t(y - mu.hat) %*% Ki %*% (y - mu.hat) / n)
  Sigmap2 <- pmax(0,diag(tau2hat*(KXX + nug - KX %*% Ki %*% t(KX))))
  
  return(list(mu=mup2, sig2=Sigmap2))
}


##### Multi-fidelity model
MF.FIGP <- function(G, d, y1, y2, nu, nug, 
                     kernel=c("linear", "nonlinear")[1],
                     theta.init=ifelse(kernel=="linear", 0.01, 1),
                     theta.lower=ifelse(kernel=="linear", 1e-6, 1e-2),
                     theta.upper=ifelse(kernel=="linear", 0.1, 100),
                     rho.init=0.5,
                     rho.lower=0,
                     rho.upper=2){
  
  n <- length(y1)
  nlsep <- function(par, G, d, Y1, Y2) 
  {
    theta <- par[1:(length(par)-1)]
    rho <- par[length(par)]
    Y <- Y1 - rho*Y2
    
    n <- length(Y1)
    
    K <- FIGP.kernel(d, theta, nu, G, kernel=kernel)
    Ki <- solve(K+diag(nug,n))
    ldetK <- determinant(K+diag(nug,n), logarithm=TRUE)$modulus
    
    one.vec <- matrix(1,ncol=1,nrow=n)
    mu.hat <- drop((t(one.vec)%*%Ki%*%Y)/(t(one.vec)%*%Ki%*%one.vec))
    
    tau2hat <- drop(t(Y - mu.hat) %*% Ki %*% (Y - mu.hat) / n)
    ll <- - (n/2)*(tau2hat) - (1/2)*ldetK
    return(-ll)
  }
  
  tic <- proc.time()[3]
  
  if(kernel=="linear"){
    out <- optim(c(rep(theta.init, d), rho.init), nlsep,
                 method="L-BFGS-B", lower=c(rep(theta.lower, d), rho.lower), upper=c(rep(theta.upper, d), rho.upper), G=G, Y1=y1, Y2=y2, d=d)
    theta <- out$par[1:d]
    rho <- out$par[d+1]
  }else{
    out <- optim(c(theta.init, rho.init), nlsep,
                 method="L-BFGS-B", lower=c(theta.lower, rho.lower), upper=c(theta.upper, rho.upper), G=G, Y1=y1, Y2=y2, d=d)
    theta <- out$par[1]
    rho <- out$par[2]
  }

  ll <- out$value
  
  toc <- proc.time()[3]
  
  y <- y1 - rho*y2
  K <- FIGP.kernel(d, theta, nu, G, kernel=kernel)
  Ki <- solve(K+diag(nug,n))
  one.vec <- matrix(1,ncol=1,nrow=n)
  mu.hat <- drop((t(one.vec)%*%Ki%*%y)/(t(one.vec)%*%Ki%*%one.vec))
  
  return(list(theta = theta, rho=rho, nu=nu, Ki=Ki, d=d, kernel=kernel, ElapsedTime=toc-tic, 
              nug = nug, G = G, y = y, mu.hat = mu.hat, ll=ll))
}

##### predict with linear kernel using discrete g #####
pred.MF <- function(fit1, fit2, gN, XN){
  
  # info of fit1
  Ki.1 <- fit1$Ki
  theta.1 <- fit1$theta
  nu <- fit1$nu
  nug <- fit1$nug
  G <- fit1$G
  y.1 <- fit1$y
  d <- fit1$d
  mu.hat.1 <- fit1$mu.hat
  kernel.1 <- fit1$kernel
  
  # info of fit2
  Ki.2 <- fit2$Ki
  theta.2 <- fit2$theta
  rho <- fit2$rho
  y.2 <- fit2$y
  mu.hat.2 <- fit2$mu.hat
  kernel.2 <- fit2$kernel
  
  n <- length(G)
  y <- c(y.1, rho*y.1 + y.2)
  mu.hat <- c(rep(mu.hat.1, n), rep(rho*mu.hat.1 + mu.hat.2, n))
  mu <- rho*mu.hat.1 + mu.hat.2
  
  Kh <- FIGP.kernel.discrete(theta.1, nu, G, gN=NULL, XN, kernel.1)
  Kd <- FIGP.kernel.discrete(theta.2, nu, G, gN=NULL, XN, kernel.2)
  
  # compute linear kernel
  # rnd <- nrow(XN)
  # R.1 <- sqrt(distance(t(t(XN)/theta.1)))
  # Phi.1 <- matern.kernel(R.1, nu=nu)
  # R.2 <- sqrt(distance(t(t(XN)/theta.2)))
  # Phi.2 <- matern.kernel(R.2, nu=nu)
  # 
  # A <- matrix(0,ncol=n,nrow=rnd)
  # for(i in 1:n)  A[,i] <- apply(XN, 1, G[[i]])
  # Kh <- (t(A) %*% Phi.1 %*% A) / rnd # should be /rnd^2 but the values become too small, but it doesn't hurt without it because of scale parameter
  # Kh <- (Kh+t(Kh))/2
  # Kd <- (t(A) %*% Phi.2 %*% A) / rnd # should be /rnd^2 but the values become too small, but it doesn't hurt without it because of scale parameter
  # Kd <- (Kd+t(Kd))/2
  K1 <- cbind(Kh, rho*Kh)
  K2 <- cbind(rho*Kh, rho^2*Kh + Kd)
  K <- rbind(K1, K2)
  Ki <- solve(K+diag(nug,2*n))
  
  # a <- matrix(0,ncol=1,nrow=rnd)
  # a[,1] <- gN
  # KhX <- (t(a) %*% Phi.1 %*% A) / rnd 
  # KdX <- (t(a) %*% Phi.2 %*% A) / rnd 
  # KhXX <- (t(a) %*% Phi.1 %*% a) / rnd 
  # KdXX <- (t(a) %*% Phi.2 %*% a) / rnd 
  
  KhX <- FIGP.kernel.discrete(theta.1, nu, G, gN=gN, XN, kernel.1)
  KdX <- FIGP.kernel.discrete(theta.2, nu, G, gN=gN, XN, kernel.2)
  KhXX <- FIGP.kernel.discrete(theta.1, nu, G=NULL, gN=gN, XN, kernel.1)
  KdXX <- FIGP.kernel.discrete(theta.2, nu, G=NULL, gN=gN, XN, kernel.2)
    
  KX <- cbind(rho*KhX, rho^2*KhX + KdX)
  KXX <- rho^2*KhXX + KdXX
  
  mup2 <- drop(mu + KX %*% Ki %*% (y - mu.hat))
  Sigmap2 <- pmax(0,diag(KXX + diag(nug, 1) - KX %*% Ki %*% t(KX)))
  
  return(list(mu=mup2, sig2=Sigmap2))
}


##### predict ys (the far-field pattern)
ys_hat <- function(figp.1, gN, XN, U, fidelity=c("single","multi")[1], figp.2=NULL){
  n.comp <- ncol(U)
  ynew <- s2new <- matrix(0,ncol=n.comp,nrow=1)
  if(fidelity=="single"){
    for(i in 1:n.comp){
      pred.out <- pred.FIGP.XN(figp.1[[i]], gN, XN)
      ynew[,i] <- pred.out$mu
      s2new[,i] <- pred.out$sig2
    }
  }else{
    for(i in 1:n.comp){
      pred.out <- pred.MF(figp.1[[i]], figp.2[[i]], gN, XN)
      ynew[,i] <- pred.out$mu
      s2new[,i] <- pred.out$sig2
    }
  }

  # reconstruct the image
  pred.recon <- c(ynew %*% t(U))
  return(list(mean=pred.recon, s2=drop(s2new), ynew=drop(ynew)))
}



