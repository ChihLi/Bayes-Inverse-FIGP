# log likelihood of B
logl.B <- function(eta, tau2, X, B, basis, nu, nug) 
{
  n <- length(B)
  R <- sqrt(distance(t(t(X)/eta)))
  K <- matern.kernel(R, nu=nu)
  E <- t(basis) %*% K %*% basis
 
  Ei <- solve(E+diag(nug,n))
  ldetE <- determinant(E, logarithm=TRUE)$modulus
  
  ll <- - (n/2)*log(tau2) - (1/2)*ldetE - (1/2)*(t(B) %*% Ei %*% B)/tau2
  return(ll)
}

# posterior of g and ys given yp
BayesInverse_KL <- function(klgp, XN, X.grid, yp, U, fraction=0.95, emulator=c("figp","klgp")[2], figp=NULL,
                            nu=2.5, nug=sqrt(.Machine$double.eps),
                            MC.samples=1e4, MC.burnin=5000, plot.fg=FALSE,
                            prior=list(prior.tau2.shape=1, prior.tau2.rate=0.2,
                                       prior.s2.shape=1, prior.s2.rate=1e-5,
                                       prior.eta.shape=1, prior.eta.rate=0.2),
                            init=list(eta=c(5,5), tau2=0.2, s2=1e-5), 
                            MH.const=list(cGP=0.5, cs2=0.5, ceta=c(0.5,0.5)),
                            seed=NULL, trace=TRUE){
  
  # vector size
  n.grid <- nrow(X.grid)
  N <- nrow(XN)
  m <- length(yp)
  
  # priors
  for (name in names(prior)) {
    assign(name, prior[[name]])
  }
  
  # init
  for (name in names(init)) {
    assign(name, init[[name]])
  }
  
  # MH constants
  for (name in names(MH.const)) {
    assign(name, MH.const[[name]])
  }
  
  # set up
  acceptCount.g <- acceptCount.s2 <- 0
  acceptCount.eta <- c(0,0)
  g.sample <- matrix(0, nrow=n.grid, ncol=MC.samples)
  yhat <- matrix(0, nrow=m, ncol=MC.samples)
  
  if(!is.null(seed)){
    set.seed(seed)
  }
  
  S <- 0
  for(i in 1:1000){
    eta <- rgamma(2, shape = prior.eta.shape, rate = prior.eta.rate)
    R <- sqrt(distance(t(t(XN)/eta)))
    S <- S + matern.kernel(R, nu=nu)/1000
  }
  eig.out <- eigen(S)
  M <- which(cumsum(eig.out$value)/sum(eig.out$value)>fraction)[1]
  B <- rep(0,M)
  basis <- eig.out$vectors[,1:M]

  yU <- yp %*% U
  pred.out <- ys_hat.KL(klgp=klgp, B.new=B, basis=basis, U=U, emulator=emulator, figp=figp, XN=XN)
  dU <- yU - pred.out$ynew
  
  R <- sqrt(distance(t(t(XN)/eta)))
  K <- matern.kernel(R, nu=nu)
  E <- t(basis) %*% K %*% basis
 
  if(trace)  pb <- txtProgressBar(min = 0, max = MC.samples, initial = 0, style=3) 
  
  if(!is.null(seed)){
    set.seed(seed)
  }
  
  ##### MCMC sampling #####
  for(ii in 1:MC.samples){
    
    if(trace)  setTxtProgressBar(pb, ii)
    
    # sample gN: MH
    B.new <- cGP*sqrt(tau2)*t(chol(E+diag(nug,M))) %*% rnorm(M) + B
    pred.out.new <- ys_hat.KL(klgp=klgp, B.new=B.new, basis=basis, U=U, emulator=emulator, figp=figp, XN=XN)
    dU.new <- yU - pred.out.new$ynew
    
    logL <- logl.Yp(pred.out, s2, dU, yp) + logl.B(eta, tau2, XN, B, basis, nu, nug)
    logL.new <- logl.Yp(pred.out.new, s2, dU.new, yp) + logl.B(eta, tau2, XN, B.new, basis, nu, nug)
    
    if(runif(1) <= exp(logL.new - logL)){
      B <- B.new
      dU <- dU.new
      pred.out <- pred.out.new
      acceptCount.g <- acceptCount.g + 1
    }
    
    # sample s2: MH
    s2.new <- exp(rnorm(1,0,cs2) + log(s2)) 
    
    logL <- logl.Yp(pred.out, s2, dU, yp) + log(dinvgamma(s2, prior.s2.shape, prior.s2.rate))
    logL.new <- logl.Yp(pred.out.new, s2.new, dU.new, yp) + log(dinvgamma(s2.new, prior.s2.shape, prior.s2.rate))
    
    if(runif(1) <= exp(logL.new - logL)){
      s2 <- s2.new
      acceptCount.s2 <- acceptCount.s2 + 1
    }
    
    # sample eta: MH
    for(j in 1:2){
      eta.new <- eta
      eta.new[j] <- exp(rnorm(1,0,ceta[j]) + log(eta[j])) 
      
      logL <- logl.B(eta, tau2, XN, B, basis, nu, nug) + sum(log(dinvgamma(1/eta, prior.eta.shape, prior.eta.rate)))
      logL.new <- logl.B(eta.new, tau2, XN, B, basis, nu, nug) + sum(log(dinvgamma(1/eta.new, prior.eta.shape, prior.eta.rate)))
      
      if(runif(1) <= exp(logL.new - logL)){
        R <- sqrt(distance(t(t(XN)/eta)))
        K <- matern.kernel(R, nu=nu)
        E.tmp <- t(basis) %*% K %*% basis
        if(cond(E.tmp)<1e14){
          eta <- eta.new
          acceptCount.eta[j] <- acceptCount.eta[j] + 1
          E <- E.tmp
        }
      }
    }
    
    # sample tau2
    tau2.shape <- prior.tau2.shape + M/2
    tau2.rate <- prior.tau2.rate + drop(t(B) %*% solve(E+diag(nug,M)) %*% B)/2
    tau2 <- 1/rgamma(1, tau2.shape, tau2.rate)
    
    g.sample[,ii] <- drop(basis %*% B)
    yhat[,ii] <- drop(U %*% diag(sqrt(pred.out$s2)) %*% t(U)) %*% rnorm(m) + pred.out$mean 
    
    if(plot.fg){
      if (ii%%100 == 0){
        par(mfrow=c(1,3))
        image(matrix(yp,sqrt(m),sqrt(m)),col=heat.colors(12, rev = FALSE), main=expression({y^p}))
        image(matrix(yhat[,ii],sqrt(m),sqrt(m)),col=heat.colors(12, rev = FALSE), main=bquote({y^s}(g[.(ii)])))
        contour(matrix(yhat[,ii],sqrt(m),sqrt(m)), add = TRUE, nlevels = 5)
        image(matrix(g.sample[,ii],sqrt(n.grid),sqrt(n.grid)),col=cm.colors(12, rev = FALSE), main=bquote(g[.(ii)]))
      }
    }
    
    
    if(ii <= MC.burnin){
      if (ii%%100 == 0){
        # cat("tau2=",tau2, "s2=",s2, "eta=",eta, "\n")
        # cat("cGP=",cGP, "cs2=",cs2, "ceta=", ceta, "\n")
        
        if (acceptCount.g/100 > 0.4){
          cGP <- 5.75 * cGP
        }else if(acceptCount.g/100 < 0.2){
          cGP <- 0.25 * cGP
        }
        
        if (acceptCount.s2/100 > 0.4){
          cs2 <- 5.75 * cs2
        }else if(acceptCount.s2/100 < 0.2){
          cs2 <- 0.25 * cs2
        }
        
        for(j in 1:2){
          if (acceptCount.eta[j]/100 > 0.4){
            ceta[j] <- 5.75 * ceta[j]
          }else if(acceptCount.eta[j]/100 < 0.2){
            ceta[j] <- 0.25 * ceta[j]
          }
        }
        
        acceptCount.g <- acceptCount.s2 <- 0
        acceptCount.eta <- c(0,0)
      }
    }
  }
  
  if(trace){
    setTxtProgressBar(pb, MC.samples)
    close(pb)
  }

  
  return(list(g.sample=g.sample, yhat=yhat))
}
