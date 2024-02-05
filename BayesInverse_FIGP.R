# posterior of g given gN
g.pred <- function(gN, XN, eta, tau2, nu, nug, x){
  
  R <- sqrt(distance(t(t(XN)/eta)))
  K <- matern.kernel(R, nu=nu)
  Ki <- solve(K+diag(nug,nrow(XN)))
  
  RX <- sqrt(distance(t(t(x)/eta), t(t(XN)/eta)))
  KX <- matern.kernel(RX, nu=nu)
  
  mu <- KX %*% Ki %*% gN
  s2 <- pmax(0,diag(tau2*(diag(1+nug,nrow(x)) - KX %*% Ki %*% t(KX))))
  
  return(list(mu=mu, s2=s2))
}

# posterior of g and ys given yp
BayesInverse_FIGP <- function(XN, X.grid, yp, U, 
                              nu=2.5, nug=sqrt(.Machine$double.eps),
                              figp.1, figp.2=NULL, fidelity=c("single","multi")[1], 
                              MC.samples=1e4, MC.burnin=5000, plot.fg=FALSE,
                              prior=list(prior.tau2.shape=1, prior.tau2.rate=0.2,
                                         prior.s2.shape=1, prior.s2.rate=1e-5,
                                         prior.eta.shape=1, prior.eta.rate=0.2),
                              init=list(eta=c(5,5), tau2=0.2, s2=1e-5, gN=rep(0,nrow(XN))), 
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
  
  yU <- yp %*% U
  pred.out <- ys_hat(figp.1, gN=gN, XN=XN, U=U, fidelity=fidelity, figp.2=figp.2)
  dU <- yU - pred.out$ynew
  R <- sqrt(distance(t(t(XN)/eta)))
  K <- matern.kernel(R, nu=nu)
  Ki <- solve(K+diag(nug,N))
  
  if(trace)  pb <- txtProgressBar(min = 0, max = MC.samples, initial = 0, style=3) 
  
  if(!is.null(seed)){
    set.seed(seed)
  }
  
  # eta <- rgamma(2, shape = prior.eta.shape, rate = prior.eta.rate)
  # tau2 <- 1/rgamma(1, shape = prior.tau2.shape, rate = prior.tau2.rate)
  # s2 <- 1/rgamma(1, shape = prior.s2.shape, rate = prior.s2.rate)
  
  ##### MCMC sampling #####
  for(ii in 1:MC.samples){
    
    if(trace) setTxtProgressBar(pb, ii)
    
    # sample gN: MH
    gN.new <- cGP*sqrt(tau2)*t(chol(K+diag(nug,N))) %*% rnorm(N) + gN
    pred.out.new <- ys_hat(figp.1, gN=gN.new, XN=XN, U=U, fidelity=fidelity, figp.2=figp.2)
    dU.new <- yU - pred.out.new$ynew
    
    logL <- logl.Yp(pred.out, s2, dU, yp) + logl.gN(eta, tau2, XN, gN, nu, nug)
    logL.new <- logl.Yp(pred.out.new, s2, dU.new, yp) + logl.gN(eta, tau2, XN, gN.new, nu, nug)
    
    if(runif(1) <= exp(logL.new - logL)){
      gN <- gN.new
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
      
      logL <- logl.gN(eta, tau2, XN, gN, nu, nug) + sum(log(dinvgamma(1/eta, prior.eta.shape, prior.eta.rate)))
      logL.new <- logl.gN(eta.new, tau2, XN, gN, nu, nug) + sum(log(dinvgamma(1/eta.new, prior.eta.shape, prior.eta.rate)))
      
      if(runif(1) <= exp(logL.new - logL)){
        R <- sqrt(distance(t(t(XN)/eta.new)))
        K.tmp <- matern.kernel(R, nu=nu)
        if(cond(K.tmp)<1e14){
          eta <- eta.new
          acceptCount.eta[j] <- acceptCount.eta[j] + 1
          K <- K.tmp
          Ki <- solve(K+diag(nug,N))
        }
      }
    }
    
    # sample tau2
    tau2.shape <- prior.tau2.shape + N/2
    tau2.rate <- prior.tau2.rate + drop(t(gN) %*% Ki %*% gN)/2
    tau2 <- 1/rgamma(1, tau2.shape, tau2.rate)
    
    
    post.g <- g.pred(gN, XN, eta, tau2, nu, nug, X.grid)
    ghat <- rnorm(n.grid, post.g$mu, sd = sqrt(post.g$s2))
    g.sample[,ii] <- ghat
    yhat[,ii] <- drop(U %*% diag(sqrt(pred.out$s2)) %*% t(U)) %*% rnorm(m) + pred.out$mean 
    
    if(plot.fg){
      if (ii%%100 == 0){
        par(mfrow=c(1,3))
        image(matrix(yp,sqrt(m),sqrt(m)),col=heat.colors(12, rev = FALSE), main=expression({y^p}))
        image(matrix(yhat[,ii],sqrt(m),sqrt(m)),col=heat.colors(12, rev = FALSE), main=bquote({y^s}(g[.(ii)])))
        contour(matrix(yhat[,ii],sqrt(m),sqrt(m)), add = TRUE, nlevels = 5)
        image(matrix(ghat,sqrt(n.grid),sqrt(n.grid)),zlim=c(0,1.2),col=cm.colors(12, rev = FALSE), main=bquote(g[.(ii)]))
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
