# posterior of g given gN
g.pred <- function(gN, XN, eta, tau2, nu, nug, x, Ki, lite){
  
  R <- sqrt(distance(t(t(XN)/eta)))
  K <- matern.kernel(R, nu=nu)
  
  RX <- sqrt(distance(t(t(x)/eta), t(t(XN)/eta)))
  KX <- matern.kernel(RX, nu=nu)
  
  mu <- KX %*% Ki %*% gN
  if(lite){
    s2 <- pmax(nug,diag(tau2*(diag(1+nug,nrow(x)) - KX %*% Ki %*% t(KX))))
  }else{
    RXX <- sqrt(distance(t(t(x)/eta), t(t(x)/eta)))
    KXX <- matern.kernel(RXX, nu=nu)
    s2 <- tau2*(KXX + diag(nug,nrow(x)) - KX %*% Ki %*% t(KX))
    s2 <- (s2+t(s2))/2
  }
  
  return(list(mu=mu, s2=s2))
}



# posterior of g and ys given yp
BayesInverse_FIGP <- function(XN, X.grid, yp, U, 
                              nu=2.5, nug=sqrt(.Machine$double.eps),
                              figp.1, figp.2=NULL, fidelity=c("single","multi")[1], 
                              MC.samples = 1e4, MC.burnin = 5000, nchain = 5, 
                              prior=list(prior.tau2.shape=4, prior.tau2.rate=1,
                                         prior.s2.shape=100, prior.s2.rate=1e-4,
                                         prior.eta.shape=1.5, prior.eta.rate=3.9/1.5),
                              lite=TRUE, seed=NULL, Rhat=FALSE, parallel=TRUE, ncores=detectCores()-1){
  
  #################################################################################################
  # XN:          the input for the gN realization
  # X.grid:      discretized angles a1 (incident wave) and a2 (scattered wave)
  # yp:          output yp
  # U:           PC of y
  # nu:          smoothness parameter of matern kernel
  # nug:         nugget for numerical stability
  # figp.1:      if `fidelity` = "single", FIGP emulator for single computer model
  #              if `fidelity` = "multi", FIGP emulator for low-fidelity computer model
  # figp.2:      if `fidelity` = "single", NULL
  #              if `fidelity` = "multi", FIGP emulator for discrepancy between high- and low-fidelity computer models
  # fidelity:    "single" or "multi" indicates single-fidelity or multi-fidelty computer model
  # MC.samples:  number of Monte-Carlo samples
  # MC.burnin:   number of initial burn-in Monte-Carlo samples
  # nchain:      number of chains of MCMC sampling; each one runs different initial parameters
  # prior:       priors for the parameters 
  # lite:        sampling gN using the full predictive covariance matrix (FALSE) or diagonal matrix (TRUE)
  # seed:        select seed number for reproducibility; if not; select `NULL`
  # Rhat:        compute Rhat
  # parallel:    do parallel for `nchain`
  # ncores:      number of cores for palatalization
  #################################################################################################
  
  # vector size
  n.grid <- nrow(X.grid)
  d <- ncol(X.grid)
  N <- nrow(XN)
  m <- length(yp)
  
  # priors
  for (name in names(prior)) {
    assign(name, prior[[name]])
  }
  
  if(parallel){
    
    # Parallel setup
    no_cores <- min(nchain, ncores)  # Detect available cores
    cl <- makeCluster(no_cores)     # Create cluster
    
    if(fidelity == "single")
      clusterExport(cl, varlist=c("fem.figp", "g.pred", "pred.MF", "pred.FIGP.XN", "cond", "eps", "dinvgamma", "logl.gN", "logl.Yp", "FIGP.kernel.discrete", "ys_hat", "U", "distance", "matern.kernel"))
    else{
      clusterExport(cl, varlist=c("born.figp", "diff.gp.fit", "g.pred", "pred.MF", "pred.FIGP.XN", "cond", "eps", "dinvgamma", "logl.gN", "logl.Yp", "FIGP.kernel.discrete", "ys_hat", "U", "distance", "matern.kernel"))
    }
    
    # Parallelized loop over jj
    results <- parLapply(cl, 1:nchain, function(jj) {

      # set up
      g.sample <- matrix(0, nrow=MC.samples, ncol=n.grid)
      yhat <- matrix(0, nrow=MC.samples, ncol=m)
      eta.sample <- matrix(0, nrow=MC.samples, ncol=d)
      tau2.sample <-  s2.sample <- rep(0, MC.samples)
      
      # initial value
      eta <- eta.sample[1,] <- rep(rgamma(1, shape = prior.eta.shape, rate = prior.eta.rate), d)
      tau2 <- tau2.sample[1] <- 1/rgamma(1, shape = prior.tau2.shape, rate = prior.tau2.rate)
      s2 <- s2.sample[1] <- 1/rgamma(1, shape = prior.s2.shape, rate = prior.s2.rate)
      
      R <- sqrt(distance(t(t(XN)/eta)))
      K <- matern.kernel(R, nu=nu)
      Ki <- solve(K+diag(nug,N))
      gN <- t(mvtnorm::rmvnorm(1, mean = rep(0,N), sigma = tau2*(K+diag(nug,N))))
      
      yU <- yp %*% U
      pred.out <- ys_hat(figp.1, gN=gN, XN=XN, U=U, fidelity=fidelity, figp.2=figp.2)
      dU <- yU - pred.out$ynew
      
      # set seed
      if(!is.null(seed)){
        set.seed(seed)
      }
      
      # likelihood
      ll.Yp <- logl.Yp(pred.out, s2, dU, yp) 
      ll.gN <- logl.gN(eta, tau2, XN, gN, nu, nug)
      
      
      ##### MCMC sampling #####
      for(ii in 1:MC.samples){
        
        ### sample gN using Elliptical Slice Sampling (ESS): this code follows the one in the `deepgp` package
        g.prior <-  t(mvtnorm::rmvnorm(1, mean = rep(0,N), sigma = tau2*(K+diag(nug,N)))) 
        
        # Initialize a and bounds on a
        a <- runif(1, min = 0, max = 2 * pi)
        amin <- a - 2 * pi
        amax <- a
        
        # Calculate proposed values, accept or reject, repeat if necessary
        accept <- FALSE
        count <- 0
        ru <- runif(1)
        
        while (accept == FALSE) {
          count <- count + 1
         
          gN.new <-  gN * cos(a) + g.prior * sin(a)
          pred.out.new <- ys_hat(figp.1, gN=gN.new, XN=XN, U=U, fidelity=fidelity, figp.2=figp.2)
          dU.new <- yU - pred.out.new$ynew
          
          logL <- ll.Yp + ll.gN
          
          ll.Yp.new <- logl.Yp(pred.out.new, s2, dU.new, yp)
          ll.gN.new <- logl.gN(eta, tau2, XN, gN.new, nu, nug)
          logL.new <- ll.Yp.new + ll.gN.new
          
          if(ru <= exp(logL.new - logL)){
            gN <- gN.new
            dU <- dU.new
            ll.Yp <- ll.Yp.new
            ll.gN <- ll.gN.new
            pred.out <- pred.out.new
            accept <- TRUE
          }else {
            # update the bounds on a and repeat
            if (a < 0) {
              amin <- a
            } else {
              amax <- a
            }
            a <- runif(1, amin, amax)
            if (count > 100) stop("reached maximum iterations of ESS")
          }
        }
        
        # sample s2: MH
        l <- 1; u <- 2
        s2.new <- runif(1, min = l * s2 / u, max = u * s2 / l)
        
        logL <- ll.Yp + dinvgamma(s2, prior.s2.shape, prior.s2.rate, log = TRUE) - log(s2) + log(s2.new)
        
        ll.Yp.new <- logl.Yp(pred.out, s2.new, dU, yp)
        logL.new <- ll.Yp.new + dinvgamma(s2.new, prior.s2.shape, prior.s2.rate, log = TRUE)
        
        if(runif(1) <= exp(logL.new - logL)){
          s2 <- s2.new
          ll.Yp <- ll.Yp.new
        }
        s2.sample[ii] <- s2
        
        # sample eta: MH
        eta.new <- eta
        for(j in 1:2){
          eta.new[j] <- runif(1, min = l * eta[j] / u, max = u * eta[j] / l)
          
          logL <- ll.gN + sum(dgamma(eta-eps, prior.eta.shape, prior.eta.rate, log = TRUE)) -
            sum(log(eta)) + sum(log(eta.new))
          ll.gN.new <- logl.gN(eta.new, tau2, XN, gN, nu, nug)
          logL.new <- ll.gN.new + sum(dgamma(eta.new-eps, prior.eta.shape, prior.eta.rate, log = TRUE))
          
          if(runif(1) <= exp(logL.new - logL)){
            R <- sqrt(distance(t(t(XN)/eta.new)))
            K.tmp <- matern.kernel(R, nu=nu)
            if(cond(K.tmp)<1e14){ # for numerical stability
              eta <- eta.new
              K <- K.tmp
              Ki <- solve(K+diag(nug,N))
              ll.gN <- ll.gN.new
            }
          }
        }
        eta.sample[ii,] <- eta
        
        # sample tau2
        tau2.shape <- prior.tau2.shape + N/2
        tau2.rate <- prior.tau2.rate + drop(t(gN) %*% Ki %*% gN)/2
        tau2 <- 1/rgamma(1, tau2.shape, tau2.rate)
        tau2.sample[ii] <- tau2
        ll.gN <- logl.gN(eta, tau2, XN, gN, nu, nug)
        post.g <- g.pred(gN, XN, eta, tau2, nu, nug, X.grid, Ki, lite)
        if(lite){
          ghat <- rnorm(n.grid, post.g$mu, sd = sqrt(post.g$s2))
        }else{
          ghat <- t(mvtnorm::rmvnorm(1, mean = post.g$mu, sigma = post.g$s2))
        }
        g.sample[ii,] <- ghat
        yhat[ii,] <- drop(U %*% diag(sqrt(pred.out$s2)) %*% t(U)) %*% rnorm(m) + pred.out$mean 
      }
      
      return(list(g.sample = g.sample, yhat = yhat, tau2.sample = tau2.sample,
                  eta.sample = eta.sample, s2.sample = s2.sample))
    })
    
    # Stop the cluster after the computation is finished
    stopCluster(cl)
    
    # Gather the results from all chains
    g.ls <- lapply(results, function(x) x$g.sample)
    yhat.ls <- lapply(results, function(x) x$yhat)
    tau2.ls <- lapply(results, function(x) x$tau2.sample)
    eta.ls <- lapply(results, function(x) x$eta.sample)
    s2.ls <- lapply(results, function(x) x$s2.sample)
    
  }else{
    
    g.ls <- yhat.ls <- tau2.ls <- eta.ls <- s2.ls <- vector("list",nchain)
    
    for(jj in 1:nchain) {
      
      # set up
      g.sample <- matrix(0, nrow=MC.samples, ncol=n.grid)
      yhat <- matrix(0, nrow=MC.samples, ncol=m)
      eta.sample <- matrix(0, nrow=MC.samples, ncol=d)
      tau2.sample <-  s2.sample <- rep(0, MC.samples)
      
      # initial value
      eta <- eta.sample[1,] <- rep(rgamma(1, shape = prior.eta.shape, rate = prior.eta.rate), d)
      tau2 <- tau2.sample[1] <- 1/rgamma(1, shape = prior.tau2.shape, rate = prior.tau2.rate)
      s2 <- s2.sample[1] <- 1/rgamma(1, shape = prior.s2.shape, rate = prior.s2.rate)
      
      R <- sqrt(distance(t(t(XN)/eta)))
      K <- matern.kernel(R, nu=nu)
      Ki <- solve(K+diag(nug,N))
      gN <- t(mvtnorm::rmvnorm(1, mean = rep(0,N), sigma = tau2*(K+diag(nug,N))))
      
      yU <- yp %*% U
      pred.out <- ys_hat(figp.1, gN=gN, XN=XN, U=U, fidelity=fidelity, figp.2=figp.2)
      dU <- yU - pred.out$ynew
      
      # set seed
      if(!is.null(seed)){
        set.seed(seed)
      }
      
      # likelihood
      ll.Yp <- logl.Yp(pred.out, s2, dU, yp) 
      ll.gN <- logl.gN(eta, tau2, XN, gN, nu, nug)
      
      ##### MCMC sampling #####
      for(ii in 1:MC.samples){
        
        ### sample gN using Elliptical Slice Sampling (ESS): this code follows the one in the `deepgp` package
        g.prior <-  t(mvtnorm::rmvnorm(1, mean = rep(0,N), sigma = tau2*(K+diag(nug,N)))) 
        
        # Initialize a and bounds on a
        a <- runif(1, min = 0, max = 2 * pi)
        amin <- a - 2 * pi
        amax <- a
        
        # Calculate proposed values, accept or reject, repeat if necessary
        accept <- FALSE
        count <- 0
        ru <- runif(1)
        
        while (accept == FALSE) {
          count <- count + 1
          
          gN.new <-  gN * cos(a) + g.prior * sin(a)
          pred.out.new <- ys_hat(figp.1, gN=gN.new, XN=XN, U=U, fidelity=fidelity, figp.2=figp.2)
          dU.new <- yU - pred.out.new$ynew
          
          logL <- ll.Yp + ll.gN
          
          ll.Yp.new <- logl.Yp(pred.out.new, s2, dU.new, yp)
          ll.gN.new <- logl.gN(eta, tau2, XN, gN.new, nu, nug)
          logL.new <- ll.Yp.new + ll.gN.new
          
          if(ru <= exp(logL.new - logL)){
            gN <- gN.new
            dU <- dU.new
            ll.Yp <- ll.Yp.new
            ll.gN <- ll.gN.new
            pred.out <- pred.out.new
            accept <- TRUE
          }else {
            # update the bounds on a and repeat
            if (a < 0) {
              amin <- a
            } else {
              amax <- a
            }
            a <- runif(1, amin, amax)
            if (count > 100) stop("reached maximum iterations of ESS")
          }
        }
        
        # sample s2: MH
        l <- 1; u <- 2
        s2.new <- runif(1, min = l * s2 / u, max = u * s2 / l)
        
        logL <- ll.Yp + dinvgamma(s2, prior.s2.shape, prior.s2.rate, log = TRUE) - log(s2) + log(s2.new)
        
        ll.Yp.new <- logl.Yp(pred.out, s2.new, dU, yp)
        logL.new <- ll.Yp.new + dinvgamma(s2.new, prior.s2.shape, prior.s2.rate, log = TRUE)
        
        if(runif(1) <= exp(logL.new - logL)){
          s2 <- s2.new
          ll.Yp <- ll.Yp.new
        }
        s2.sample[ii] <- s2
        
        # sample eta: MH
        eta.new <- eta
        for(j in 1:2){
          eta.new[j] <- runif(1, min = l * eta[j] / u, max = u * eta[j] / l)
          
          logL <- ll.gN + sum(dgamma(eta-eps, prior.eta.shape, prior.eta.rate, log = TRUE)) -
            sum(log(eta)) + sum(log(eta.new))
          ll.gN.new <- logl.gN(eta.new, tau2, XN, gN, nu, nug)
          logL.new <- ll.gN.new + sum(dgamma(eta.new-eps, prior.eta.shape, prior.eta.rate, log = TRUE))
          
          if(runif(1) <= exp(logL.new - logL)){
            R <- sqrt(distance(t(t(XN)/eta.new)))
            K.tmp <- matern.kernel(R, nu=nu)
            if(cond(K.tmp)<1e14){  # for numerical stability
              eta <- eta.new
              K <- K.tmp
              Ki <- solve(K+diag(nug,N))
              ll.gN <- ll.gN.new
            }
          }
        }
        eta.sample[ii,] <- eta
        
        # sample tau2
        tau2.shape <- prior.tau2.shape + N/2
        tau2.rate <- prior.tau2.rate + drop(t(gN) %*% Ki %*% gN)/2
        tau2 <- 1/rgamma(1, tau2.shape, tau2.rate)
        tau2.sample[ii] <- tau2
        ll.gN <- logl.gN(eta, tau2, XN, gN, nu, nug)
        post.g <- g.pred(gN, XN, eta, tau2, nu, nug, X.grid, Ki, lite)
        if(lite){
          ghat <- rnorm(n.grid, post.g$mu, sd = sqrt(post.g$s2))
        }else{
          ghat <- t(mvtnorm::rmvnorm(1, mean = post.g$mu, sigma = post.g$s2))
        }
        g.sample[ii,] <- ghat
        yhat[ii,] <- drop(U %*% diag(sqrt(pred.out$s2)) %*% t(U) %*% rnorm(m) + pred.out$mean)
      }
      g.ls[[jj]] <- g.sample
      yhat.ls[[jj]] <- yhat
      tau2.ls[[jj]] <- tau2.sample
      eta.ls[[jj]] <- eta.sample
      s2.ls[[jj]] <- s2.sample
    }
  }
  
  if(Rhat){
    R.hat <- rep(0, 4)
    names(R.hat) <- c("ginv", "tau2", "eta", "s2")
    
    # Compute R-hat for each component
    mcmc_list <- mcmc.list(lapply(g.ls, mcmc))
    R.hat[1] <- gelman.diag(mcmc_list)$mpsrf
    cat("R-hat:", R.hat[1], "for g.inverse.\n")
    
    mcmc_list <- mcmc.list(lapply(tau2.ls, mcmc))
    R.hat[2] <- gelman.diag(mcmc_list)$psrf[1]
    cat("R-hat:", R.hat[2], "for tau2.\n")
    
    mcmc_list <- mcmc.list(lapply(eta.ls, mcmc))
    R.hat[3] <- gelman.diag(mcmc_list)$mpsrf
    cat("R-hat:", R.hat[3], "for eta.\n")
    
    mcmc_list <- mcmc.list(lapply(s2.ls, mcmc))
    R.hat[4] <- gelman.diag(mcmc_list)$psrf[1]
    cat("R-hat:", R.hat[4], "for s2.\n")
  }else{
    R.hat <- NULL
  }

  
  g.ls <- lapply(g.ls, FUN=function(x) x[(MC.burnin+1):MC.samples, ])
  yhat.ls <- lapply(yhat.ls, FUN=function(x) x[(MC.burnin+1):MC.samples, ])
  tau2.ls <- lapply(tau2.ls, FUN=function(x) x[(MC.burnin+1):MC.samples])
  eta.ls <- lapply(eta.ls, FUN=function(x) x[(MC.burnin+1):MC.samples, ])
  s2.ls <- lapply(s2.ls, FUN=function(x) x[(MC.burnin+1):MC.samples])
  
  return(list(g.inverse=g.ls, yhat=yhat.ls, tau2=tau2.ls, eta=eta.ls, s2=s2.ls, R.hat=R.hat))
}
