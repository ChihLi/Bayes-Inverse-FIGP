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
BayesInverse_KL <- function(klgp, XN, X.grid, yp, U, 
                            fraction=0.95, emulator=c("figp","klgp")[2], figp=NULL,
                            nu=2.5, nug=sqrt(.Machine$double.eps),
                            MC.samples = 1e4, MC.burnin = 5000, nchain = 5, 
                            prior=list(prior.tau2.shape=4, prior.tau2.rate=1,
                                       prior.s2.shape=100, prior.s2.rate=1e-4,
                                       prior.eta.shape=1.5, prior.eta.rate=3.9/1.5),
                            seed=NULL, parallel=TRUE, ncores=detectCores()-1){
  
  ##################################################################################################
  # klgp:        if `emulator` is `klgp`, GP emulator based on KL expansion
  # XN:          the input for the gN realization
  # X.grid:      discretized angles a1 (incident wave) and a2 (scattered wave)
  # yp:          output yp
  # U:           PC of y
  # fraction:    determine how many components of KL expansion for estimating g inverse
  # emulator:    "figp" or "klgp" indicates FIGP emulator or KLGP emulator
  # nu:          smoothness parameter of matern kernel
  # nug:         nugget for numerical stability
  # figp:        if `emulator` = "figp", FIGP emulator for computer model
  # MC.samples:  number of Monte-Carlo samples
  # MC.burnin:   number of initial burn-in Monte-Carlo samples
  # nchain:      number of chains of MCMC sampling; each one runs different initial parameters
  # prior:       priors for the parameters 
  # seed:        select seed number for reproducibility; if not; select `NULL`
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
  
  
  # save Ki for computational ease
  if(emulator == "figp"){
    Ki <- vector("list",ncol(U))
    for(i in 1:ncol(U)){
      K <- FIGP.kernel.discrete(figp[[i]]$theta, figp[[i]]$nu, figp[[i]]$G, gN=NULL, XN, figp[[i]]$kernel)
      Ki[[i]] <- solve(K+diag(figp[[i]]$nug, length(figp[[i]]$y)))
    }  
  }else {
    Ki <- NULL
  }
  
  
  
  if(parallel){
    
    # Parallel setup
    no_cores <- min(nchain, ncores)  # Detect available cores
    cl <- makeCluster(no_cores)     # Create cluster
    
    # Export necessary objects to the cluster
    if(emulator=="klgp"){
      clusterExport(cl, varlist=c("klgp", "cond", "eps", "pred.sepGP", "KL.Bnew", "FIGP.kernel.discrete", "pred.FIGP.XN", "dinvgamma", "logl.B", "logl.Yp", "ys_hat.KL", "U", "distance", "matern.kernel"))
    }else{
      clusterExport(cl, varlist=c("fem.figp", "cond", "eps", "pred.sepGP", "KL.Bnew", "FIGP.kernel.discrete", "pred.FIGP.XN", "dinvgamma", "logl.B", "logl.Yp", "ys_hat.KL", "U", "distance", "matern.kernel"))
    }
    
    # Parallelized loop over jj
    results <- parLapply(cl, 1:nchain, function(jj) {
      # set up
      g.sample <- matrix(0, nrow=MC.samples, ncol=n.grid)
      yhat <- matrix(0, nrow=MC.samples, ncol=m)
      eta.sample <- matrix(0, nrow=MC.samples, ncol=d)
      tau2.sample <-  s2.sample <- rep(0, MC.samples)
      
      eta <- eta.sample[1,] <- rep(rgamma(1, shape = prior.eta.shape, rate = prior.eta.rate), d)
      tau2 <- tau2.sample[1] <- 1/rgamma(1, shape = prior.tau2.shape, rate = prior.tau2.rate)
      s2 <- s2.sample[1] <- 1/rgamma(1, shape = prior.s2.shape, rate = prior.s2.rate)
      
      S <- 0
      for(i in 1:1000){
        eta <- rgamma(2, shape = prior.eta.shape, rate = prior.eta.rate)
        R <- sqrt(distance(t(t(XN)/eta)))
        S <- S + matern.kernel(R, nu=nu)/1000
      }
      eig.out <- eigen(S)
      M <- which(cumsum(eig.out$value)/sum(eig.out$value)>fraction)[1]
      basis <- eig.out$vectors[,1:M]
      R <- sqrt(distance(t(t(XN)/eta)))
      K <- matern.kernel(R, nu=nu)
      E <- t(basis) %*% K %*% basis
      B <- t(mvtnorm::rmvnorm(1, mean = rep(0,M), sigma = tau2*(E+diag(nug,M))))
      
      yU <- yp %*% U
      pred.out <- ys_hat.KL(klgp=klgp, B.new=B, basis=basis, U=U, emulator=emulator, figp=figp, XN=XN, Ki=Ki)
      dU <- yU - pred.out$ynew
      
      if(!is.null(seed)){
        set.seed(seed)
      }
      
      ll.Yp <- logl.Yp(pred.out, s2, dU, yp) 
      ll.B <- logl.B(eta, tau2, XN, B, basis, nu, nug)
      
      ##### MCMC sampling #####
      for(ii in 1:MC.samples){
        
        ### sample B using Elliptical Slice Sampling (ESS): this code follows the one in the `deepgp` package
        B.prior <-  t(mvtnorm::rmvnorm(1, mean = rep(0,M), sigma = tau2*(E+diag(nug,M)))) 
        
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
          # sample gN: ESS
          
          B.new <-  B * cos(a) + B.prior * sin(a)
          pred.out.new <- ys_hat.KL(klgp=klgp, B.new=B.new, basis=basis, U=U, emulator=emulator, figp=figp, XN=XN, Ki=Ki)
          dU.new <- yU - pred.out.new$ynew
          
          logL <- ll.Yp + ll.B
          
          ll.Yp.new <- logl.Yp(pred.out.new, s2, dU.new, yp)
          ll.B.new <- logl.B(eta, tau2, XN, B.new, basis, nu, nug)
          logL.new <- ll.Yp.new + ll.B.new
          
          if(ru <= exp(logL.new - logL)){
            B <- B.new
            dU <- dU.new
            ll.Yp <- ll.Yp.new
            ll.B <- ll.B.new
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
          
          logL <- ll.B + sum(dgamma(eta-eps, prior.eta.shape, prior.eta.rate, log = TRUE)) -
            sum(log(eta)) + sum(log(eta.new))
          ll.B.new <- logl.B(eta.new, tau2, XN, B, basis, nu, nug)
          logL.new <- ll.B.new + sum(dgamma(eta.new-eps, prior.eta.shape, prior.eta.rate, log = TRUE))
          
          if(runif(1) <= exp(logL.new - logL)){
            R <- sqrt(distance(t(t(XN)/eta.new)))
            K <- matern.kernel(R, nu=nu)
            E.tmp <- t(basis) %*% K %*% basis
            if(cond(E.tmp)<1e14){
              eta <- eta.new
              E <- E.tmp
              ll.B <- ll.B.new
            }
          }
        }
        eta.sample[ii,] <- eta
        
        # sample tau2
        tau2.shape <- prior.tau2.shape + M/2
        tau2.rate <- prior.tau2.rate + drop(t(B) %*% solve(E+diag(nug,M)) %*% B)/2
        tau2 <- 1/rgamma(1, tau2.shape, tau2.rate)
        tau2.sample[ii] <- tau2
        ll.B <- logl.B(eta, tau2, XN, B, basis, nu, nug)
        g.sample[ii,] <- drop(basis %*% B)
        yhat[ii,] <- drop(U %*% diag(sqrt(pred.out$s2)) %*% t(U)) %*% rnorm(m) + pred.out$mean 
        
      }
      # At the end of each chain, return the necessary results
      
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
      
      eta <- eta.sample[1,] <- rep(rgamma(1, shape = prior.eta.shape, rate = prior.eta.rate), d)
      tau2 <- tau2.sample[1] <- 1/rgamma(1, shape = prior.tau2.shape, rate = prior.tau2.rate)
      s2 <- s2.sample[1] <- 1/rgamma(1, shape = prior.s2.shape, rate = prior.s2.rate)
      
      
      S <- 0
      for(i in 1:1000){
        eta <- rgamma(2, shape = prior.eta.shape, rate = prior.eta.rate)
        R <- sqrt(distance(t(t(XN)/eta)))
        S <- S + matern.kernel(R, nu=nu)/1000
      }
      eig.out <- eigen(S)
      M <- which(cumsum(eig.out$value)/sum(eig.out$value)>fraction)[1]
      basis <- eig.out$vectors[,1:M]
      R <- sqrt(distance(t(t(XN)/eta)))
      K <- matern.kernel(R, nu=nu)
      E <- t(basis) %*% K %*% basis
      B <- t(mvtnorm::rmvnorm(1, mean = rep(0,M), sigma = tau2*(E+diag(nug,M))))
      
      yU <- yp %*% U
      pred.out <- ys_hat.KL(klgp=klgp, B.new=B, basis=basis, U=U, emulator=emulator, figp=figp, XN=XN, Ki=Ki)
      dU <- yU - pred.out$ynew
      
      if(!is.null(seed)){
        set.seed(seed)
      }
      
      ll.Yp <- logl.Yp(pred.out, s2, dU, yp) 
      ll.B <- logl.B(eta, tau2, XN, B, basis, nu, nug)
      
      ##### MCMC sampling #####
      for(ii in 1:MC.samples){
        
        ### sample B using Elliptical Slice Sampling (ESS): this code follows the one in the `deepgp` package
        B.prior <-  t(mvtnorm::rmvnorm(1, mean = rep(0,M), sigma = tau2*(E+diag(nug,M)))) 
        
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
          # sample gN: ESS
          
          B.new <-  B * cos(a) + B.prior * sin(a)
          pred.out.new <- ys_hat.KL(klgp=klgp, B.new=B.new, basis=basis, U=U, emulator=emulator, figp=figp, XN=XN, Ki=Ki)
          dU.new <- yU - pred.out.new$ynew
          
          logL <- ll.Yp + ll.B
          
          ll.Yp.new <- logl.Yp(pred.out.new, s2, dU.new, yp)
          ll.B.new <- logl.B(eta, tau2, XN, B.new, basis, nu, nug)
          logL.new <- ll.Yp.new + ll.B.new
          
          if(ru <= exp(logL.new - logL)){
            B <- B.new
            dU <- dU.new
            ll.Yp <- ll.Yp.new
            ll.B <- ll.B.new
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
          
          logL <- ll.B + sum(dgamma(eta-eps, prior.eta.shape, prior.eta.rate, log = TRUE)) -
            sum(log(eta)) + sum(log(eta.new))
          ll.B.new <- logl.B(eta.new, tau2, XN, B, basis, nu, nug)
          logL.new <- ll.B.new + sum(dgamma(eta.new-eps, prior.eta.shape, prior.eta.rate, log = TRUE))
          
          if(runif(1) <= exp(logL.new - logL)){
            R <- sqrt(distance(t(t(XN)/eta.new)))
            K <- matern.kernel(R, nu=nu)
            E.tmp <- t(basis) %*% K %*% basis
            if(cond(E.tmp)<1e14){
              eta <- eta.new
              E <- E.tmp
              ll.B <- ll.B.new
            }
          }
        }
        eta.sample[ii,] <- eta
        
        
        # sample tau2
        tau2.shape <- prior.tau2.shape + M/2
        tau2.rate <- prior.tau2.rate + drop(t(B) %*% solve(E+diag(nug,M)) %*% B)/2
        tau2 <- 1/rgamma(1, tau2.shape, tau2.rate)
        tau2.sample[ii] <- tau2
        ll.B <- logl.B(eta, tau2, XN, B, basis, nu, nug)
        g.sample[ii,] <- drop(basis %*% B)
        yhat[ii,] <- drop(U %*% diag(sqrt(pred.out$s2)) %*% t(U) %*% rnorm(m) + pred.out$mean) 

      }
      g.ls[[jj]] <- g.sample
      yhat.ls[[jj]] <- yhat
      tau2.ls[[jj]] <- tau2.sample
      eta.ls[[jj]] <- eta.sample
      s2.ls[[jj]] <- s2.sample
    }
  }
  
  g.ls <- lapply(g.ls, FUN=function(x) x[(MC.burnin+1):MC.samples, ])
  yhat.ls <- lapply(yhat.ls, FUN=function(x) x[(MC.burnin+1):MC.samples, ])
  tau2.ls <- lapply(tau2.ls, FUN=function(x) x[(MC.burnin+1):MC.samples])
  eta.ls <- lapply(eta.ls, FUN=function(x) x[(MC.burnin+1):MC.samples, ])
  s2.ls <- lapply(s2.ls, FUN=function(x) x[(MC.burnin+1):MC.samples])
  
  return(list(g.inverse=g.ls, yhat=yhat.ls, tau2=tau2.ls, eta=eta.ls, s2=s2.ls))
}
