library(numDeriv)
library(coda)
library(parallel)
library(randtoolbox)
library(cubature)
library(plgp)
library(pracma)
source("FIGP.R")                # FIGP 
source("matern.kernel.R")       # matern kernel computation
source("FIGP.kernel.R")         # kernels for FIGP
source("loocv.R")               # LOOCV for FIGP
source("likelihood.R")
source("ys_emulators.R")
source("BayesInverse_FIGP.R")
source("FIGP.kernel.discrete.R")
source("KL.expan.R")
source("GP.R")
source("BayesInverse_KL.R")

# nugget for numerical stability
eps <- 1e-6

# computing predictive scores; the higher the better 
score <- function(x, mu, s2){   
  -(x-mu)^2/s2-log(s2)
}

# number of chain for MCMC
nchain <- 5

# grid points spanning the range [0,2pi]
s1 <- seq(0, 2*pi, length.out = 32)
s2 <- seq(0, 2*pi, length.out = 32)

# double integral function
double_integral <- function(g){
  h <- function(x,y) apply(cbind(x,y), 1, g)
  inner_integral <- function(x) integrate(function(y) h(x, y), lower = 0, upper = 1)$value
  result <- integrate(Vectorize(inner_integral), lower = 0, upper = 1)
  return(result$value)
}

# testing function: high-fidelity function
f <- function(g, s1, s2){
  a0 <- double_integral(function(z) g(z))
  a1 <- double_integral(function(z) grad(g,z)[1])
  a2 <- double_integral(function(z) -grad(g,z)[2])

  return(a0 + a1*sin(s1) + a2*cos(s2))
}

# testing function: low-fidelity function
fl <- function(g, s1, s2){
  a0 <- double_integral(function(z) g(z))
  a1 <- double_integral(function(z) grad(g,z)[1])
  
  return(a0 + a1*sin(s1))
}

# training functional inputs (G)
G <- list(function(x) 1+x[1],
          function(x) 1-x[1],
          function(x) 1+x[1]*x[2],
          function(x) 1-x[1]*x[2],
          function(x) 1+x[2],
          function(x) 1-x[2],
          function(x) 1+x[1]^2,
          function(x) 1-x[1]^2,
          function(x) 1+x[2]^2,
          function(x) 1-x[2]^2)
n <- length(G)

func.title <- c(expression(g==1+x[1]), 
                expression(g==1-x[1]),
                expression(g==1+x[1]*x[2]),
                expression(g==1-x[1]*x[2]),
                expression(g==1+x[2]),
                expression(g==1-x[2]),
                expression(g==1+x[1]^2),
                expression(g==1-x[1]^2),
                expression(g==1+x[2]^2),
                expression(g==1-x[2]^2))    

# running the training data
Ys <- Yb <- matrix(0,nrow=10,ncol=32*32)
for(i in 1:10)  {
  Ys[i,] <- c(outer(s1, s2, function(s1, s2) f(G[[i]],s1,s2)))
  Yb[i,] <- c(outer(s1, s2, function(s1, s2) fl(G[[i]],s1,s2)))
} 


# PCA to reduce output dimension
pca.out <- prcomp(Ys, scale = FALSE, center = FALSE)
n.comp <- which(summary(pca.out)$importance[3,] > 0.9999)[1]
U <- pca.out$rotation[,1:n.comp] # eigenvectors

# simulation setting
N.vt <- seq(50,150,50)  # choice of N
sig2 <- 0.005^2         # yp variance
beta1 <- runif(10,-2,2) # random coefficients for test input g
beta2 <- runif(10,-2,2) # random coefficients for test input g

comp.time <- rmse.result <- score.result <- rmse.y.result <- score.y.result <- matrix(0, ncol=length(N.vt), nrow=length(beta1))
comp.KL.time <- rmse.KL.result <- score.KL.result <- rmse.y.KL.result <- score.y.KL.result <- 
  comp.KLKL.time <- rmse.KLKL.result <- score.KLKL.result <- rmse.y.KLKL.result <- score.y.KLKL.result <- 
  comp.multi.time <- rmse.multi.result <- score.multi.result <- rmse.y.multi.result <- score.y.multi.result <- rep(0, length(beta1))

for(jjj in 1:length(beta1)){
  
  # generate test functional inputs (gnew) and output 
  gnew <- list(function(x) 1+beta1[jjj]*x[1]+beta2[jjj]*x[2])
  ys.true <- c(outer(s1, s2, function(s1, s2) f(gnew[[1]],s1,s2)))
  m <- length(ys.true)
  
  # simulate yp
  set.seed(jjj*123)       # for reproducibility
  yp <- ys.true + rnorm(m, 0, sd=sqrt(sig2))
  X.grid <- expand.grid(seq(0,1,0.1),seq(0,1,0.1)) # GP test locations (for visualization of inverse g)
  n.grid <- nrow(X.grid)
  g.true <- gnew[[1]](X.grid)[,1]
  
  ### single-fidelity
  for(iii in 1L:length(N.vt)){
    N <- N.vt[iii]
    
    time.start <- proc.time()[3]
    fem.figp <- gp.fit <- gpnl.fit <- vector("list",n.comp)
    for(i in 1:n.comp){
      y <- Ys %*% U[,i]
      # fit FIGP with a linear kernel  
      gp.fit[[i]] <- FIGP(G, d=2, y, nu=2.5, nug=eps, kernel = "linear", rnd = N)
      # fit FIGP with a nonlinear kernel    
      gpnl.fit[[i]] <- FIGP(G, d=2, y, nu=2.5, nug=eps, kernel = "nonlinear", rnd = N)
    }
    # computing LOOCVs
    loocv.linear <- sapply(gp.fit, loocv)
    loocv.nonlinear <- sapply(gpnl.fit, loocv)
    time.end <- proc.time()[3]
    # emulation computational cost
    time.emulation <- difftime(time.end, time.start, units = "secs")
    
    # loocv to determine linear or nonlinear kernel
    for(i in 1:n.comp){
      if(loocv.linear[i] < loocv.nonlinear[i]) {
        fem.figp[[i]] <- gp.fit[[i]]
      }else {
        fem.figp[[i]] <- gpnl.fit[[i]]
      }
    }
    
    XN <- sobol(N, 2) # for gN realizations
    
    # posterior of inverse g and ys: single fidelity
    time.start <- proc.time()[3]
    post.single <- BayesInverse_FIGP(XN, X.grid, yp, U, 
                                     nu=2.5, nug=eps,
                                     figp.1=fem.figp, figp.2=NULL, fidelity=c("single","multi")[1],
                                     MC.samples = 10000, MC.burnin = 3000, nchain = nchain,
                                     parallel=TRUE)
    time.end <- proc.time()[3]
    # add inverse computational cost
    time.single <- time.emulation + difftime(time.end, time.start, units = "secs")

    ### thinning
    g.sample <- matrix(0, nrow = 3500 * nchain, ncol=121)
    ys.sample <- matrix(0, nrow = 3500 * nchain, ncol=m)
    for(i in 1:nchain){
      g.sample[((i-1)*3500+1):(3500*i),] <- post.single$g.inverse[[i]][seq(1,7000,2),]
      ys.sample[((i-1)*3500+1):(3500*i),] <- post.single$yhat[[i]][seq(1,7000,2),]
    }
    
    g.single.mean <- apply(g.sample,2,mean)
    g.single.var <- apply(g.sample,2,var)
    ys.single.mean <- apply(ys.sample,2,mean)
    ys.single.var <- apply(ys.sample,2,var)
    
    # single-fidelity results with N.vt[iii]
    comp.time[jjj, iii] <- time.single
    rmse.result[jjj, iii] <- sqrt(mean((g.single.mean - g.true)^2))
    score.result[jjj, iii] <- mean(score(g.true, mu = g.single.mean, s2 = g.single.var))
    rmse.y.result[jjj, iii] <- sqrt(mean((ys.single.mean - ys.true)^2))
    score.y.result[jjj, iii] <- mean(score(ys.true, mu = ys.single.mean, s2 = ys.single.var))
  }
  
  
  ### multi-fidelity
  N <- 100
  XN <- randtoolbox::sobol(N, 2) # for gN realizations
  # fit FIGP for born and diff
  time.start <- proc.time()[3]
  born.gp.fit <- born.gpnl.fit <- diff.gp.fit <- diff.gpnl.fit <- vector("list",n.comp)
  for(i in 1:n.comp){
    yb <- Yb %*% U[,i]
    # fit FIGP with a linear kernel  
    born.gp.fit[[i]] <- FIGP(G, d=2, yb, nu=2.5, nug=eps, kernel = "linear", rnd = N)
    # fit FIGP with a nonlinear kernel  
    born.gpnl.fit[[i]] <- FIGP(G, d=2, yb, nu=2.5, nug=eps, kernel = "nonlinear", rnd = N)
    
    ys <- Ys %*% U[,i]
    # fit FIGP with a linear kernel  
    diff.gp.fit[[i]] <- MF.FIGP(G, d=2, ys, yb, nu=2.5, nug=eps, kernel = "linear", rnd = N)
    # fit FIGP with a nonlinear kernel  
    diff.gpnl.fit[[i]] <- MF.FIGP(G, d=2, ys, yb, nu=2.5, nug=eps, kernel = "nonlinear", rnd = N)
  }
  
  # computing LOOCVs
  born.loocv.linear <- sapply(born.gp.fit, loocv)
  born.loocv.nonlinear <- sapply(born.gpnl.fit, loocv)
  
  diff.loocv.linear <- sapply(diff.gp.fit, loocv)
  diff.loocv.nonlinear <- sapply(diff.gpnl.fit, loocv)
  
  # loocv to determine linear or nonlinear kernel
  born.figp <- born.gpnl.fit
  diff.figp <- diff.gpnl.fit
  for(i in 1:n.comp){
    if(born.loocv.linear[i] < born.loocv.nonlinear[i]) {
      born.figp[[i]] <- born.gp.fit[[i]]
    }
    if(diff.loocv.linear[i] < diff.loocv.nonlinear[i]) {
      diff.figp[[i]] <- diff.gp.fit[[i]]
    }
  }

  # estimated rho
  rho.hat <- sapply(diff.figp, function(x) x$rho)
  
  # posterior of inverse g and ys: multifidelity
  post.multi <- BayesInverse_FIGP(XN, X.grid, yp, U, 
                                  nu=2.5, nug=eps,
                                  figp.1=born.figp, figp.2=diff.gp.fit, fidelity=c("single","multi")[2],
                                  MC.samples = 10000, MC.burnin = 3000, nchain = nchain,
                                  parallel= TRUE)
  time.end <- proc.time()[3]
  # add inverse computational cost
  time.multi <- difftime(time.end, time.start, units = "secs")
  
  ### thinning
  g.sample <- matrix(0, nrow = 3500 * nchain, ncol=121)
  ys.sample <- matrix(0, nrow = 3500 * nchain, ncol=m)
  for(i in 1:nchain){
    g.sample[((i-1)*3500+1):(3500*i),] <- post.multi$g.inverse[[i]][seq(1,7000,2),] 
    ys.sample[((i-1)*3500+1):(3500*i),] <- post.multi$yhat[[i]][seq(1,7000,2),]
  }
  
  g.multi.mean <- apply(g.sample,2,mean)
  g.multi.var <- apply(g.sample,2,var)
  ys.multi.mean <- apply(ys.sample,2,mean)
  ys.multi.var <- apply(ys.sample,2,var)
  
  # multifidelity results
  comp.multi.time[jjj] <- time.multi
  rmse.multi.result[jjj] <- sqrt(mean((g.multi.mean - g.true)^2))
  score.multi.result[jjj] <- mean(score(g.true, mu = g.multi.mean, s2 = g.multi.var))
  rmse.y.multi.result[jjj] <- sqrt(mean((ys.multi.mean - ys.true)^2))
  score.y.multi.result[jjj] <- mean(score(ys.true, mu = ys.multi.mean, s2 = ys.multi.var))
  
  ### FIGP + KL
  N <- 100
  XN <- randtoolbox::sobol(N, 2) # for gN realizations
  fem.figp <- gp.fit <- gpnl.fit <- vector("list",n.comp)
  for(i in 1:n.comp){
    y <- Ys %*% U[,i]
    # fit FIGP with a linear kernel  
    gp.fit[[i]] <- FIGP(G, d=2, y, nu=2.5, nug=eps, kernel = "linear", rnd = N)
    # fit FIGP with a nonlinear kernel    
    gpnl.fit[[i]] <- FIGP(G, d=2, y, nu=2.5, nug=eps, kernel = "nonlinear", rnd = N)
  }

  loocv.linear <- sapply(gp.fit, loocv)
  loocv.nonlinear <- sapply(gpnl.fit, loocv)
  time.end <- proc.time()[3]
  # emulation computational cost
  time.emulation <- difftime(time.end, time.start, units = "secs")

  # loocv to determine linear or nonlinear kernel
  for(i in 1:n.comp){
    if(loocv.linear[i] < loocv.nonlinear[i]) {
      fem.figp[[i]] <- gp.fit[[i]]
    }else {
      fem.figp[[i]] <- gpnl.fit[[i]]
    }
  }
  
  # posterior of inverse g and ys: FIGP+KL
  time.start <- proc.time()[3]
  post.KL <- BayesInverse_KL(klgp=NULL, X.grid, X.grid, yp, U, fraction=0.95, emulator="figp", figp=fem.figp,
                             nu=2.5, nug=eps,
                             MC.samples = 10000, MC.burnin = 3000, nchain = nchain,
                             parallel= TRUE)
  time.end <- proc.time()[3]
  
  # add inverse computational cost
  time.KL <- time.emulation + difftime(time.end, time.start, units = "secs")
  
  ### thinning
  g.sample <- matrix(0, nrow = 3500 * nchain, ncol=121)
  ys.sample <- matrix(0, nrow = 3500 * nchain, ncol=m)
  for(i in 1:nchain){
    g.sample[((i-1)*3500+1):(3500*i),] <- post.KL$g.inverse[[i]][seq(1,7000,2),]
    ys.sample[((i-1)*3500+1):(3500*i),] <- post.KL$yhat[[i]][seq(1,7000,2),]
  }
  
  g.KL.mean <- apply(g.sample,2,mean)
  g.KL.var <- apply(g.sample,2,var)
  ys.KL.mean <- apply(ys.sample,2,mean)
  ys.KL.var <- apply(ys.sample,2,var)
  
  # FIGP + KL results
  comp.KL.time[jjj] <- time.KL
  rmse.KL.result[jjj] <- sqrt(mean((g.KL.mean - g.true)^2))
  score.KL.result[jjj] <- mean(score(g.true, mu = g.KL.mean, s2 = g.KL.var))
  rmse.y.KL.result[jjj] <- sqrt(mean((ys.KL.mean - ys.true)^2))
  score.y.KL.result[jjj] <- mean(score(ys.true, mu = ys.KL.mean, s2 = ys.KL.var))
  
  ### KL + KL
  time.start <- proc.time()[3]
  klgp <- KLGP.fit(d=2, Ys=Ys, G=G, U=U, fraction=0.95, XN=X.grid)
  # posterior of inverse g and ys: KL + KL
  post.KLKL <- BayesInverse_KL(klgp=klgp, X.grid, X.grid, yp, U, fraction=0.95,emulator="klgp", figp=NULL,
                             nu=2.5, nug=eps,
                             MC.samples = 10000, MC.burnin = 3000, nchain = nchain,
                             parallel= TRUE)
  time.end <- proc.time()[3]
  
  # add inverse computational cost
  time.KLKL <- difftime(time.end, time.start, units = "secs")
  
  # thinning
  g.sample <- matrix(0, nrow = 3500 * nchain, ncol=121)
  ys.sample <- matrix(0, nrow = 3500 * nchain, ncol=m)
  for(i in 1:nchain){
    g.sample[((i-1)*3500+1):(3500*i),] <- post.KLKL$g.inverse[[i]][seq(1,7000,2),]
    ys.sample[((i-1)*3500+1):(3500*i),] <- post.KLKL$yhat[[i]][seq(1,7000,2),]
  }
  
  g.KLKL.mean <- apply(g.sample,2,mean)
  g.KLKL.var <- apply(g.sample,2,var)
  ys.KLKL.mean <- apply(ys.sample,2,mean)
  ys.KLKL.var <- apply(ys.sample,2,var)
  
  # KL + KL results
  comp.KLKL.time[jjj] <- time.KLKL
  rmse.KLKL.result[jjj] <- sqrt(mean((g.KLKL.mean - g.true)^2))
  score.KLKL.result[jjj] <- mean(score(g.true, mu = g.KLKL.mean, s2 = g.KLKL.var))
  rmse.y.KLKL.result[jjj] <- sqrt(mean((ys.KLKL.mean - ys.true)^2))
  score.y.KLKL.result[jjj] <- mean(score(ys.true, mu = ys.KLKL.mean, s2 = ys.KLKL.var))
}

### summarize the results
CompTime <- cbind(comp.time, comp.multi.time, comp.KL.time, comp.KLKL.time)
RMSE.g <- cbind(rmse.result, rmse.multi.result, rmse.KL.result, rmse.KLKL.result)
Score.g <- cbind(score.result, score.multi.result, score.KL.result, score.KLKL.result)
RMSE.y <- cbind(rmse.y.result, rmse.y.multi.result, rmse.y.KL.result, rmse.y.KLKL.result)
Score.y <- cbind(score.y.result, score.y.multi.result, score.y.KL.result, score.y.KLKL.result)

colnames(CompTime) <- colnames(RMSE.g) <- colnames(Score.g) <- 
  colnames(RMSE.y) <- colnames(Score.y) <- c("Single - N=50", "Single - N=100", "Single - N=150", "Multi", "FIGP+KL", "KL+KL")

pdf("SimulationResults.pdf", width = 7, height = 6)
# Set up the layout for boxplots in a single row
par(mfrow = c(2, 3), mar = c(7.5, 4, 4, 0.5) + 0.1)  # Adjust margins for better spacing

# Boxplot for Computation Time
boxplot(CompTime, main = "Computation Time", xlab = "", ylab = "Time (s)", las = 2, col = "skyblue")

# Boxplot for RMSE of g
boxplot(RMSE.g, ylim = c(0, 0.6), main = "RMSE of g", xlab = "", ylab = "RMSE", las = 2, col = "lightgreen")

# Boxplot for Score of g
boxplot(Score.g, ylim = c(1, 8), main = "Score of g", xlab = "", ylab = "Score", las = 2, col = "lightcoral")

# Blank panel (2, 1)
plot.new()  # Creates a blank plot

# Boxplot for RMSE of y
boxplot(RMSE.y, ylim = c(0, 0.03), main = expression("RMSE of " * y^s * "(g)"), xlab = "", ylab = "RMSE", las = 2, col = "lightgoldenrod")

# Boxplot for Score of y
boxplot(Score.y, ylim = c(1, 9), main = expression("Score of " * y^s * "(g)"), xlab = "", ylab = "Score", las = 2, col = "plum")

# Reset layout
par(mfrow = c(1, 1))
dev.off()
