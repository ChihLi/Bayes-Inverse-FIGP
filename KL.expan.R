KLGP.fit <- function(d, Ys, G, U, fraction=0.99, rnd=1e3, XN=NULL){
  
  # KL expansion that explains fraction of the variance
  KL.out <- KL.expan(d=d, G=G, fraction=fraction, rnd=rnd, XN=XN)
  B <- KL.out$B
  
  n.comp <- ncol(U)
  KL.fit <- vector("list", n.comp)
  for(i in 1:n.comp){
    y <- Ys %*% U[,i]
    KL.fit[[i]] <- sepGP(B, y, nu=2.5, nug=eps)
  }
  return(list(d=d, fraction=fraction, kl=KL.out, fit=KL.fit, XN=XN))
}

ys_hat.KL <- function(klgp=NULL, B.new, basis, U, 
                      emulator=c("figp","klgp")[2], figp=NULL, XN=NULL, Ki=NULL){
  
  B.new <- matrix(B.new, nrow=1)
  gN <- basis %*% t(B.new)
  
  if(emulator == "klgp"){
    B <- KL.Bnew(klgp$kl, gN) 
  }
  
  n.comp <- ncol(U)
  ynew <- s2new <- matrix(0,ncol=n.comp,nrow=1)
  for(i in 1:n.comp){
    if(emulator == "klgp"){
      pred.out <- pred.sepGP(klgp$fit[[i]], B)
      ynew[,i] <- drop(pred.out$mu)
      s2new[,i] <- drop(pred.out$sig2)
    }else{
      pred.out <- pred.FIGP.XN(figp[[i]], gN, XN, Ki[[i]])
      ynew[,i] <- pred.out$mu
      s2new[,i] <- pred.out$sig2
    }
  }
  
  # reconstruct the image
  pred.recon <- c(ynew %*% t(U))
  return(list(mean=pred.recon, s2=drop(s2new), ynew=drop(ynew)))
}


KL.expan <- function(d, G, fraction=0.99, rnd=1e3, XN=NULL){
  
  if(is.null(XN)){
    XN <- sobol(rnd, d)
  }else{
    rnd <- nrow(XN)
  }
  n <- length(G)
  Y <- matrix(0,ncol=n,nrow=rnd)
  for(i in 1:n) Y[,i] <- apply(XN,1,G[[i]])
  Y.center <- apply(Y, 1, mean)
  Y <- Y - Y.center
  
  R <- matrix(0,n,n)
  for(i in 1:n) {
    for(j in i:n){
      R[i,j] <- R[j,i] <- sum(Y[,i] * Y[,j])
    }
  }
  eig.out <- eigen(R)

  varphi <- Y %*% eig.out$vectors
  varphi <- t(t(varphi)/apply(varphi, 2, FUN = function(x) sqrt(sum(x^2))))   # normalization
  betai <- t(Y) %*% varphi
  
  # fraction of variance explained
  FoVE <- rep(0,n)
  for(m in 1:n) FoVE[m] <- sum((varphi[,1:m]%*%t(betai[,1:m]))^2)/sum(Y^2)
  M <- which(FoVE > fraction)[1]
  
  return(list(basis=varphi[,1:M], B=betai[,1:M], rnd=rnd,
              rndX=XN, Y.center=Y.center, M=M, FoVE=FoVE))
}

KL.Bnew <- function(KL.ls, gN){
  Y <- gN - KL.ls$Y.center
  return(t(Y) %*% KL.ls$basis)
}
