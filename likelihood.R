# pdf of inverse gamma
dinvgamma <- function(x, alpha, beta) {
  beta^alpha / gamma(alpha) * x^(-alpha-1) * exp(-beta/x)
}

# log likelihood of gN
logl.gN <- function(eta, tau2, X, Y, nu, nug) 
{
  n <- length(Y)
  R <- sqrt(distance(t(t(X)/eta)))
  K <- matern.kernel(R, nu=nu)
  Ki <- solve(K+diag(nug,n))
  ldetK <- determinant(K, logarithm=TRUE)$modulus
  
  ll <- - (n/2)*log(tau2) - (1/2)*ldetK - (1/2)*(t(Y) %*% Ki %*% Y)/tau2
  return(ll)
}

# log likelihood of yp
logl.Yp <- function(pred.out, s2, dU, yp){
  mul <- pred.out$mean
  vl <- pred.out$s2
  part1 <- -0.5 * (sum(log(1+vl/s2))+log(s2)*length(yp)) 
  part2 <- - 0.5/s2 *(sum((yp-mul)^2)-sum(dU^2*(vl/(s2+vl))))
  return(part1+part2)
}