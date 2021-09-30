QRp<-function(X,Y,T,nu,solver){
# Quantile regression

LARGE<-1000000
n<-dim(Y)[1]
d<-dim(Y)[2]
r<-dim(X)[2]
m<-length(T)
U<-matrix(c(T[-1],1),m,1)
D <-diag(1,m);
for (i in 1:(m-1)) {D[i+1,i] <- (-1)}
mu <- D%*%U;
xbar <- t(nu)%*%X;

if ((n != dim(X)[1]) |( d != dim(U)[2] )) {stop("wrong dimensions")}
xbar <- t(nu)%*% X

############### VECTORIZATION OF THE MATRICES FOR LP ###############
c <- -kronecker(Y,mu)
A <- kronecker(t(X),sparseMatrix(1:m,1:m))
d <- matrix((matrix(1,m,1)-T)%*%xbar,nrow=m)
e <- matrix( matrix(1,m,1)%*% t(nu),nrow=m)

z_init <- matrix((matrix(1,m,1)-T)%*%t(nu),nrow=m)


############### LP SOLVING PHASE       ##############################
if (solver == 'gurobi')
{
model <- list()

model$A           <- A
model$obj         <- c
model$modelsense  <- "min"
model$rhs         <- d
model$ub          <- e
model$sense       <- '='
model$start		<- z_init
result		<- gurobi ( model, params=NULL ) 
if (result$status=="OPTIMAL") {Zvec <- result$x; L1vec <- t(result$pi); L2vec <- -t(Zvec>0) * t(result$rc) } else {stop("optimization problem with Gurobi")}

}
else
{
stop("Using MATLAB solver needs to be added")
}
#############################################

Z		<- matrix(Zvec,nrow=m)
L1		<- matrix(L1vec,nrow=m)
L2		<- matrix(L2vec,nrow=m)
beta		<- -diag(c(1/mu))%*% L1
P		<- -t(L2) %*% diag(c(1/mu))
val		<- t(mu) %*% Z %*% Y

 return(list(Z=Z,P=P,beta=beta, val=val))
}


