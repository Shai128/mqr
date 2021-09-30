source(paste0(getwd(),"/QuantileRegression/VQR/init_VQR.R"))
inputs<- init_VQR("Engel")
X<-inputs$X;Y<-inputs$Y;n<-inputs$n;d<-inputs$d;r<-inputs$r;step<-inputs$step;solver<-inputs$solver 



nu	<- matrix(1/n,n,1)
T	<- seq(0,1-step,by=step)
m	<- length(T)
U	<- matrix(T,m,1)
mu	<- matrix(1/m, m,1)


x_quantiles1	<- quantile(X[,2],probs=c(0,0.25,0.5,0.75,1))
Nb_display 		<- length(x_quantiles1)
x_quantiles 	<- cbind(matrix(1,Nb_display,1),matrix(x_quantiles1,Nb_display,1))



solsMKQR=MKQRTp( X,Y,U,mu,nu,solver ) 
beta <- ComputeBeta1D( mu,solsMKQR$b )
yhat_MKQR <- x_quantiles%*%t(beta)


solsQR	<- QRp( X,Y,T,nu,solver )
yhat_QR	<- x_quantiles %*% t(solsQR$beta)

xaxis <- T[2:(m-1)]
for (i in 2:Nb_display){ xaxis <- rbind(T[2:(m-1)],xaxis) }
plot(xaxis,yhat_MKQR[,2:(m-1)],type='n')
for(i in 1:Nb_display){lines(T[2:(m-1)],yhat_QR[i,2:(m-1)],col=rgb(0.2*i, 0,0));lines(T[2:(m-1)],yhat_MKQR[i,2:(m-1)],col=rgb(0,0.2*i,0))}



