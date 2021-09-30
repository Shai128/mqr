source(paste0(getwd(),"/QuantileRegression/VQR/init_VQR.R"))
inputs<- initMKQR("GDPDebtDeficit")
X<-inputs$X;Y<-inputs$Y;n<-inputs$n;d<-inputs$d;r<-inputs$r;solver<-inputs$solver 
step<-0.1

nu<-matrix(1/n,n,1)
T	<- seq(0,1,by=step)
lT	<- length(T)

if (d!=2){stop("Dimension d not supported")}
UEtAlprov		<- prepareU2D(T)
MKQRprov		<- MKQRTp(X,Y,UEtAlprov$U,UEtAlprov$mu,nu,solver)
#print(MKQRprov$b)
#betaEtAl		<- ComputeBetaEtAl2D(MKQRprov$b,T,UEtAlprov$U,MKQRprov$pi,step)
#xeval			<- matrix(c(1,2.33),1,2)
#PlotFixedx2D(xeval,betaEtAl$beta1,betaEtAl$beta2,lT)

