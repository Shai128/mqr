export<-function(destfile,n,d,r,X0,Y){
write(c(n,d,r-1,X0,Y),file=destfile)
}