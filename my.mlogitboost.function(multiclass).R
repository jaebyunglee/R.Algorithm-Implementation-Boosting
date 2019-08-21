rm(list=ls())
library(rpart)
library(VGAM)
cmc.data = read.table("C:\\Users\\kis91\\Dropbox\\2018 Jaebyung Lee\\cmc.data.txt",header=F,sep=",")
#cmc.data = read.table("C:\\Users\\jaebyung.lee\\Desktop\\cmc.data.txt",header=F,sep=",")
str(cmc.data)
cmc.data$V1 = as.numeric(cmc.data$V1)
cmc.data$V4 = as.numeric(cmc.data$V4)
cols <- c(names(cmc.data)[-c(1,4)])
cmc.data[,cols] <- data.frame(apply(cmc.data[,cols], 2, as.factor))
str(cmc.data)
dim(cmc.data)
train = cmc.data[1:1000,]
test = cmc.data[1001:1473,]
 
tr.y = train[,10]
tr.x = train[,-10]
te.y = test[,10]
te.x = test[,-10]
iter = 5
try.vec = tr.y ; trx.mat = tr.x ; tex.mat = te.x
###########################################################################

my.mlogitb.fun = function(try.vec,trx.mat,tex.mat,iter){
  n = length(try.vec); y.vec = try.vec; ncl = nlevels(y.vec)
  px = matrix(c(1/ncl),ncl,n)
  Fx = matrix(0,ncl,n); fx = matrix(0,ncl,n)
  pFx = matrix(0,ncl,nrow(tex.mat)); pfx = matrix(0,ncl,nrow(tex.mat))
  z.mat = matrix(NA,ncl,n); w.mat = matrix(NA,ncl,n)
 ########### Multi logit boost ############
  for(id in 1:iter){
    for(j in 1:ncl){
      z.mat[j,] = (as.numeric(y.vec==levels(y.vec)[j]) - px[j,])/(px[j,]*(1-px[j,]))
      w.mat[j,] = (px[j,]*(1-px[j,]))
      h = rpart(z.mat[j,]~.,data=trx.mat,weights=w.mat[j,],method="anova",control = rpart.control(minsplit = 10,cp=0,xval = 5,maxdepth = 1))
      #h = lm(z.mat[j,]~.,data = trx.mat,weights = w.mat[j,])
      fx[j,] = predict(h,trx.mat)
      pfx[j,] = predict(h,tex.mat)
    }
    
    g = colSums(fx)
    for(j in 1:ncl){
      fx[j,] = (ncl-1)/ncl*(fx[j,]-1/ncl*g)
      Fx[j,] = Fx[j,] + fx[j,]
      pFx[j,] = pFx[j,] + pfx[j,]
    }
    
    for(j in 1:ncl){
      px[j,] = exp(Fx[j,])/(rowSums(apply(Fx,1,exp)))
    }
  }
  #train strong classifier
  train.clf = apply(Fx,2,which.max)
  #test strong classifier
  test.clf = apply(pFx,2,which.max)
  
  return(list(train=train.clf,test=test.clf))
}



v = my.mlogitb.fun(tr.y,tr.x,te.x,10)
#train acc
mean(v$train==tr.y)
#test acc
mean(v$test==te.y)



################################## multi logistic #################################################
fit.glm = vglm(V10~.,data = train,family = "multinomial")
tr.pred.glm = predict(fit.glm,train,type = "response")
te.pred.glm = predict(fit.glm,test,type = "response")
#glm train acc
mean(apply(tr.pred.glm,1,which.max)==train$V10)

#glm test acc
mean(apply(te.pred.glm,1,which.max)==test$V10)


