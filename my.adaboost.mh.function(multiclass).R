rm(list=ls())
#library(adabag)
#library(rpart)
#library(randomForest)
#library(VGAM)
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
iter = 20
################################## Adaboost.mh #############################

my.adamh.fun = function(try.vec,trx.mat,tex.mat,iter){
  n = length(try.vec); nlevels = length(levels(tr.y))
  tr.xy.df = data.frame(try.vec,trx.mat);te.xy.df = data.frame(tex.mat)
  tr.y.mat = matrix(NA,n,nlevels);te.y.mat = matrix(NA,nrow(tex.mat),nlevels)
  e.array = array(NA,c(iter,n,nlevels)); pred.array = array(NA,c(iter,nrow(tex.mat),nlevels))
  for(j in 1:nlevels){
    w.vec = rep(1/n,n)
    tr.y.mat[try.vec==levels(try.vec)[j],j] = 1 
    tr.y.mat[try.vec!=levels(try.vec)[j],j] = -1 
    tr.xy.df$try.vec = tr.y.mat[,j]
    #real adaboost
    for(i in 1:iter){
      h = rpart(try.vec~.,data = tr.xy.df,method="class",weights = w.vec,maxdepth=5)
      tr.p = as.numeric(predict(h,tr.xy.df,type="prob")[,2])
      te.p = as.numeric(predict(h,te.xy.df,type="prob")[,2])
      f.vec = 0.5*log(tr.p/(1-tr.p)); tef.vec = 0.5*log(te.p/(1-te.p))
      f.vec[f.vec==-Inf] = -1e+7; f.vec[f.vec==Inf] = 1e+7
      tef.vec[tef.vec==-Inf] = -1e+7; tef.vec[tef.vec==Inf] = 1e+7
      w.vec = w.vec*exp(-tr.xy.df$try.vec*f.vec)
      w.vec[w.vec==0] = 1e-7; w.vec[w.vec==Inf] = 1e+7
      #train classifier
      e.array[i,,j] = f.vec; pred.array[i,,j] = tef.vec
    }
  }
  C.mat = t(colSums(e.array)); p.C.mat = t(colSums(pred.array))
  Cx = apply(C.mat,2,which.max); p.Cx = apply(p.C.mat,2,which.max)
  return(list(train=Cx,test=p.Cx))
}


v = my.adamh.fun(tr.y,tr.x,te.x,30)
#train acc
mean(tr.y==v$train)
mean(test$V10==v$test)
