rm(list=ls())
library(rpart)
library(adabag)
library(ada)
################################# data ###################################
mydata = read.csv("https://stats.idre.ucla.edu/stat/data/binary.csv")
mydata[,4] = as.factor(mydata[,4])
train = mydata[1:200,]
test = mydata[201:400,]
train$admit2[train$admit==1] = 1
train$admit2[train$admit==0] = -1
test$admit2[test$admit==1] = 1
test$admit2[test$admit==0] = -1

tr.y = train[,5]
tr.x = train[,-c(1,5)]
te.y = test[,5]
te.x = test[-c(1,5)]
iter = 1
########################## Real AdaBoost ###########################
my.realada.fun = function(try.vec,trx.mat,tex.mat,iter){
  n = length(try.vec); iter = iter; w.vec = rep(1/n,n)
  f.mat = matrix(NA,iter,n); tef.mat = f.mat = matrix(NA,iter,n)
  tr.xy.df = data.frame(try.vec,trx.mat); te.xy.df = data.frame(tex.mat)
  #real adaboost
  for(i in 1:iter){
    h = rpart(try.vec~.,data = tr.xy.df,method="class",weights = w.vec,maxdepth=5)
    tr.p = as.numeric(predict(h,tr.xy.df,type="prob")[,2])
    te.p = as.numeric(predict(h,te.xy.df,type="prob")[,2])
    f.vec = 0.5*log(tr.p/(1-tr.p)); tef.vec = 0.5*log(te.p/(1-te.p))
    f.vec[f.vec==-Inf] = -1e+7; f.vec[f.vec==Inf] = 1e+7
    tef.vec[tef.vec==-Inf] = -1e+7; tef.vec[tef.vec==Inf] = 1e+7
    w.vec = w.vec*exp(-try.vec*f.vec); w.vec = w.vec/sum(w.vec)
    w.vec[w.vec==0] = 1e-7; w.vec[w.vec==Inf] = 1e+7
    #train classifier
    f.mat[i,] = f.vec
    #test classifier
    tef.mat[i,] = tef.vec
  }
  train = ifelse(colSums(f.mat)>0,1,0)
  test = ifelse(colSums(tef.mat)>0,1,0)
  return(list(train=train,test=test))
}

v = my.realada.fun(tr.y,tr.x,te.x,iter)
#train acc
mean(train$admit==v$train)
#test acc
mean(test$admit==v$test)


xy.df = data.frame(tr.y,tr.x)
ada=ada(tr.y~.,data=xy.df,loss="exponential",iter = 1,type="real",rpart.control(maxdepth = 5),bag.frac=1)
nxy.df = data.frame(te.y,te.x)
mean(tr.y==predict(ada,xy.df))
mean(te.y==predict(ada,nxy.df))
mean(predict(ada,nxy.df)==ifelse(v$test>0,1,-1))
