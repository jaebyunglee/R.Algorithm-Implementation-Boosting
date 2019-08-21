rm(list=ls())
library(rpart)
library(adabag)
library(caTools)
################################# data ##################################
bcw.data = read.table("C:\\Users\\kis91\\Desktop\\bcw.data.txt",sep=",",header=F)
bcw.data = bcw.data[,-1]
bcw.data$V11[bcw.data$V11==2] = -1
bcw.data$V11[bcw.data$V11==4] = 1
bcw.data = data.frame(apply(bcw.data[,-10],2,as.numeric),bcw.data$V11)


train = bcw.data[1:499,]
train = na.omit(train) 
test = bcw.data[500:699,]
test = na.omit(test)
tr.y = train$bcw.data.V11
tr.x = train[,-10]
te.y = test$bcw.data.V11
te.x = test[,-10]
iter = 100
################################### my gentle adaboost funtion#######################################
my.gentleb.fun = function(try.vec,trx.mat,tex.mat,iter){
  n = length(try.vec); ys.vec = (try.vec+1)/2
  w.vec = rep(1/n,n); Fx = 0; pFx = 0
  tr.xy.df = data.frame(try.vec,trx.mat)
  te.xy.df = data.frame(tex.mat)
  #train & test
  for(i in 1:iter){
    h = rpart(try.vec~.,data = tr.xy.df,method="anova",weights = w.vec,
              control = rpart.control(minsplit = 10,cp=0,xval = 5,maxdepth = 1))
    #h = lm(try.vec~.,data = tr.xy.df,weights = w.vec)
    fx = predict(h,tr.xy.df); pfx = predict(h,te.xy.df)
    Fx = Fx + fx; pFx = pFx + pfx
    w.vec = w.vec*exp(-try.vec*fx)/sum(w.vec*exp(-try.vec*fx))
  }
  #train strong classifier
  train = ifelse(Fx>=0,1,0)
  #test strong classifier
  test = ifelse(pFx>=0,1,0)
  return(list(train=train,test=test))
}

v = my.gentleb.fun(tr.y,tr.x,te.x,iter)

#train acc
mean(v$train==ifelse(tr.y==1,1,0))
mean(v$test==ifelse(te.y==1,1,0))



xy.df = data.frame(tr.y,tr.x)
ada=ada(tr.y~.,data=xy.df,loss="exponential",iter = iter,type="gentle",rpart.control(maxdepth = 1),bag.frac=1)
nxy.df = data.frame(te.y,te.x)
mean(tr.y==predict(ada,xy.df))
mean(te.y==predict(ada,nxy.df))
mean(ifelse(predict(ada,nxy.df)==1,1,0)==v$test)
my.sen.spc = sum(te.y==1&v$test==1)/sum(te.y==1)+sum(te.y==-1&v$test==0)/sum(te.y==-1)
ada.sen.spc = sum(te.y==1&predict(ada,nxy.df)==1)/sum(te.y==1)+sum(te.y==-1&predict(ada,nxy.df)==-1)/sum(te.y==-1)

