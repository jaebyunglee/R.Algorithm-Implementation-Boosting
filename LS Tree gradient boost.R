rm(list=ls())
library(rpart)
library(gbm)
set.seed(12)
n = 10000; p = 30
x.mat = matrix(rnorm(n*p),n,p)
y.vec = rnorm(n)

iter = 1:15 
loss.mat = matrix(NA,length(iter),2)

trid = sample(1:n)[1:(n*0.7)]
trx.mat = x.mat[trid,]
try.vec = y.vec[trid]
tex.mat = x.mat[-trid,]
tey.vec = y.vec[-trid]


xy.df = data.frame(try.vec,trx.mat)
nxy.df = data.frame(tey.vec,tex.mat)
F0.mat = matrix(NA,length(iter),length(try.vec))
F1.mat = matrix(NA,length(iter),length(tey.vec))

### LS boost
for(j in iter){
  print(j)
  F0 = mean(try.vec)
  F1 = mean(try.vec)
  for(i in 1:j){
    res = try.vec - F0
    data = data.frame(res,trx.mat)
    te.data = data.frame(tey.vec,tex.mat)
    fit = rpart(res~.,data=data,method = "anova",control = rpart.control(minsplit = 10,cp=0,xval=5,maxdepth = 1))
    p=predict(fit,te.data)
    h=predict(fit,data)
    F1 = F1 + p
    F0 = F0 + h
  }
  F0.mat[j,] = F0
  F1.mat[j,] = F1
  loss.mat[j,1] = sum((try.vec-F0)^2)
  loss.mat[j,2] = sum((tey.vec-F1)^2)
}
colnames(loss.mat) = c("tr.loss","te.loss")
rownames(loss.mat) = 1:length(iter)
loss.mat



###gbm package
gbm=gbm(try.vec~.,data=xy.df,distribution = "gaussian", shrinkage=1,interaction.depth = 1,n.trees=300,bag.frac=1)
gbm.loss.mat = matrix(NA,length(iter),2)
for(i in 1:length(iter)){
  gbm.loss.mat[i,1]=sum((predict(gbm,xy.df,n.trees = i)-try.vec)^2)
  gbm.loss.mat[i,2]=sum((tey.vec-predict(gbm,nxy.df,n.trees = i))^2)
}
colnames(gbm.loss.mat) = c("tr.loss","te.loss")
rownames(gbm.loss.mat) = 1:length(iter)
gbm.loss.mat


###linear regression
lm.fit = lm(try.vec~.,data=xy.df)
lm.loss.vec = c(sum((try.vec-predict(lm.fit,xy.df))^2),sum((tey.vec-predict(lm.fit,nxy.df))^2))
names(lm.loss.vec) = c("tr.loss","te.loss")
lm.loss.vec



#plot
par(mfrow=c(1,3))
plot(tey.vec,predict(lm.fit,nxy.df))
plot(tey.vec,F1.mat[which.min(loss.mat[,2]),])
plot(tey.vec,predict(gbm,nxy.df,which.min(gbm.loss.mat[,2])))



### my.lst.gb.fun
my.lst.gb.fun = function(try.vec,trx.mat,tex.mat,iter){
  F0 = mean(try.vec)
  F1 = mean(try.vec)
  for(i in 1:iter){
    res = try.vec - F0
    data = data.frame(res,trx.mat)
    te.data = data.frame(tex.mat)
    fit = rpart(res~.,data=data,method = "anova",control = rpart.control(minsplit = 10,cp=0,xval=5,maxdepth = 1))
    p=predict(fit,te.data)
    h=predict(fit,data)
    F1 = F1 + p
    F0 = F0 + h
  }
  return(list(test=F1))
}
v=my.lst.gb.fun(try.vec,trx.mat,tex.mat,10)

sum((v$test-tey.vec)^2)

########################################### stochastic
### my.lst.s.gb.fun
my.lst.s.gb.fun = function(try.vec,trx.mat,tex.mat,iter,k = 0.7,nu = 0.05){
  len = length(try.vec)
  F0 = rep(mean(try.vec),len)
  F1 = mean(try.vec)
  for(i in 1:iter){
    trid = sample(1:len)[1:(len*k)]
    trid = sort(trid)
    ttry.vec = try.vec[trid]
    res = ttry.vec - F0[trid]
    data = data.frame(res,trx.mat[trid,])
    data1 = data.frame(trx.mat)
    te.data = data.frame(tex.mat)
    fit = rpart(res~.,data=data,method = "anova",control = rpart.control(minsplit = 10,cp=0,xval=5,maxdepth = 1))
    p=predict(fit,te.data)
    h=predict(fit,data1)
    F1 = F1 + nu*p
    F0 = F0 + nu*h
  }
  return(list(pred=F1))
}

v=my.lst.s.gb.fun(try.vec,trx.mat,tex.mat,250,k = 0.7,nu = 0.005)
sum((v$pred-tey.vec)^2)

