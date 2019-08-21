rm(list=ls())
library(rpart)
library(VGAM)
library(ada)
################################# data ##################################
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
iter = 10
try.vec = tr.y ; trx.mat = tr.x ; tex.mat = te.x
################################## Logit boost #############################
my.logitb.fun = function(try.vec,trx.mat,tex.mat,iter){
  n = length(try.vec); y.vec = try.vec; ys.vec = (y.vec+1)/2
  tr.xy.df = data.frame(try.vec,trx.mat); te.xy.df = data.frame(tex.mat)
  Fx = 0; pFx = 0; p.vec = rep(0.5,n)
  #train and test
  for(i in 1:iter){
    w.vec = pmax(p.vec*(1-p.vec),1e-24); z.vec = (ys.vec-p.vec)/w.vec
    #h = lm(z.vec~.,data = trx.mat,weights = w.vec)
    h = rpart(z.vec~.,data = trx.mat,method="anova",weights = w.vec,
              control = rpart.control(minsplit = 10,cp=0,xval = 5,maxdepth = 1))
    fx = predict(h,trx.mat); pfx = predict(h,tex.mat)
    pFx = pFx+0.5*pfx; Fx = Fx+0.5*fx
    ef = exp(Fx); eff = (exp(Fx)+exp(-Fx))
    ef[ef==Inf] = 1e+7; ef[ef==0] = 1e-07
    eff[eff==Inf] = 1e+07; eff[eff==0] = 1e-07
    p.vec = ef/eff
  }
  train = ifelse(Fx>0,1,0)
  test = ifelse(pFx>0,1,0)
  return(list(train=train,test=test))
}
v = my.logitb.fun(tr.y,tr.x,te.x,iter)
mean(v$train==train$admit)
mean(v$test==test$admit)


#using vglm
fit.glm = glm(admit~.,data = train[,-5],family = "binomial")
tr.pred.glm = predict(fit.glm,train[,-5],type = "response")
te.pred.glm = predict(fit.glm,test[,-5],type = "response")
mean(ifelse(tr.pred.glm>0.5,1,0)==train$admit)
mean(ifelse(te.pred.glm>0.5,1,0)==test$admit)


xy.df = data.frame(tr.y,tr.x)
ada=ada(tr.y~.,data=xy.df,loss="logistic",iter = iter,type="real",rpart.control(maxdepth = 1),bag.frac=1)
nxy.df = data.frame(te.y,te.x)
mean(tr.y==predict(ada,xy.df))
mean(te.y==predict(ada,nxy.df))
mean(predict(ada,nxy.df)==ifelse(v$test==1,1,-1))


