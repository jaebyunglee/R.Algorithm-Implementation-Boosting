rm(list=ls())
library(rpart)
library(gbm)
library(quantreg)
set.seed(12)
n = 1000; p = 30
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
trg.vec = rep(NA,length(try.vec))
teg.vec = rep(NA,length(tey.vec))


### LAD boost
for(j in iter){
  print(j)
  F0 = median(try.vec)
  F1 = median(try.vec)
  for(i in 1:j){
    res = try.vec - F0
    t.y.vec = sign(try.vec - F0)
    data = data.frame(t.y.vec,trx.mat)
    te.data = data.frame(tey.vec,tex.mat)
    fit = rpart(t.y.vec~.,data=data,method = "anova",control = rpart.control(minsplit = 10,cp=0,xval = 5,maxdepth = 1))
    p = predict(fit,te.data); h=predict(fit,data)
    p = round(p,digits=6); h = round(h,digits=6)
    pos = as.numeric(names(table(h))); pos = round(pos,digits=6)
    trg.vec[h==pos[1]] = median(res[h==pos[1]])
    trg.vec[h==pos[2]] = median(res[h==pos[2]])
    teg.vec[p==pos[1]] = median(res[h==pos[1]])
    teg.vec[p==pos[2]] = median(res[h==pos[2]])
    F1 = F1 + teg.vec
    F0 = F0 + trg.vec
  }
  F0.mat[j,] = F0
  F1.mat[j,] = F1
  loss.mat[j,1] = sum(abs(try.vec-F0))
  loss.mat[j,2] = sum(abs(tey.vec-F1))
}
colnames(loss.mat) = c("tr.loss","te.loss")
rownames(loss.mat) = 1:length(iter)
loss.mat


###gbm package
gbm=gbm(try.vec~.,data=xy.df,distribution = "laplace", shrinkage=1,interaction.depth = 1,n.trees=300,bag.frac=1)
gbm.loss.mat = matrix(NA,length(iter),2)
for(i in 1:length(iter)){
  gbm.loss.mat[i,1]=sum(abs(predict(gbm,xy.df,n.trees = i)-try.vec))
  gbm.loss.mat[i,2]=sum(abs(tey.vec-predict(gbm,nxy.df,n.trees = i)))
}
colnames(gbm.loss.mat) = c("tr.loss","te.loss")
rownames(gbm.loss.mat) = 1:length(iter)
gbm.loss.mat


###linear regression
lm.fit = lm(try.vec~.,data=xy.df)
lm.loss.vec = c(sum(abs(try.vec-predict(lm.fit,xy.df))),sum(abs(tey.vec-predict(lm.fit,nxy.df))))
names(lm.loss.vec) = c("tr.loss","te.loss")
lm.loss.vec

###quantile regression
rq.fit = rq(try.vec~.,data=xy.df,tau=0.5)
rq.loss.vec = c(sum(abs(try.vec-predict(rq.fit,xy.df))),sum(abs(tey.vec-predict(rq.fit,nxy.df))))
names(rq.loss.vec) = c("tr.loss","te.loss")
rq.loss.vec



#plot
par(mfrow=c(1,3))
plot(tey.vec,predict(lm.fit,nxy.df))
plot(tey.vec,F1.mat[which.min(loss.mat[,2]),])
plot(tey.vec,predict(gbm,nxy.df,which.min(gbm.loss.mat[,2])))


### my.ladt.gb.fun
### lad tree gradient boost
my.ladt.gb.fun = function(try.vec,trx.mat,tex.mat,iter){
  trg.vec = rep(NA,length(try.vec)); teg.vec = rep(NA,nrow(tex.mat))
  F0 = median(try.vec); F1 = median(try.vec)
  for(i in 1:iter){
    res = try.vec - F0; t.y.vec = sign(try.vec - F0)
    data = data.frame(t.y.vec,trx.mat); te.data = data.frame(tex.mat)
    fit = rpart(t.y.vec~.,data=data,method = "anova",control = rpart.control(minsplit = 10,cp=0,xval = 5,maxdepth = 1))
    p = predict(fit,te.data); h=predict(fit,data)
    p = round(p,digits=6); h = round(h,digits=6)
    pos = as.numeric(names(table(h))); pos = round(pos,digits=6)
    trg.vec[h==pos[1]] = median(res[h==pos[1]])
    trg.vec[h==pos[2]] = median(res[h==pos[2]])
    teg.vec[p==pos[1]] = median(res[h==pos[1]])
    teg.vec[p==pos[2]] = median(res[h==pos[2]])
    F1 = F1 + teg.vec
    F0 = F0 + trg.vec
  }
  return(list(pred = F1))
}

v=my.ladt.gb.fun(try.vec,trx.mat,trx.mat,10)
sum(abs(v$pred-try.vec))
v=my.ladt.gb.fun(try.vec,trx.mat,tex.mat,10)
sum(abs(v$pred-tey.vec))


### lad tree gradient boost
my.ladt.s.gb.fun = function(try.vec,trx.mat,tex.mat,iter,k = 0.7,nu = 0.05){
  len = length(try.vec)
  trg.vec = rep(NA,len*k); teg.vec = rep(NA,nrow(tex.mat))
  F0 = median(try.vec); F1 = median(try.vec)
  for(i in 1:iter){
    trid = sample(1:len)[1:(len*k)]
    trid = sort(trid)
    ttry.vec = try.vec[trid]
    res = ttry.vec - F0; t.y.vec = sign(ttry.vec - F0)
    data = data.frame(t.y.vec,trx.mat[trid,]); te.data = data.frame(tex.mat)
    fit = rpart(t.y.vec~.,data=data,method = "anova",control = rpart.control(minsplit = 10,cp=0,xval = 5,maxdepth = 1))
    p = predict(fit,te.data); h=predict(fit,data)
    p = round(p,digits=6); h = round(h,digits=6)
    pos = as.numeric(names(table(h))); pos = round(pos,digits=6)
    trg.vec[h==pos[1]] = median(res[h==pos[1]])
    trg.vec[h==pos[2]] = median(res[h==pos[2]])
    teg.vec[p==pos[1]] = median(res[h==pos[1]])
    teg.vec[p==pos[2]] = median(res[h==pos[2]])
    F1 = F1 + nu*teg.vec
    F0 = F0 + nu*trg.vec
  }
  return(list(pred = F1))
}


v=my.ladt.s.gb.fun(try.vec,trx.mat,trx.mat,100,k=0.7,nu=0.05)
sum(abs(v$pred-try.vec))
v=my.ladt.s.gb.fun(try.vec,trx.mat,tex.mat,100,k=0.7,nu=0.05)
sum(abs(v$pred-tey.vec))
