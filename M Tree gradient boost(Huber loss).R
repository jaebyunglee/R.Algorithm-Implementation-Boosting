rm(list=ls())
library(rpart)
library(gbm)
set.seed(1234)
n = 1000; p = 30
x.mat = matrix(rnorm(n*p),n,p)
y.vec = rnorm(n)
#y.vec[y.vec > max(y.vec)*0.8] = 10

###
iter = 1:15
trid = sample(1:n)[1:(n*0.7)] 
try.vec = y.vec[trid]; tey.vec = y.vec[-trid]
trx.mat = x.mat[trid,]; tex.mat = x.mat[-trid,]
xy.df = data.frame(try.vec,trx.mat); nxy.df = data.frame(tey.vec,tex.mat)
trr.vec = rep(NA,length(try.vec)); ter.vec = rep(NA,length(tey.vec))
trg.vec = rep(NA,length(try.vec)); teg.vec = rep(NA,length(tey.vec))
F0.mat = matrix(NA,length(iter),length(try.vec))
F1.mat = matrix(NA,length(iter),length(tey.vec))
loss.mat = matrix(NA,length(iter),2)

### M-tree boost(huber loss)
for(j in iter){
  print(j)
  F0 = median(try.vec)
  F1 = median(try.vec)
  for(i in 1:j){
    res = try.vec - F0 
    del = quantile(abs(res),0.8)
    t.y.vec = res*I(abs(res)<=del)+del*sign(res)*I(abs(res)>del)
    data = data.frame(t.y.vec,trx.mat); te.data = data.frame(tey.vec,tex.mat)
    fit = rpart(t.y.vec~.,data=data,method = "anova",control = rpart.control(minsplit = 10,cp=0,maxdepth = 1))
    h = round(predict(fit,data),6); p = round(predict(fit,te.data),6)
    pos = round(as.numeric(names(table(h))),6)
    h.pos = h==pos[1] ; p.pos = p==pos[1]
    #train
    trr.vec[h.pos] = median(res[h.pos]); trr.vec[!h.pos] = median(res[!h.pos])
    v.vec = sign((res-trr.vec))*apply(cbind(del,abs(res-trr.vec)),1,min)
    trg.vec[h.pos] = 1/sum(h.pos)*sum(v.vec[h.pos])
    trg.vec[!h.pos] = 1/sum(!h.pos)*sum(v.vec[!h.pos])
    trg.vec = trr.vec+trg.vec
    #test
    ter.vec[p.pos] = median(res[h.pos]); ter.vec[!p.pos] = median(res[!h.pos])
    teg.vec[p.pos] = 1/sum(h.pos)*sum(v.vec[h.pos])
    teg.vec[!p.pos] = 1/sum(!h.pos)*sum(v.vec[!h.pos])
    teg.vec = ter.vec+teg.vec
    #F
    F1 = F1 + teg.vec; F0 = F0 + trg.vec
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
gbm=gbm(try.vec~.,data=xy.df,distribution = "tdist", shrinkage=1,interaction.depth = 1,n.trees=300)
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



#plot
par(mfrow=c(1,3))
plot(tey.vec,predict(lm.fit,nxy.df))
plot(tey.vec,F1.mat[which.min(loss.mat[,2]),])
plot(tey.vec,predict(gbm,nxy.df,which.min(gbm.loss.mat[,2])))



### m tree gradient boost(Huber)
my.mt.gb.fun = function(try.vec,trx.mat,tex.mat,iter){
  trr.vec = rep(NA,length(try.vec)); ter.vec = rep(NA,nrow(tex.mat))
  trg.vec = rep(NA,length(try.vec)); teg.vec = rep(NA,nrow(tex.mat))
  F0 = median(try.vec); F1 = median(try.vec)
  for(i in 1:iter){
    res = try.vec - F0 
    del = quantile(abs(res),0.8)
    t.y.vec = res*I(abs(res)<=del)+del*sign(res)*I(abs(res)>del)
    data = data.frame(t.y.vec,trx.mat); te.data = data.frame(tex.mat)
    fit = rpart(t.y.vec~.,data=data,method = "anova",control = rpart.control(minsplit = 10,cp=0,maxdepth = 1))
    h = round(predict(fit,data),6); p = round(predict(fit,te.data),6)
    pos = round(as.numeric(names(table(h))),6)
    h.pos = h==pos[1] ; p.pos = p==pos[1]
    #train
    trr.vec[h.pos] = median(res[h.pos]); trr.vec[!h.pos] = median(res[!h.pos])
    v.vec = sign((res-trr.vec))*apply(cbind(del,abs(res-trr.vec)),1,min)
    trg.vec[h.pos] = 1/sum(h.pos)*sum(v.vec[h.pos])
    trg.vec[!h.pos] = 1/sum(!h.pos)*sum(v.vec[!h.pos])
    trg.vec = trr.vec+trg.vec
    #test
    ter.vec[p.pos] = median(res[h.pos]); ter.vec[!p.pos] = median(res[!h.pos])
    teg.vec[p.pos] = 1/sum(h.pos)*sum(v.vec[h.pos])
    teg.vec[!p.pos] = 1/sum(!h.pos)*sum(v.vec[!h.pos])
    teg.vec = ter.vec+teg.vec
    #F
    F1 = F1 + teg.vec; F0 = F0 + trg.vec
  }
  return(list(pred = F1))
}

v=my.mt.gb.fun(try.vec,trx.mat,trx.mat,15)
sum(abs(v$pred-try.vec))
v=my.mt.gb.fun(try.vec,trx.mat,tex.mat,15)
sum(abs(v$pred-tey.vec))


### m tree gradient boost(Huber)
my.mt.s.gb.fun = function(try.vec,trx.mat,tex.mat,iter,k = 0.7,nu = 0.05){
  len = length(try.vec)
  trr.vec = rep(NA,len*k); ter.vec = rep(NA,nrow(tex.mat))
  trg.vec = rep(NA,len*k); teg.vec = rep(NA,nrow(tex.mat))
  F0 = median(try.vec); F1 = median(try.vec)
  for(i in 1:iter){
    trid = sample(1:len)[1:(len*k)]; trid = sort(trid)
    ttry.vec = try.vec[trid]
    res = ttry.vec - F0 
    del = quantile(abs(res),0.8)
    t.y.vec = res*I(abs(res)<=del)+del*sign(res)*I(abs(res)>del)
    data = data.frame(t.y.vec,trx.mat[trid,]); te.data = data.frame(tex.mat)
    fit = rpart(t.y.vec~.,data=data,method = "anova",control = rpart.control(minsplit = 10,cp=0,maxdepth = 1))
    h = round(predict(fit,data),6); p = round(predict(fit,te.data),6)
    pos = round(as.numeric(names(table(h))),6)
    h.pos = h==pos[1] ; p.pos = p==pos[1]
    #train
    trr.vec[h.pos] = median(res[h.pos]); trr.vec[!h.pos] = median(res[!h.pos])
    v.vec = sign((res-trr.vec))*apply(cbind(del,abs(res-trr.vec)),1,min)
    trg.vec[h.pos] = 1/sum(h.pos)*sum(v.vec[h.pos])
    trg.vec[!h.pos] = 1/sum(!h.pos)*sum(v.vec[!h.pos])
    trg.vec = trr.vec+trg.vec
    #test
    ter.vec[p.pos] = median(res[h.pos]); ter.vec[!p.pos] = median(res[!h.pos])
    teg.vec[p.pos] = 1/sum(h.pos)*sum(v.vec[h.pos])
    teg.vec[!p.pos] = 1/sum(!h.pos)*sum(v.vec[!h.pos])
    teg.vec = ter.vec+teg.vec
    #F
    F1 = F1 + nu*teg.vec; F0 = F0 + nu*trg.vec
  }
  return(list(pred = F1))
}

v=my.mt.s.gb.fun(try.vec,trx.mat,trx.mat,15,k=0.7,nu=0.05)
sum(abs(v$pred-try.vec))
v=my.mt.s.gb.fun(try.vec,trx.mat,tex.mat,15,k=0.7,nu=0.05)
sum(abs(v$pred-tey.vec))

