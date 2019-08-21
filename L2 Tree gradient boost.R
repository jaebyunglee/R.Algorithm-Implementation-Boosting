rm(list=ls())
library(rpart)
library(gbm)
library(quantreg)
library(randomForest)
set.seed(12)
n = 1000; p = 30
x.mat = matrix(rnorm(n*p),n,p)
y.vec = rnorm(n)
y.vec = ifelse(y.vec>0,1,-1)

iter = 1:15
acc.mat = matrix(NA,length(iter),2)

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


######################################## L_k Tree boost #############################
for(j in iter){
  print(j)
  F0 = 0.5*log((1+mean(try.vec))/(1-mean(try.vec)))
  F1 = 0.5*log((1+mean(try.vec))/(1-mean(try.vec)))
  for(i in 1:j){
    t.y.vec = 2*try.vec/(1+exp(2*try.vec*F0))
    data = data.frame(t.y.vec,trx.mat)
    te.data = data.frame(tey.vec,tex.mat)
    fit = rpart(t.y.vec~.,data=data,method = "anova",control = rpart.control(minsplit = 10,cp=0,xval = 5,maxdepth = 1))
    p = predict(fit,te.data); h=predict(fit,data)
    p = round(p,digits=6); h = round(h,digits=6)
    pos = as.numeric(names(table(h))); pos = round(pos,digits=6)
    h.pos = h==pos[1] ; p.pos = p==pos[1]
    trg.vec[h.pos] = sum(t.y.vec[h.pos])/sum((abs(t.y.vec)*(2-abs(t.y.vec)))[h.pos])
    trg.vec[!h.pos] = sum(t.y.vec[!h.pos])/sum((abs(t.y.vec)*(2-abs(t.y.vec)))[!h.pos])
    teg.vec[p.pos] = sum(t.y.vec[h.pos])/sum((abs(t.y.vec)*(2-abs(t.y.vec)))[h.pos])
    teg.vec[!p.pos] = sum(t.y.vec[!h.pos])/sum((abs(t.y.vec)*(2-abs(t.y.vec)))[!h.pos])
    F1 = F1 + teg.vec
    F0 = F0 + trg.vec
  }
  F0 = ifelse(1/(1+exp(-2*F0))>1/(1+exp(2*F0)),1,-1)
  F1 = ifelse(1/(1+exp(-2*F1))>1/(1+exp(2*F1)),1,-1)
  F0.mat[j,] = F0
  F1.mat[j,] = F1
  acc.mat[j,1] = mean(try.vec==F0)
  acc.mat[j,2] = mean(tey.vec==F1)
}
colnames(acc.mat) = c("tr.acc","te.acc")
rownames(acc.mat) = 1:length(iter)
acc.mat





########################################## gbm package #############################
gxy.df = data.frame(ifelse(try.vec>0,1,0),trx.mat)
names(gxy.df)[1] = "try.vec"
gnxy.df = data.frame(ifelse(tey.vec>0,1,0),tex.mat)
names(gnxy.df)[1] = "tey.vec"
gbm=gbm(try.vec~.,data=gxy.df,distribution = "bernoulli", shrinkage=1,interaction.depth = 1,n.trees=300,bag.frac=1)
gbm.acc.mat = matrix(NA,length(iter),2)
for(i in 1:length(iter)){
  a=predict(gbm,gxy.df,n.trees = i)
  b=predict(gbm,gnxy.df,n.trees = i)
  gtr = ifelse(1/(1+exp(-2*a))>1/(1+exp(2*a)),1,0)
  gte = ifelse(1/(1+exp(-2*b))>1/(1+exp(2*b)),1,0)
  gbm.acc.mat[i,1]=mean(gtr==ifelse(try.vec>0,1,0))
  gbm.acc.mat[i,2]=mean(gte==ifelse(tey.vec>0,1,0))
}
colnames(gbm.acc.mat) = c("tr.acc","te.acc")
rownames(gbm.acc.mat) = 1:length(iter)
gbm.acc.mat


############################### generalized linear regression #####################
glm.fit = glm(try.vec~.,data=gxy.df,family = "binomial")
glm.tr = predict(glm.fit,gxy.df,type="response")
glm.te = predict(glm.fit,gnxy.df,type="response")
glm.acc.vec = c(mean(try.vec==ifelse(glm.tr>0.5,1,-1)),mean(tey.vec==ifelse(glm.te>0.5,1,-1)))
names(glm.acc.vec) = c("tr.acc","te.acc")
glm.acc.vec

############################### tree #####################################
fit.rpart <- rpart(try.vec ~ ., data = xy.df, method="class")
# print(fit.rpart)
printcp(fit.rpart)   
plotcp(fit.rpart)    
# Pruning
prn.rpart <- prune(fit.rpart, cp = 0.01)


rpart = predict(prn.rpart,newdata=xy.df,type="class")
pred.rpart = predict(prn.rpart,newdata=nxy.df,type="class")
#tr.acc
mean(rpart==try.vec)
#te.acc
mean(pred.rpart==tey.vec)


################################## Random Forest ##################################
# Create a Random Forest model with default parameters
fit1.RF <- randomForest(as.factor(try.vec)~., data = xy.df, method="class")
print(fit1.RF)

# Predicting 
RF <- predict(fit1.RF, xy.df, type = "class")
pred.RF <- predict(fit1.RF, nxy.df, type = "class")
# Checking classification accuracy
mean(try.vec==RF)  
mean(tey.vec==pred.RF)





### l2 tree gradient boost
my.l2t.gb.fun = function(try.vec,trx.mat,tex.mat,iter){
  trg.vec = rep(NA,length(try.vec)); teg.vec = rep(NA,nrow(tex.mat))
  F0 = 0.5*log((1+mean(try.vec))/(1-mean(try.vec)))
  F1 = 0.5*log((1+mean(try.vec))/(1-mean(try.vec)))
  for(i in 1:iter){
    t.y.vec = 2*try.vec/(1+exp(2*try.vec*F0))
    data = data.frame(t.y.vec,trx.mat)
    te.data = data.frame(tex.mat)
    fit = rpart(t.y.vec~.,data=data,method = "anova",control = rpart.control(minsplit = 10,cp=0,xval = 5,maxdepth = 1))
    p = predict(fit,te.data); h=predict(fit,data)
    p = round(p,digits=6); h = round(h,digits=6)
    pos = as.numeric(names(table(h))); pos = round(pos,digits=6)
    h.pos = h==pos[1] ; p.pos = p==pos[1]
    trg.vec[h.pos] = sum(t.y.vec[h.pos])/sum((abs(t.y.vec)*(2-abs(t.y.vec)))[h.pos])
    trg.vec[!h.pos] = sum(t.y.vec[!h.pos])/sum((abs(t.y.vec)*(2-abs(t.y.vec)))[!h.pos])
    teg.vec[p.pos] = sum(t.y.vec[h.pos])/sum((abs(t.y.vec)*(2-abs(t.y.vec)))[h.pos])
    teg.vec[!p.pos] = sum(t.y.vec[!h.pos])/sum((abs(t.y.vec)*(2-abs(t.y.vec)))[!h.pos])
    F1 = F1 + teg.vec
    F0 = F0 + trg.vec
  }
  F0 = ifelse(1/(1+exp(-2*F0))>1/(1+exp(2*F0)),1,-1)
  F1 = ifelse(1/(1+exp(-2*F1))>1/(1+exp(2*F1)),1,-1)
  return(list(pred = F1))
}


v=my.l2t.gb.fun(try.vec,trx.mat,tex.mat,14)
mean(v$pred==tey.vec)

### l2 tree gradient boost
my.l2t.s.gb.fun = function(try.vec,trx.mat,tex.mat,iter,k = 0.7,nu = 0.05){
  len = length(try.vec)
  trg.vec = rep(NA,len*k); teg.vec = rep(NA,nrow(tex.mat))
  F0 = 0.5*log((1+mean(try.vec))/(1-mean(try.vec)))
  F1 = 0.5*log((1+mean(try.vec))/(1-mean(try.vec)))
  for(i in 1:iter){
    set.seed(123)
    trid = sample(1:len)[1:(len*k)]; trid = sort(trid)
    ttry.vec = try.vec[trid]
    t.y.vec = 2*ttry.vec/(1+exp(2*ttry.vec*F0))
    data = data.frame(t.y.vec,trx.mat[trid,])
    te.data = data.frame(tex.mat)
    fit = rpart(t.y.vec~.,data=data,method = "anova",control = rpart.control(minsplit = 10,cp=0,xval = 5,maxdepth = 1))
    p = predict(fit,te.data); h=predict(fit,data)
    p = round(p,digits=6); h = round(h,digits=6)
    pos = as.numeric(names(table(h))); pos = round(pos,digits=6)
    h.pos = h==pos[1] ; p.pos = p==pos[1]
    trg.vec[h.pos] = sum(t.y.vec[h.pos])/sum((abs(t.y.vec)*(2-abs(t.y.vec)))[h.pos])
    trg.vec[!h.pos] = sum(t.y.vec[!h.pos])/sum((abs(t.y.vec)*(2-abs(t.y.vec)))[!h.pos])
    teg.vec[p.pos] = sum(t.y.vec[h.pos])/sum((abs(t.y.vec)*(2-abs(t.y.vec)))[h.pos])
    teg.vec[!p.pos] = sum(t.y.vec[!h.pos])/sum((abs(t.y.vec)*(2-abs(t.y.vec)))[!h.pos])
    F1 = F1 + nu*teg.vec
    F0 = F0 + nu*trg.vec
  }
  F0 = ifelse(1/(1+exp(-2*F0))>1/(1+exp(2*F0)),1,-1)
  F1 = ifelse(1/(1+exp(-2*F1))>1/(1+exp(2*F1)),1,-1)
  return(list(pred = F1))
}

for(i in 1:20){
  print(i)
  for(j in seq(1,0.1,length.out = 10)){
    v=my.l2t.s.gb.fun(try.vec,trx.mat,tex.mat,i,k = j,nu = 0.05)
    print(mean(v$pred==tey.vec))
  }
}

v=my.l2t.s.gb.fun(try.vec,trx.mat,tex.mat,14,k = 0.5,nu = 0.05)

tab=table(tey.vec,v$pred)
tab[2,2]/sum(tab[2,])
tab[1,1]/sum(tab[1,])
sum(diag(tab))/sum(tab)

tab1 = table(tey.vec,F1.mat[15,])
tab1[2,2]/sum(tab1[2,])
tab1[1,1]/sum(tab1[1,])
sum(diag(tab1))/sum(tab1)
