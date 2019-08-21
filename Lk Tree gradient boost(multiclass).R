rm(list=ls())
library(rpart)
library(gbm)
library(VGAM)
###################################### DATA ######################################
cmc.data = read.table("C:\\Users\\kis91\\Dropbox\\2018 Jaebyung Lee\\cmc.data.txt",header=F,sep=",")
#cmc.data = read.table("C:\\Users\\jaebyung.lee\\Desktop\\cmc.data.txt",header=F,sep=",")
str(cmc.data)
cmc.data$V1 = as.numeric(cmc.data$V1)
cmc.data$V4 = as.numeric(cmc.data$V4)
cols <- c(names(cmc.data)[-c(1,4)])
cmc.data[,cols] <- data.frame(apply(cmc.data[,cols], 2, as.factor))
str(cmc.data)
dim(cmc.data)

##################################################################################
n = dim(cmc.data)[1]
iter = 1:20
acc.mat = matrix(NA,length(iter),2)

# trid = sample(1:n)[1:(n*0.7)]
# train = cmc.data[trid,]
# test = cmc.data[-trid,]

train = cmc.data[1:1000,]
test = cmc.data[1001:1473,]
trx.mat = train[,-10]
try.vec = train[,10]
tex.mat = test[,-10]
tey.vec = test[,10]


xy.df = data.frame(try.vec,trx.mat)
nxy.df = data.frame(tey.vec,tex.mat)
nlevels = nlevels(try.vec)
F0.mat = matrix(0,nlevels,length(try.vec))
F1.mat = matrix(0,nlevels,length(tey.vec))
trg.vec = rep(NA,length(try.vec))
teg.vec = rep(NA,length(tey.vec))
Fx = matrix(NA,nlevels,length(try.vec))
pFx = matrix(NA,nlevels,length(tey.vec))

######################################## L_k Tree boost #############################
for(j in iter){
  print(j)
  F0 = F0.mat
  F1 = F1.mat
  p.mat = matrix(NA,nlevels,length(try.vec))
  
  for(i in 1:j){
    for(k in 1:nlevels){
      p.mat[k,] = exp(F0[k,])/colSums(exp(F0))
    }
    
    for(l in 1:nlevels){
      t.y.vec = ifelse(try.vec==l,1,0)-p.mat[l,]
      data = data.frame(t.y.vec,trx.mat)
      te.data = data.frame(tey.vec,tex.mat)
      fit = rpart(t.y.vec~.,data=data,method = "anova",control = rpart.control(minsplit = 10,cp=0.001,xval = 10,maxdepth = 1))
      p = predict(fit,te.data); h=predict(fit,data)
      p = round(p,digits=6); h = round(h,digits=6)
      pos = as.numeric(names(table(h))); pos = round(pos,digits=6)
      h.pos = h==pos[1] ; p.pos = p==pos[1]
      trg.vec[h.pos] = (nlevels-1)/nlevels*(sum(t.y.vec[h.pos])/sum((abs(t.y.vec)*(1-abs(t.y.vec)))[h.pos]))
      trg.vec[!h.pos] = (nlevels-1)/nlevels*(sum(t.y.vec[!h.pos])/sum((abs(t.y.vec)*(1-abs(t.y.vec)))[!h.pos]))
      teg.vec[p.pos] = (nlevels-1)/nlevels*(sum(t.y.vec[h.pos])/sum((abs(t.y.vec)*(1-abs(t.y.vec)))[h.pos]))
      teg.vec[!p.pos] = (nlevels-1)/nlevels*(sum(t.y.vec[!h.pos])/sum((abs(t.y.vec)*(1-abs(t.y.vec)))[!h.pos]))
      F0[l,] = F0[l,] + trg.vec
      F1[l,] = F1[l,] + teg.vec
    }
  }
  
  for(m in 1:nlevels){
    Fx[m,] = exp(F0[m,])/colSums(exp(F0))
    pFx[m,] = exp(F1[m,])/colSums(exp(F1))
  }
  
  train.clf = apply(Fx,2,which.max)
  test.clf = apply(pFx,2,which.max)

 acc.mat[j,1] = mean(train.clf==try.vec)
 acc.mat[j,2] = mean(test.clf==tey.vec)
}


colnames(acc.mat) = c("tr.acc","te.acc")
rownames(acc.mat) = 1:length(iter)
acc.mat


############################################################################################
gbm=gbm(try.vec~.,data=xy.df,distribution = "multinomial", shrinkage=1,interaction.depth = 1,n.trees=300,bag.frac=1)
gbm.acc.mat = matrix(NA,length(iter),2)

for(i in 1:length(iter)){
  a=predict(gbm,xy.df,n.trees = i)
  b=predict(gbm,nxy.df,n.trees = i)
  gtr = apply(a,1,which.max)
  gte = apply(b,1,which.max)
  gbm.acc.mat[i,1]=mean(gtr==try.vec)
  gbm.acc.mat[i,2]=mean(gte==tey.vec)
}
colnames(gbm.acc.mat) = c("tr.acc","te.acc")
rownames(gbm.acc.mat) = 1:length(iter)
gbm.acc.mat

###################################vglm###################################
fit.glm = vglm(try.vec~.,data=xy.df,family = "multinomial")
c = predict(fit.glm,nxy.df,type="response")
mean(apply(c,1,which.max)==tey.vec)



####

my.lkt.gb.fun = function(try.vec,trx.mat,tex.mat,iter){
  nlevels = nlevels(try.vec)
  xy.df = data.frame(try.vec,trx.mat); nxy.df = data.frame(tex.mat)
  trg.vec = rep(NA,length(try.vec)); teg.vec = rep(NA,nrow(tex.mat))
  Fx = matrix(NA,nlevels,length(try.vec)); pFx = matrix(NA,nlevels,nrow(tex.mat))
  F0 = matrix(0,nlevels,length(try.vec)); F1 = matrix(0,nlevels,nrow(tex.mat))
  p.mat = matrix(NA,nlevels,length(try.vec))
  for(i in 1:iter){
    for(k in 1:nlevels){
      p.mat[k,] = exp(F0[k,])/colSums(exp(F0))
    }
    for(l in 1:nlevels){
      t.y.vec = ifelse(try.vec==levels(try.vec)[l],1,0)-p.mat[l,]
      data = data.frame(t.y.vec,trx.mat); te.data = data.frame(tex.mat)
      fit = rpart(t.y.vec~.,data=data,method = "anova",control = rpart.control(minsplit = 10,cp=0.001,xval = 10,maxdepth = 1))
      p = predict(fit,te.data); h=predict(fit,data)
      p = round(p,digits=6); h = round(h,digits=6)
      pos = as.numeric(names(table(h))); pos = round(pos,digits=6)
      h.pos = h==pos[1] ; p.pos = p==pos[1]
      trg.vec[h.pos] = (nlevels-1)/nlevels*(sum(t.y.vec[h.pos])/sum((abs(t.y.vec)*(1-abs(t.y.vec)))[h.pos]))
      trg.vec[!h.pos] = (nlevels-1)/nlevels*(sum(t.y.vec[!h.pos])/sum((abs(t.y.vec)*(1-abs(t.y.vec)))[!h.pos]))
      teg.vec[p.pos] = (nlevels-1)/nlevels*(sum(t.y.vec[h.pos])/sum((abs(t.y.vec)*(1-abs(t.y.vec)))[h.pos]))
      teg.vec[!p.pos] = (nlevels-1)/nlevels*(sum(t.y.vec[!h.pos])/sum((abs(t.y.vec)*(1-abs(t.y.vec)))[!h.pos]))
      F0[l,] = F0[l,] + trg.vec; F1[l,] = F1[l,] + teg.vec
    }
  }
  for(m in 1:nlevels){
    Fx[m,] = exp(F0[m,])/colSums(exp(F0))
    pFx[m,] = exp(F1[m,])/colSums(exp(F1))
  }
  train = apply(Fx,2,which.max); test = apply(pFx,2,which.max)
  return(list(train=train,test=test))
}

v = my.lkt.gb.fun(try.vec,trx.mat,tex.mat,10)
mean(v$train==try.vec)
mean(v$test==tey.vec)
