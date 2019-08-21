rm(list=ls())
library(adabag)
library(rpart)
library(randomForest)
library(VGAM)
library(xgboost)
library(Matrix)
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


comp.mat = matrix(NA,2,6) 
colnames(comp.mat) = c("myadaboost","boosting","vglm","tree","RandomFrest","xgboost")
rownames(comp.mat) = c("train.acc","test.acc")


tr.y = train[,10]
tr.x = train[,-10]
te.y = test[,10]
te.x = test[,-10]
iter = 20
################################## Multi Adaboost #############################

my.mada.fun = function(try.vec,trx.mat,tex.mat,iter){
  n = length(try.vec); nlevels = length(levels(tr.y)); w.vec = rep(1/n,n)
  e.array = array(NA,c(iter,n,nlevels)); pred.array = array(NA,c(iter,nrow(tex.mat),nlevels))
  tr.xy.df = data.frame(try.vec,trx.mat); te.xy.df = data.frame(tex.mat)
  #train
  for(i in 1:iter){
    h = rpart(try.vec~.,data = tr.xy.df,method="class",weights = w.vec,maxdepth=10)
    trh.pred = predict(h,tr.xy.df,type="class"); teh.pred = predict(h,te.xy.df,type="class")
    e = sum(w.vec*(try.vec!=trh.pred)); c = log((1-e)/e)/2 
    if(e>=0.5){ c = log((1-0.5)/0.5)/2 } ;if(e==0){ c = log((1-0.001)/0.001)/2 }
    i.vec = try.vec==trh.pred; w.vec = w.vec*exp(c*(1-i.vec)); w.vec = w.vec/sum(w.vec)
    for(j in 1:nlevels){
      e.array[i,,j] = c*(trh.pred==j); pred.array[i,,j] = c*(teh.pred==j)
    }
  }
  C.mat = t(colSums(e.array)); p.C.mat = t(colSums(pred.array))
  Cx = apply(C.mat,2,which.max); p.Cx = apply(p.C.mat,2,which.max)
  return(list(train=Cx,test=p.Cx))
}

v = my.mada.fun(tr.y,tr.x,te.x,20)
#train acc
comp.mat[1,1] = mean(tr.y==v$train)
comp.mat[2,1] = mean(test$V10==v$test)
################################ adaboosting ###################################


boost = boosting(V10~.,data=train,boos = F,mfinal=iter,
                 control=rpart.control(maxdepth=10))
boost.train  = predict.boosting(boost,train,type="class")  
comp.mat[1,2] = 1-boost.train$error
boost.test  = predict.boosting(boost,test,type="class") 
comp.mat[2,2] = 1-boost.test$error


################################## multi logistic #################################################
fit.glm = vglm(V10~.,data = train,family = "multinomial")
tr.pred.glm = predict(fit.glm,train,type = "response")
te.pred.glm = predict(fit.glm,test,type = "response")
#multi logstic acc

#glm train acc
comp.mat[1,3] = mean(apply(tr.pred.glm,1,which.max)==train$V10)

#glm test acc
comp.mat[2,3] = mean(apply(te.pred.glm,1,which.max)==test$V10)
################################## deciseion tree ##################################


fit.rpart <- rpart(V10 ~ ., data = train, method="class")
plot(fit.rpart)
text(fit.rpart)

# print(fit.rpart)
printcp(fit.rpart)   
plotcp(fit.rpart)    

# Pruning
prn.rpart <- prune(fit.rpart, cp = 0.01)
# print(prn.rpart)
plot(prn.rpart)
text(prn.rpart)

rpart = predict(prn.rpart,newdata=train,type="class")
comp.mat[1,4] = mean(rpart==train$V10)
pred.rpart = predict(prn.rpart,newdata=test,type="class")
comp.mat[2,4] = mean(pred.rpart==test$V10)

################################## Random Forest ##################################
# Create a Random Forest model with default parameters
fit1.RF <- randomForest(V10 ~ ., data = train, importance = TRUE)
print(fit1.RF)

# Predicting 
RF <- predict(fit1.RF, train, type = "class")
pred.RF <- predict(fit1.RF, test, type = "class")
# Checking classification accuracy
comp.mat[1,5] = mean(train$V10==RF)  
comp.mat[2,5] = mean(test$V10==pred.RF)

# To check important variables
importance(fit1.RF)
varImpPlot(fit1.RF)


sparse_matrix = sparse.model.matrix(V10~.-1,data = cmc.data)
train.gb = sparse_matrix[1:1000,]
train.label = as.numeric(cmc.data[,10][1:1000]) -1
test.gb = sparse_matrix[1001:1473,]
test.label = as.numeric(cmc.data[,10][1001:1473]) -1

dtrain = xgb.DMatrix(data = train.gb , label = train.label)
dtest = xgb.DMatrix(data = test.gb , label = test.label)
watchlist = list(train = dtrain, test = dtest)
params <- list(booster = "gbtree", objective = "multi:softprob", num_class = 3)
bstSparse = xgb.train(params,dtrain,watchlist = watchlist, nrounds = 200, max_depth=2,eta=0.05,verbose = F)
best.epoch = which.min((bstSparse$evaluation_log$test_merror))
comp.mat[1,6] = 1- bstSparse$evaluation_log$train_merror[best.epoch]
comp.mat[2,6] = 1- bstSparse$evaluation_log$test_merror[best.epoch]

comp.mat
