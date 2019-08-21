rm(list=ls())
library(rpart)
library(adabag)
library(xgboost)
library(Matrix)
################################# data ##################################
mydata = read.csv("https://stats.idre.ucla.edu/stat/data/binary.csv")
mydata[,4] = as.factor(mydata[,4])
train = mydata[1:350,]
test = mydata[351:400,]
train$admit2[train$admit==1] = 1
train$admit2[train$admit==0] = -1
test$admit2[test$admit==1] = 1
test$admit2[test$admit==0] = -1

tr.y = train[,5]
tr.x = train[,-c(1,5)]
te.y = test[,5]
te.x = test[-c(1,5)]
iter = 10
################################## Adaboost #############################
my.ada.fun = function(try.vec,trx.mat,tex.mat,iter){
  n = length(try.vec); w.vec = rep(1/n,n)
  tre.mat = matrix(NA,iter,n); tee.mat = matrix(NA,iter,nrow(tex.mat))
  tr.xy.df = data.frame(try.vec,trx.mat); te.xy.df = data.frame(tex.mat)
  for(i in 1:iter){
    h = rpart(try.vec~.,data = tr.xy.df,method="class",weights = w.vec,
              control = rpart.control(minsplit = 10,cp=0,xval = 5,maxdepth = 1))
    #train
    trh.pred = as.numeric(predict(h,tr.xy.df,type="class"))
    trh.pred[trh.pred==1] = -1; trh.pred[trh.pred==2] = 1
    #test
    teh.pred = as.numeric(predict(h,te.xy.df,type="class"))
    teh.pred[teh.pred==1] = -1; teh.pred[teh.pred==2] = 1
    e = sum(w.vec*(try.vec!=trh.pred)); c = log((1-e)/e)/2
    if(e>=0.5){ c = log((1-0.5)/0.5)/2 }; if(e==0){ c = log((1-0.001)/0.001)/2 }
    i.vec = try.vec==trh.pred; w.vec = w.vec*exp(c*(1-i.vec)); w.vec = w.vec/sum(w.vec)
    tre.mat[i,] = c*trh.pred; tee.mat[i,] = c*teh.pred
  }
  #train strong classifier
  train = ifelse(colSums(tre.mat)>0,1,0)
  #test strong classifier
  test = ifelse(colSums(tee.mat)>0,1,0)
  return(list(train=train,test=test))
}

v = my.ada.fun(tr.y,tr.x,te.x,20)

############################################################################## 
comp.mat = matrix(NA,2,3) 
colnames(comp.mat) = c("myadaboost","boosting","xgboost")
rownames(comp.mat) = c("train.acc","test.acc")

#myadaboost train, test acc
comp.mat[1,1] = mean(train$admit==v$train)
comp.mat[2,1] = mean(test$admit==v$test)

#boosting function train, test acc
train$admit=as.factor(train$admit)
test$admit=as.factor(test$admit)
boost = boosting(admit~gre+gpa+rank,data=train,boos = F,mfinal=iter,
                 control=rpart.control(maxdepth=1))
boost.train = predict.boosting(boost,newdata=train[,-5])
boost.test = predict.boosting(boost,newdata=test)
comp.mat[1,2] = mean(train$admit==boost.train$class)
comp.mat[2,2] = mean(test$admit==boost.test$class)


#xgboost
sparse_matrix <- sparse.model.matrix(admit ~ .-1, data = mydata)
output_vector = mydata$admit
train.xgb = sparse_matrix[1:350,]
test.xgb = sparse_matrix[351:400,]
train.label = output_vector[1:350]
test.label = output_vector[351:400]


dtrain <- xgb.DMatrix(data = train.xgb, label = train.label)
dtest <- xgb.DMatrix(data = test.xgb, label = test.label)
watchlist <- list(train=dtrain, test=dtest)

parms = list(max.depth = 1, eta = 0.3)
bstSparse <- xgb.train(parms,data = dtrain,
                       nrounds = 2,
                       nthread = 2,
                       watchlist = watchlist,objective = "binary:logistic")



tab=table(predict(bstSparse,train.xgb)>0.5,train.label)
comp.mat[1,3] = sum(diag(tab))/sum(tab)
tab=table(predict(bstSparse,test.xgb)>0.5,test.label)
comp.mat[2,3] = sum(diag(tab))/sum(tab)

comp.mat

