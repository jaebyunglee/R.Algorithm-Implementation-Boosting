####################################################################
#########################              #############################
#########################   adaboost   #############################
#########################              #############################
####################################################################

### adaboost
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
    i.vec = try.vec==trh.pred; w.vec = w.vec*exp(-c*try.vec*trh.pred); w.vec = w.vec/sum(w.vec)
    tre.mat[i,] = c*trh.pred; tee.mat[i,] = c*teh.pred
  }
  pred = ifelse(colSums(tee.mat)>0,1,0)
  return(list(pred = pred))
}



### real adaboost
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
  pred = ifelse(colSums(tef.mat)>0,1,0)
  return(list(pred = pred))
}



### logit boost
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
  pred = ifelse(pFx>0,1,0)
  return(list(pred = pred))
}



### gentle adaboost
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
  pred = ifelse(pFx>=0,1,0)
  return(list(pred = pred))
}



### multi adaboost
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
  p.C.mat = t(colSums(pred.array))
  pred = apply(p.C.mat,2,which.max)
  return(list(pred=pred))
}


### multi logit boost
my.mlogitb.fun = function(try.vec,trx.mat,tex.mat,iter){
  n = length(try.vec); y.vec = try.vec; ncl = nlevels(y.vec)
  px = matrix(c(1/ncl),ncl,n)
  Fx = matrix(0,ncl,n); fx = matrix(0,ncl,n)
  pFx = matrix(0,ncl,nrow(tex.mat)); pfx = matrix(0,ncl,nrow(tex.mat))
  z.mat = matrix(NA,ncl,n); w.mat = matrix(NA,ncl,n)
  ########### Multi logit boost ############
  for(id in 1:iter){
    for(j in 1:ncl){
      z.mat[j,] = (as.numeric(y.vec==levels(y.vec)[j]) - px[j,])/(px[j,]*(1-px[j,]))
      w.mat[j,] = (px[j,]*(1-px[j,]))
      h = rpart(z.mat[j,]~.,data=trx.mat,weights=w.mat[j,],method="anova",control = rpart.control(minsplit = 10,cp=0,xval = 5,maxdepth = 1))
      #h = lm(z.mat[j,]~.,data = trx.mat,weights = w.mat[j,])
      fx[j,] = predict(h,trx.mat)
      pfx[j,] = predict(h,tex.mat)
    }
    
    g = colSums(fx)
    for(j in 1:ncl){
      fx[j,] = (ncl-1)/ncl*(fx[j,]-1/ncl*g)
      Fx[j,] = Fx[j,] + fx[j,]
      pFx[j,] = pFx[j,] + pfx[j,]
    }
    
    for(j in 1:ncl){
      px[j,] = exp(Fx[j,])/(rowSums(apply(Fx,1,exp)))
    }
  }
  pred = apply(pFx,2,which.max)
  return(list(pred=pred))
}



### adaboost mh
my.adamh.fun = function(try.vec,trx.mat,tex.mat,iter){
  n = length(try.vec); nlevels = length(levels(tr.y))
  tr.xy.df = data.frame(try.vec,trx.mat);te.xy.df = data.frame(tex.mat)
  tr.y.mat = matrix(NA,n,nlevels);te.y.mat = matrix(NA,nrow(tex.mat),nlevels)
  e.array = array(NA,c(iter,n,nlevels)); pred.array = array(NA,c(iter,nrow(tex.mat),nlevels))
  for(j in 1:nlevels){
    w.vec = rep(1/n,n)
    tr.y.mat[try.vec==levels(try.vec)[j],j] = 1 
    tr.y.mat[try.vec!=levels(try.vec)[j],j] = -1 
    tr.xy.df$try.vec = tr.y.mat[,j]
    #real adaboost
    for(i in 1:iter){
      h = rpart(try.vec~.,data = tr.xy.df,method="class",weights = w.vec,maxdepth=5)
      tr.p = as.numeric(predict(h,tr.xy.df,type="prob")[,2])
      te.p = as.numeric(predict(h,te.xy.df,type="prob")[,2])
      f.vec = 0.5*log(tr.p/(1-tr.p)); tef.vec = 0.5*log(te.p/(1-te.p))
      f.vec[f.vec==-Inf] = -1e+7; f.vec[f.vec==Inf] = 1e+7
      tef.vec[tef.vec==-Inf] = -1e+7; tef.vec[tef.vec==Inf] = 1e+7
      w.vec = w.vec*exp(-tr.xy.df$try.vec*f.vec)
      w.vec[w.vec==0] = 1e-7; w.vec[w.vec==Inf] = 1e+7
      #train classifier
      e.array[i,,j] = f.vec; pred.array[i,,j] = tef.vec
    }
  }
  p.C.mat = t(colSums(pred.array))
  pred = apply(p.C.mat,2,which.max)
  return(list(pred = pred))
}



####################################################################
#########################              #############################
#########################   gradient   #############################
#########################              #############################
####################################################################


### ls tree gradient boost
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
  return(list(pred = F1))
}




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



###lk tree gradient boost
my.lkt.gb.fun = function(try.vec,trx.mat,tex.mat,iter){
  nlevels = nlevels(try.vec)
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
  pred = apply(pFx,2,which.max)
  return(list(pred = pred))
}

