library(caret)
library(MLmetrics)
library(DescTools)
library(pROC)
library(mltools)
library(mccr)
library(mcc)
library(measures)
library(rpart)
library(visdat)
library(tidyverse)
library(tidymodels)
library(patchwork)
library(DEoptim)
library(parallel)
library(glmnet)
library(doParallel)
library(foreach)


#days<-c('camel','derby','eclipse','groovy','hbase','hive','ivy','jruby','log4j','lucene','poi','prop1','prop2','prop3','prop4','prop5','redaktor','synapse','tomcat','velocity','wicket','xalan','xerces')
days<-c('1days','7days','14days','30days','90days','180days','365days')
#'camel','cloudstack',
files<-c('camel','cloudstack','cocoon','deeplearning','hadoop','hive','node','ofbiz','qpid')

for (k in 1:length(days)){
  for (j in 1:length(files)){
    s<-paste("E:/gln/C/mypackage/dataset/dataset/issueclosetime/",days[k],'/',files[j],'/',"train/",sep='')
    filename <- list.files(s)
    df<- data.frame()
    df_para<-data.frame()
    auc<-vector(mode="numeric",length=100)
    mcc1<-vector(mode="numeric",length=100)
    recal<-vector(mode="numeric",length=100)
    f1<-vector(mode="numeric",length=100)
    brier<-vector(mode="numeric",length=100)
    balan<-vector(mode="numeric",length=100)
    prec<-vector(mode="numeric",length=100)
    gmean<-vector(mode="numeric",length=100)
    
    dt_alpha<-vector(mode="numeric",length=100)
    dt_lambda<-vector(mode="numeric",length=100)
    
    
    for (i in 1:length(filename)){
      
      s1=paste("E:/gln/C/mypackage/dataset/dataset/issueclosetime/",days[k],'/',files[j],'/',"train/",filename[i],sep='')
      data_train<-read.csv(s1)
      data_train$issude_close_time<-as.factor(data_train$issude_close_time)
      #levels(data_train$label)<-c('No','Yes')
      #data_train<-droplevels(data_train)
      Traindata<-data_train[,-ncol(data_train)]
      Trainclass<-data_train$issude_close_time
      Traindata<-data.matrix(Traindata)
      
      s2=paste("E:/gln/C/mypackage/dataset/dataset/issueclosetime/",days[k],'/',files[j],'/',"test/",filename[i],sep='')
      data_test<-read.csv(s2)
      Testdata<-data_test[,-ncol(data_test)]
      Testdata<-data.matrix(Testdata)
      
      glmnet_model<- glmnet(Traindata,data_train$issude_close_time, type.measure = "class",family='binomial')
      
      ss_min<-min(glmnet_model$lambda)
      
      y_pred <- predict(glmnet_model,Testdata,type='class',s=ss_min)
      y_prob<-predict(glmnet_model,Testdata,type='response', s=ss_min)
      
      y_true<-data_test$issude_close_time
      # y_true[which(y_true==0)]<-'No'
      #y_true[which(y_true==1)]<-'Yes'
      meas<-confusionMatrix(as.factor(y_pred),as.factor(y_true),positive='1', mode = "prec_recall")
      prec[i]<-meas$byClass['Precision']
      f1[i]<-meas$byClass['F1']
      recal[i]<-meas$byClass['Recall']
      balan[i]<-meas$byClass['Balanced Accuracy']
      brier[i]<-BrierScore(data_test$issude_close_time, y_prob)
      auc[i]<-auc(as.factor(data_test$issude_close_time),y_prob)
      y_pred1<-as.factor(as.numeric(y_prob > 0.5))
      mcc1[i]<-mccr(data_test$issude_close_time,y_pred1)
      gmean[i]<-GMEAN(data_test$issude_close_time,y_pred1, 0, 1)
    }
    va<-cbind(auc,mcc1,recal,f1,brier,balan,prec,gmean)
    
    names <- c("AUC","MCC1","Recall","F1","Brierce","balance","precision","G-Mean")
    df<- data.frame()
    df[[names[1]]] <- as.numeric()
    df[[names[2]]] <- as.numeric()
    df[[names[3]]] <- as.numeric()
    df[[names[4]]] <- as.numeric()
    df[[names[5]]] <- as.numeric()
    df[[names[6]]] <- as.numeric()
    df[[names[7]]] <- as.numeric()
    df[[names[8]]] <- as.numeric()
    
    df <- rbind(df,va)
    s_result<-paste("E:/gln/C/mypackage/result_new/RQ1/defaultsetting/LR/issueclosetime/",days[k],"/glmnet/",files[j],".csv",sep='')
    write.table(df,s_result,row.names=FALSE,col.names=TRUE,sep=",")
  }
}
  