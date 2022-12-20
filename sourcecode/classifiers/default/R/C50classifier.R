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
library(C50)
library(doParallel)
library(foreach)


#files<-c('derby','eclipse','groovy','hbase','hive','ivy','jruby','log4j','lucene','poi','prop1','prop2','prop3','prop4','prop5','redaktor','synapse','tomcat','velocity','wicket','xalan','xerces')
days<-c('1days','7days','14days','30days','90days','180days','365days')
#'camel','cloudstack',
files<-c('camel','cloudstack','cocoon','deeplearning','hadoop','hive','node','ofbiz','qpid')

for (k in 1:length(days)){
  for (j in 1:length(files)){
    s<-paste("E:/gln/C/mypackage/dataset/dataset/dataset/",files[j],'/',"s_train/",sep='')
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
    
    dt_trials<-vector(mode="numeric",length=100)
    
    
    for (i in 1:length(filename)){
      
      s1=paste("E:/gln/C/mypackage/dataset/dataset/dataset/",files[j],'/',"s_train/",filename[i],sep='')
      data_train<-read.csv(s1)
      data_train$label<-as.factor(data_train$label)
      levels(data_train$label)<-c('No','Yes')
      Traindata<-data_train[,-ncol(data_train)]
      Trainclass<-data_train$label
      #data_train<-droplevels(data_train)
      
      s2=paste("E:/gln/C/mypackage/dataset/dataset/dataset/",files[j],'/',"s_test/",filename[i],sep='')
      data_test<-read.csv(s2)
      Testdata<-data_test[,-ncol(data_test)]
      Testclass<-data_test$label
      
      
      DE_XGBoost_classifier_model <-C5.0(x=Traindata,y=Trainclass,control = C5.0Control(minCases = 20))
      
      
      y_pred <- predict(DE_XGBoost_classifier_model, Testdata)
      y_prob<-array(as.data.frame(predict(DE_XGBoost_classifier_model,Testdata,type = "prob")))
      
      y_true<-data_test$label
      y_true[which(y_true==0)]<-'No'
      y_true[which(y_true==1)]<-'Yes'
      meas<-confusionMatrix(as.factor(y_pred),as.factor(y_true),positive='Yes', mode = "prec_recall")
      prec[i]<-meas$byClass['Precision']
      f1[i]<-meas$byClass['F1']
      recal[i]<-meas$byClass['Recall']
      balan[i]<-meas$byClass['Balanced Accuracy']
      brier[i]<-BrierScore(data_test$label, y_prob[,2])
      auc[i]<-auc(as.factor(data_test$label),y_prob[,2])
      y_pred1<-as.factor(as.numeric(y_prob[,2] > 0.5))
      mcc1[i]<-mccr(data_test$label,y_pred1)
      gmean[i]<-GMEAN(data_test$label,y_pred1, 0, 1)
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
    s_result<-paste("E:/gln/C/mypackage/result_new/RQ1/samesetting/DT/SDP/",'c50/',files[j],".csv",sep='')
    write.table(df,s_result,row.names=FALSE,col.names=TRUE,sep=",")
  }
}
 

