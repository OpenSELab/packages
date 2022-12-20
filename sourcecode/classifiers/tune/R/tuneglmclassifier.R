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

days<-c('1days','7days','14days','30days','90days','180days','365days')
#
files<-c('camel','cloudstack','cocoon','deeplearning','hadoop','hive','node','ofbiz','qpid')
#files<-c('DataClass','FeatureEnvy','GodClass','LongMethod')
#files<-c('camel','derby','eclipse','groovy','hbase','hive','ivy','jruby','log4j','lucene','poi','prop1','prop2','prop3','prop4','prop5','redaktor','synapse','tomcat','velocity','wicket','xalan','xerces')
for (k in 1:length(days)){
  for (j in 1:length(files)){
    s<-paste("C:/gln/mypackage/dataset/dataset/data_new/issue_close_time/",days[k],'/',files[j],'/','train/',sep='')
    filename <- list.files(s)
    df<- data.frame()
    #df_para<-data.frame()
    auc<-vector(mode="numeric",length=100)
    mcc1<-vector(mode="numeric",length=100)
    recal<-vector(mode="numeric",length=100)
    f1<-vector(mode="numeric",length=100)
    brier<-vector(mode="numeric",length=100)
    balan<-vector(mode="numeric",length=100)
    prec<-vector(mode="numeric",length=100)
    gmean<-vector(mode="numeric",length=100)
    
  
    
    for (i in 1:length(filename)){
      
      s1=paste("C:/gln/mypackage/dataset/dataset/data_new/issue_close_time/",days[k],'/',files[j],'/','train/',filename[i],sep='')
      data_train<-read.csv(s1)
    
      Traindata<-data_train[,-ncol(data_train)]
      Trainclass<-data_train$issude_close_time
      Trainclass[which(Trainclass==0)]<-'No'
      Trainclass[which(Trainclass==1)]<-'Yes'
      data_train$issude_close_time<-as.factor( data_train$issude_close_time)
      
      
      s2=paste("C:/gln/mypackage/dataset/dataset/data_new/issue_close_time/",days[k],'/',files[j],'/','test/',filename[i],sep='')
      data_test<-read.csv(s2)
     # LR_mod<-logistic_reg()%>%set_engine("glm")
      
      #LR_fit<-LR_mod%>%fit(Smell~.,data=data_train)
      DT_model<-train(Traindata,Trainclass,method = "glm",family="binomial")
      
      y_pred <- predict(DT_model, data_test)
      y_prob<-array(as.data.frame(predict(DT_model,data_test,type = "prob")))
      
      y_true<-data_test$issude_close_time
      y_true[which(y_true==0)]<-'No'
      y_true[which(y_true==1)]<-'Yes'
      meas<-confusionMatrix(as.factor(y_pred),as.factor(y_true),positive='Yes', mode = "prec_recall")
      prec[i]<-meas$byClass['Precision']
      f1[i]<-meas$byClass['F1']
      recal[i]<-meas$byClass['Recall']
      balan[i]<-meas$byClass['Balanced Accuracy']
      brier[i]<-BrierScore(data_test$issude_close_time, y_prob[,2])
      auc[i]<-auc(as.factor(data_test$issude_close_time),y_prob[,2])
      y_pred1<-as.factor(as.numeric(y_prob[,2] > 0.5))
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
    s_result<-paste("C:/gln/mypackage/result/data_new_performance_py/R/nb/glm/issueclosetime/",days[k],'/',files[j],".csv",sep='')
    write.table(df,s_result,row.names=FALSE,col.names=TRUE,sep=",")
  }
}
