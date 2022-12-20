library(iml)
library(mlr3)
library(dplyr)
library(patchwork)
library(mlr3verse)
library(mlr3learners)
library(mlr3extralearners)
library(tibble)
library(tidyverse)
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
library(conflicted)
library(themis)
library(xgboost)

files<-c('camel','derby','eclipse','groovy','hbase','hive','ivy','jruby','log4j','lucene','poi','prop1','prop2','prop3','prop4','prop5','redaktor','synapse','tomcat','velocity','wicket','xalan','xerces')#

for (j in 1:length(files)){
  s_feature<-paste("E:/gln/C/mypackage/autospearmandataset/",files[j],'.csv',sep='')
  df_feature<-read.csv(s_feature)
  df_feature<-df_feature[,-ncol(df_feature)]
  features<-names(df_feature)
  
  s_parameters<-paste("E:/gln/C/mypackage/result_new/RQ2/R/tune/nb/paras/nb/xgboost/SDP/",files[j],'.csv',sep='')
  df_parameters<-read.csv(s_parameters)
  tree<-df_parameters[,1]
  
  n<-length(features)
  
  #train_features_GBM<-matrix(0,n,100)
  test_features_GBM<-matrix(0,n-1,100)
  
  s<-paste("E:/gln/C/mypackage/dataset/dataset/dataset/",files[j],'/s_train/',sep='')
  filename <- list.files(s)
  
  for (i in 1:length(filename)){
    
    s1=paste("E:/gln/C/mypackage/dataset/dataset/dataset/",files[j],'/s_train/',filename[i],sep='')
    data_train<-read.csv(s1)
    s2=paste("E:/gln/C/mypackage/dataset/dataset/dataset/",files[j],'/s_test/',filename[i],sep='')
    data_test<-read.csv(s2)
    data_train<-dplyr::select(data_train,features)
    data_train$label<-as.factor(data_train$label)
    data_test<-dplyr::select(data_test,features)
    data_test$label<-as.factor(data_test$label)
    train<-data_train[,]
    test<-data_test[,]
    Traindata<-train[,-ncol(train)]
    Trainclass<-train$label
    Testdata<-test[,-ncol(test)]
    Testclass<-test$label
    
    
    task <-as_task_classif(id="projects", train,target = "label")
    
    #GBM--------------------------------------
    
    learner_GBM = lrn("classif.xgboost", nrounds = tree[i],predict_type = "prob")
    learner_GBM$train(task)
    model_test_GBM = Predictor$new(learner_GBM, data = test, y = "label")
    effect_test_GBM = FeatureImp$new(model_test_GBM, loss = "ce" )
    test_features_GBM[,i]<-effect_test_GBM$results$importance
    
   
  }

  
  df_test_GBM<-as.data.frame(cbind(features,test_features_GBM))
  s_GBM<-paste("E:/gln/C/mypackage/result_new/RQ3/featureimportant/tunesetting/xgboost/xgboost/",files[j],'.csv',sep='')
  write.table(df_test_GBM,s_GBM,row.names=FALSE,col.names=TRUE,sep=",")
  
}

