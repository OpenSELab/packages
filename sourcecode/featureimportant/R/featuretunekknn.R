library(iml)
library(mlr3)
library(dplyr)
library(patchwork)
library(mlr3verse)
library(mlr3learners)
library(mlr3extralearners)
library(tibble)
library(tidyverse)
library(kknn)
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

files<-c('camel','derby','eclipse','groovy','hbase','hive','ivy','jruby','log4j','lucene','poi','prop1','prop2','prop3','prop4','prop5','redaktor','synapse','tomcat','velocity','wicket','xalan','xerces')#

for (j in 1:length(files)){
  s_feature<-paste("E:/gln/C/mypackage/autospearmandataset/",files[j],'.csv',sep='')
  df_feature<-read.csv(s_feature)
  df_feature<-df_feature[,-ncol(df_feature)]
  features<-names(df_feature)
  
  s_parameters<-paste("E:/gln/C/mypackage/result_new/RQ2/R/tune/nb/paras/nb/kknn/SDP/",files[j],'.csv',sep='')
  df_parameters<-read.csv(s_parameters)
  kk<-df_parameters[,1]
  
  n<-length(features)
 
  
  #train_features_KNN<-matrix(0,n,100)
  test_features_KkNN<-matrix(0,n-1,100)
  
 
  
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
    
    #KNN--------------------------------------
    learner_KNN = lrn("classif.kknn",k=kk[i],predict_type = "prob")
    learner_KNN$train(task)
    model_test_KNN = Predictor$new(learner_KNN, data = test, y = "label")
    effect_test_KNN = FeatureImp$new(model_test_KNN, loss = "ce" )
    test_features_KkNN[,i]<-effect_test_KNN$results$importance
    
  }
  
  df_test_KNN<-as.data.frame(cbind(features,test_features_KNN))
  s_KNN<-paste("E:/gln/C/mypackage/result_new/RQ3/featureimportant/tunesetting/KNN/kknn/",files[j],'.csv',sep='')
  write.table(df_test_KNN,s_KNN,row.names=FALSE,col.names=TRUE,sep=",")
  
 
}

