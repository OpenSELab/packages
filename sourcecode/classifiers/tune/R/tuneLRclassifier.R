library(caret)
library(MLmetrics)
library(DescTools)
library(pROC)
library(mltools)
library(mccr)
library(mcc)
library(measures)
library(visdat)
library(tidyverse)
library(tidymodels)
library(patchwork)

#days<-c('7days')
#
#files<-c('node','ofbiz','qpid')
files<-c('camel','derby','eclipse','groovy','hbase','hive','ivy','jruby','log4j','lucene','poi','prop1','prop2','prop3','prop4','prop5','redaktor','synapse','tomcat','velocity','wicket','xalan','xerces')

#for (k in 1:length(days)){
  for (j in 1:length(files)){
    s<-paste("C:/gln/mypackage/dataset/dataset/dataset/",files[j],'/','s_train/',sep='')
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
    
    lr_penalty<-vector(mode="numeric",length=100)
    lr_mixture<-vector(mode="numeric",length=100)
    lr_iter<-vector(mode="character",length=100)
    
    for (i in 1:length(filename)){
      
      s1=paste("C:/gln/mypackage/dataset/dataset/dataset/",files[j],'/','s_train/',filename[i],sep='')
      data_train<-read.csv(s1)
      data_train$Smell<-as.factor(data_train$Smell)
      data_train<-droplevels(data_train)
      s2=paste("C:/gln/mypackage/dataset/dataset/dataset/",files[j],'/','s_test/',filename[i],sep='')
      data_test<-read.csv(s2)
      folds <- vfold_cv(data_train, v = 5)
      cell_pre_proc <-recipe(Smell ~ ., data = data_train) %>%step_downsample(Smell)
      LR_mod<-logistic_reg(mode = "classification",penalty = tune(),mixture = tune())%>%set_engine("glmnet")
      
      LR_wflow <-workflow() %>%add_model(LR_mod) %>%add_recipe(cell_pre_proc)
      
      LR_set <-LR_wflow %>%parameters()
      LR_search_2 <- tune_bayes(LR_wflow, resamples = folds,iter = 10,metrics = metric_set(roc_auc),param_info = LR_set,control = control_bayes(verbose = TRUE))
      LR_best_model <- select_best(LR_search_2, "roc_auc")
      lr_penalty[i]<-LR_best_model$penalty
      lr_mixture[i]<-LR_best_model$mixture
      lr_iter[i]<-LR_best_model$.config
      LR_final_model <- finalize_model(LR_mod, LR_best_model)
      final_workflow    <- LR_wflow %>% update_model(LR_final_model)
      LR_fit     <- fit(final_workflow, data = data_train)
      y_true<-data_test$Smell
      data_test$Smell<-as.factor(data_test$Smell)
      y_pred <- predict(LR_fit, data_test)
      y_prob<-predict(LR_fit,data_test,type = "prob")
      y_pred1<-as.factor(as.numeric(y_prob[,2]>0.5))
     
      meas<-confusionMatrix(as.factor(y_pred1),as.factor(y_true),positive='1', mode = "prec_recall")
      prec[i]<-meas$byClass['Precision']
      f1[i]<-meas$byClass['F1']
      recal[i]<-meas$byClass['Recall']
      balan[i]<-meas$byClass['Balanced Accuracy']
      
      brier[i]<-BrierScore(y_true, y_prob$.pred_1)
      auc[i]<-auc(y_true, y_prob$.pred_1)
      mcc1[i]<-mccr(y_true,y_pred1)
      gmean[i]<-GMEAN(y_true, y_pred1, 0, 1)
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
    s_result<-paste("C:/gln/mypackage/result/data_new_performance_py/R/nb/glnmet/Smell/",files[j],".csv",sep='')
    write.table(df,s_result,row.names=FALSE,col.names=TRUE,sep=",")
    va_para<-cbind(lr_penalty,lr_mixture,lr_iter)
    names_para <- c("penalty","mixture","iter")
    df_para[[names_para[1]]] <- as.numeric()
    df_para[[names_para[2]]] <- as.numeric()
    df_para[[names_para[3]]] <- as.character()
    df_para <- rbind(df_para,va_para)
    s_para=paste("C:/gln/mypackage/result/data_new_performance_py/R/param/nb/glnmet/Smell/",files[j],".csv",sep='')
    write.table(df_para,s_para,row.names=FALSE,col.names=TRUE,sep=",")
  }
#}
