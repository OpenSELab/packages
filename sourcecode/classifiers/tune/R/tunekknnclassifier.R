library(caret)
library(MLmetrics)
library(DescTools)
library(pROC)
library(mltools)
library(mccr)
library(mcc)
library(measures)
library(kknn)
library(visdat)
library(tidyverse)
library(tidymodels)
library(patchwork)

#days<-c('1days')
#
files<-c('synapse')

#files<-c('DataClass','FeatureEnvy','GodClass','LongMethod')
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
    
    knn_neighbor<-vector(mode="numeric",length=100)
    knn_dist_power<-vector(mode="numeric",length=100)
    knn_iter<-vector(mode="character",length=100)
    
    for (i in 1:length(filename)){
      s1=paste("C:/gln/mypackage/dataset/dataset/dataset/",files[j],'/','s_train/',filename[i],sep='')
      data_train<-read.csv(s1)
      data_train$label<-as.factor(data_train$label)
      #data_train<-droplevels(data_train)
      s2=paste("C:/gln/mypackage/dataset/dataset/dataset/",files[j],'/','s_test/',filename[i],sep='')
      data_test<-read.csv(s2)
      folds <- vfold_cv(data_train, v = 5)
      cell_pre_proc <-recipe(label ~ ., data = data_train) %>%step_downsample(label)
      knn_mod<-nearest_neighbor(mode = "classification",neighbors = tune(),dist_power = tune())%>%set_engine("kknn")
      
      knn_wflow <-workflow() %>%add_model(knn_mod) %>%add_recipe(cell_pre_proc)
      
      knn_set <-knn_wflow %>%parameters() %>%update(neighbors = neighbors(c(1, 30))) %>%
        update(dist_power = dist_power(c(1/4, 2)))
      knn_search_2 <- tune_bayes(knn_wflow, resamples = folds,iter = 10,metrics = metric_set(roc_auc),param_info = knn_set,control = control_bayes(verbose = TRUE))
      knn_best_model <- select_best(knn_search_2, "roc_auc")
      knn_neighbor[i]<-knn_best_model$neighbors
      knn_dist_power[i]<-knn_best_model$dist_power
      knn_iter[i]<-knn_best_model$.config
      knn_final_model <- finalize_model(knn_mod, knn_best_model)
      final_workflow    <- knn_wflow %>% update_model(knn_final_model)
      knn_fit     <- fit(final_workflow, data = data_train)
      y_pred <- predict(knn_fit, data_test)
      y_prob<-predict(knn_fit,data_test,type = "prob")
      y_true<-data_test$label
      meas<-confusionMatrix(as.factor(y_pred$.pred_class),as.factor(y_true),positive='1', mode = "prec_recall")
      prec[i]<-meas$byClass['Precision']
      f1[i]<-meas$byClass['F1']
      recal[i]<-meas$byClass['Recall']
      balan[i]<-meas$byClass['Balanced Accuracy']
      brier[i]<-BrierScore(y_true, y_prob$.pred_1)
      auc[i]<-auc(as.factor(y_true),y_prob$.pred_1)
      mcc1[i]<-mccr(y_true,y_pred$.pred_class)
      gmean[i]<-GMEAN(y_true, y_pred$.pred_class, 0, 1)
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
    s_result<-paste("C:/gln/mypackage/result/data_new_performance_py/R/nb/kknn/SDP/",files[j],".csv",sep='')
    write.table(df,s_result,row.names=FALSE,col.names=TRUE,sep=",")
    va_para<-cbind(knn_neighbor,knn_dist_power,knn_iter)
    names_para <- c("neighbor","dist_power","iter")
    df_para<- data.frame()
    df_para[[names_para[1]]] <- as.numeric()
    df_para[[names_para[2]]] <- as.numeric()
    df_para[[names_para[3]]] <- as.character()
    df_para <- rbind(df_para,va_para)
    s_para=paste("C:/gln/mypackage/result/data_new_performance_py/R/param/nb/kknn/SDP/",files[j],".csv",sep='')
    write.table(df_para,s_para,row.names=FALSE,col.names=TRUE,sep=",")
  }
#}