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


#days<-c('jruby','log4j','lucene','poi','prop1','prop2','prop3','prop4','prop5','redaktor','synapse','tomcat','velocity','wicket','xalan','xerces')
#days<-c('DataClass','FeatureEnvy','GodClass','LongMethod')
days<-c('7days','14days','30days','90days','180days','365days')
#
file<-c('ofbiz','qpid')
#files<-c('camel','derby','eclipse','groovy','hbase','hive','ivy','jruby','log4j','lucene','poi','prop1','prop2','prop3','prop4','prop5','redaktor','synapse','tomcat','velocity','wicket','xalan','xerces')


for (k in 1:length(days)){
  for (j in 1:length(file)){
    s<-paste('C:/gln/mypackage/dataset/dataset/data_new/issue_close_time/', days[k] ,'/' , file[j] ,'/','train/',sep='')
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
    
    rf_mtry<-vector(mode="numeric",length=100)
    rf_min_n<-vector(mode="numeric",length=100)
    rf_tree<-vector(mode="numeric",length=100)
    rf_iter<-vector(mode="character",length=100)
    
    for (i in 1:length(filename)){
      
      s1=paste('C:/gln/mypackage/dataset/dataset/data_new/issue_close_time/', days[k] ,'/' , file[j] ,'/','train/',filename[i],sep='')
      data_train<-read.csv(s1)
      data_train$issude_close_time<-as.factor(data_train$issude_close_time)
      data_train<-droplevels(data_train)
      s2=paste('C:/gln/mypackage/dataset/dataset/data_new/issue_close_time/', days[k] ,'/' , file[j] ,'/','test/',filename[i],sep='')
      data_test<-read.csv(s2)
      folds <- vfold_cv(data_train, v = 5)
      cell_pre_proc <-recipe(issude_close_time ~ ., data = data_train) %>%step_downsample(issude_close_time)
      rf_mod<-rand_forest(mode = "classification",mtry = tune(),trees = tune(),min_n = tune())%>%set_engine("ranger")
      
      rf_wflow <-workflow() %>%add_model(rf_mod) %>%add_recipe(cell_pre_proc)
      
      rf_set <-rf_wflow %>%dials::parameters() %>%update(mtry =mtry(c(100, 500)))
      rf_search_2 <- tune_bayes(rf_wflow, resamples = folds,iter = 10,metrics = metric_set(roc_auc),param_info = rf_set,control = control_bayes(verbose = TRUE))
      rf_best_model <- select_best(rf_search_2, "roc_auc")
      rf_tree[i]<-rf_best_model$trees
      rf_min_n[i]<-rf_best_model$min_n
      rf_mtry[i]<-rf_best_model$mtry
      rf_iter[i]<-rf_best_model$.config
      rf_final_model <- finalize_model(rf_mod, rf_best_model)
      final_workflow    <- rf_wflow %>% update_model(rf_final_model)
      rf_fit     <- fit(final_workflow, data = data_train)
      y_pred <- predict(rf_fit, data_test)
      y_prob<-predict(rf_fit,data_test,type = "prob")
      y_true<-data_test$issude_close_time
      y_pred1<-as.factor(as.numeric(y_prob[,2]>0.5))
      meas<-confusionMatrix(as.factor(y_pred1),as.factor(y_true),positive='1', mode = "prec_recall")
      prec[i]<-meas$byClass['Precision']
      f1[i]<-meas$byClass['F1']
      recal[i]<-meas$byClass['Recall']
      balan[i]<-meas$byClass['Balanced Accuracy']
      brier[i]<-BrierScore(y_true, y_prob$.pred_1)
      auc[i]<-auc(as.factor(y_true),y_prob$.pred_1)
      mcc1[i]<-mccr(y_true,y_pred)
      gmean[i]<-GMEAN(y_true, y_pred, 0, 1)
    }
    va<-cbind(auc,mcc1,recal,f1,brier,balan,prec,gmean)
    
    names <- c("AUC","MCC1","Recall","F1","Brierce","balance","precision","G-Mean")
    df[[names[1]]] <- as.numeric()
    df[[names[2]]] <- as.numeric()
    df[[names[3]]] <- as.numeric()
    df[[names[4]]] <- as.numeric()
    df[[names[5]]] <- as.numeric()
    df[[names[6]]] <- as.numeric()
    df[[names[7]]] <- as.numeric()
    df[[names[8]]] <- as.numeric()
    
    df <- rbind(df,va)
    s_result<-paste("C:/gln/mypackage/result/data_new_performance_py/R/nb/ranger/issueclosetime/",days[k], '/', file[j],".csv",sep='')
    write.table(df,s_result,row.names=FALSE,col.names=TRUE,sep=",")
    
    va_para<-cbind(rf_tree,rf_min_n,rf_mtry,rf_iter)
    names_para <- c("tree","min_n","mtry","iter")
    df_para[[names_para[1]]] <- as.numeric()
    df_para[[names_para[2]]] <- as.numeric()
    df_para[[names_para[3]]] <- as.numeric()
    df_para[[names_para[4]]] <- as.character()
    df_para <- rbind(df_para,va_para)
    s_para=paste("C:/gln/mypackage/result/data_new_performance_py/R/param/nb/ranger/issueclosetime/",days[k], '/', file[j],".csv",sep='')
    write.table(df_para,s_para,row.names=FALSE,col.names=TRUE,sep=",")
  }
}



