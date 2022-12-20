library(caret)
library(MLmetrics)
library(DescTools)
library(pROC)
library(mltools)
library(mccr)
library(mcc)
library(measures)
library(gbm)
library(visdat)
library(tidyverse)
library(tidymodels)
library(patchwork)
library(parsnip)
library(conflicted)

#days<-c('camel','derby','eclipse','groovy','hbase','hive','ivy','jruby','log4j','lucene','poi','prop1','prop2','prop3','prop4','prop5','redaktor','synapse','tomcat','velocity','wicket','xalan','xerces')

days<-c('7days','14days','30days','90days','180days','365days')
#
files<-c('camel','cloudstack','cocoon','deeplearning','hadoop','hive','node','ofbiz','qpid')

#files<-c('DataClass','FeatureEnvy','GodClass','LongMethod')
for (k in 1:length(days)){
  for (j in 1:length(files)){
    s<-paste("C:/gln/mypackage/dataset/dataset/data_new/issue_close_time/",days[k],'/',files[j],'/','train/',sep='')
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
    
    gbm_tree<-vector(mode="numeric",length=100)
    gbm_learn_rate<-vector(mode="numeric",length=100)
    gbm_iter<-vector(mode="character",length=100)
    
    for (i in 1:length(filename)){
      
      s1=paste("C:/gln/mypackage/dataset/dataset/data_new/issue_close_time/",days[k],'/',files[j],'/','train/',filename[i],sep='')
      data_train<-read.csv(s1)
      data_train$issude_close_time<-as.factor(data_train$issude_close_time)
      data_train<-droplevels(data_train)
      s2=paste("C:/gln/mypackage/dataset/dataset/data_new/issue_close_time/",days[k],'/',files[j],'/','test/',filename[i],sep='')
      data_test<-read.csv(s2)
      folds <- vfold_cv(data_train, v = 5)
      cell_pre_proc <-recipe(issude_close_time ~ ., data = data_train) %>%step_downsample(issude_close_time)
      GBM_mod<- boost_tree(mode = "classification", trees =tune()) %>%set_engine("xgboost")
      
      GBM_wflow <-workflow() %>%add_model(GBM_mod) %>%add_recipe(cell_pre_proc)
      GBM_set <-dials::parameters(GBM_wflow)
      
      GBM_set <- GBM_set%>%update(trees = trees(c(50,500)))
      GBM_search_2 <- tune_bayes(GBM_wflow, resamples = folds,iter = 10,metrics = metric_set(roc_auc),param_info = GBM_set,control = control_bayes(verbose = TRUE))
      GBM_best_model <- select_best(GBM_search_2, "roc_auc")
      gbm_tree[i]<-GBM_best_model$trees
     # gbm_learn_rate[i]<-GBM_best_model$learn_rate
      gbm_iter[i]<-GBM_best_model$.config
      GBM_final_model <- finalize_model(GBM_mod, GBM_best_model)
      final_workflow    <- GBM_wflow %>% update_model(GBM_final_model)
      GBM_fit     <- fit(final_workflow, data = data_train)
      y_pred <- predict(GBM_fit, data_test)
      y_prob<-predict(GBM_fit,data_test,type = "prob")
      y_true<-data_test$issude_close_time
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
    s_result<-paste("C:/gln/mypackage/result/data_new_performance_py/R/nb/xgboost/issueclosetime/",days[k],'/',files[j],".csv",sep='')
    write.table(df,s_result,row.names=FALSE,col.names=TRUE,sep=",")
    va_para<-cbind(gbm_tree,gbm_iter)
    names_para <- c("tree","iter")
    #df_para[[names_para[1]]] <- as.numeric()
    #df_para[[names_para[2]]] <- as.numeric()
    #df_para[[names_para[3]]] <- as.character()
    df_para <- rbind(df_para,va_para)
    s_para=paste("C:/gln/mypackage/result/data_new_performance_py/R/param/nb/xgboost/issueclosetime/",days[k],'/',files[j],".csv",sep='')
    write.table(df_para,s_para,row.names=FALSE,col.names=TRUE,sep=",")
  }
}