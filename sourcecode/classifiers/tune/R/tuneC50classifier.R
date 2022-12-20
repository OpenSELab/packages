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
#days<-c('90days')
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
    
    #dt_cost<-vector(mode="numeric",length=100)
    dt_tree<-vector(mode="numeric",length=100)
    #dt_min_n<-vector(mode="numeric",length=100)
    dt_iter<-vector(mode="character",length=100)
    
    for (i in 1:length(filename)){
      
      s1=paste("C:/gln/mypackage/dataset/dataset/dataset/",files[j],'/','s_train/',filename[i],sep='')
      data_train<-read.csv(s1)
      data_train$label<-as.factor(data_train$label)
      data_train<-droplevels(data_train)
      s2=paste("C:/gln/mypackage/dataset/dataset/dataset/",files[j],'/','s_test/',filename[i],sep='')
      data_test<-read.csv(s2)
      folds <- vfold_cv(data_train, v = 5)
      cell_pre_proc <-recipe(label ~ ., data = data_train) %>%step_downsample(label)
      DT_mod<-boost_tree(mode = "classification", trees =tune()) %>%set_engine("C5.0")
      
      DT_wflow <-workflow() %>%add_model(DT_mod) %>%add_recipe(cell_pre_proc)
      
      DT_set <-dials::parameters(DT_wflow)
      
      DT_set <- DT_set%>%update(trees = trees(c(1,30)))
      DT_search_2 <- tune_bayes(DT_wflow, resamples = folds,iter = 10,metrics = metric_set(roc_auc),param_info = DT_set,control = control_bayes(verbose = TRUE))
      DT_best_model <- select_best(DT_search_2, "roc_auc")
      #dt_cost[i]<-DT_best_model$cost_complexity
      dt_tree[i]<-DT_best_model$trees
      #dt_min_n[i]<-DT_best_model$min_n
      dt_iter[i]<-DT_best_model$.config
      DT_final_model <- finalize_model(DT_mod, DT_best_model)
      final_workflow    <- DT_wflow %>% update_model(DT_final_model)
      DT_fit     <-fit(final_workflow, data = data_train)
      y_pred <- predict(DT_fit, data_test)
      y_prob<-predict(DT_fit,data_test,type = "prob")
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
    s_result<-paste("
                    ",files[j],".csv",sep='')
    write.table(df,s_result,row.names=FALSE,col.names=TRUE,sep=",")
    va_para<-cbind(dt_tree,dt_iter)
    #names_para <- c("depth","min_n","iter")
    df_para[[names_para[1]]] <- as.numeric()
    df_para[[names_para[2]]] <- as.numeric()
    #df_para[[names_para[3]]] <- as.numeric()                            
   # df_para[[names_para[4]]] <- as.character()
    df_para <- rbind(df_para,va_para)
    s_para=paste("C:/gln/mypackage/result/data_new_performance_py/R/param/nb/C50/SDP/",files[j],".csv",sep='')
    write.table(df_para,s_para,row.names=FALSE,col.names=TRUE,sep=",")
  }

#}
