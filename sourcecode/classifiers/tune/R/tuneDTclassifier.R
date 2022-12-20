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

filename <- list.files("E:/SDP/Classifiers/xerces/s_train/")
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

dt_cost<-vector(mode="numeric",length=100)
dt_tree_depth<-vector(mode="numeric",length=100)
dt_min_n<-vector(mode="numeric",length=100)
dt_iter<-vector(mode="character",length=100)

for (i in 1:length(filename)){
  
  s1=paste("E:/SDP/Classifiers/xerces/s_train/",'/',filename[i],sep='')
  data_train<-read.csv(s1)
  data_train$label<-as.factor(data_train$label)
  data_train<-droplevels(data_train)
  s2=paste("E:/SDP/Classifiers/xerces/s_test/",'/',filename[i],sep='')
  data_test<-read.csv(s2)
  folds <- vfold_cv(data_train, v = 5)
  cell_pre_proc <-recipe(label ~ ., data = data_train) %>%step_downsample(label)
  DT_mod<-decision_tree(mode = "classification",cost_complexity = tune(),tree_depth = tune(),min_n = tune())%>%set_engine("rpart")
  
  DT_wflow <-workflow() %>%add_model(DT_mod) %>%add_recipe(cell_pre_proc)
  
  DT_set <-DT_wflow %>%parameters() %>%update(tree_depth = tree_depth(c(1, 30)))
  DT_search_2 <- tune_bayes(DT_wflow, resamples = folds,iter = 100,metrics = metric_set(roc_auc),param_info = DT_set,control = control_bayes(verbose = TRUE))
  DT_best_model <- select_best(DT_search_2, "roc_auc")
  dt_cost[i]<-DT_best_model$cost_complexity
  dt_tree_depth[i]<-DT_best_model$tree_depth
  dt_min_n[i]<-DT_best_model$min_n
  dt_iter[i]<-DT_best_model$.config
  DT_final_model <- finalize_model(DT_mod, DT_best_model)
  final_workflow    <- DT_wflow %>% update_model(DT_final_model)
  DT_fit     <- fit(final_workflow, data = data_train)
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
write.table(df,"E:/SDP/TuneClassifiers/result/R/DT/xerces.csv",row.names=FALSE,col.names=TRUE,sep=",")
va_para<-cbind(dt_cost,dt_tree_depth,dt_min_n,dt_iter)
names_para <- c("cost","depth","min_n","iter")
df_para[[names_para[1]]] <- as.numeric()
df_para[[names_para[2]]] <- as.numeric()
df_para[[names_para[3]]] <- as.numeric()                            
df_para[[names_para[4]]] <- as.character()
df_para <- rbind(df_para,va_para)
write.table(df_para,"E:/SDP/TuneClassifiers/result/Parameters/DT/xerces.csv",row.names=FALSE,col.names=TRUE,sep=",")