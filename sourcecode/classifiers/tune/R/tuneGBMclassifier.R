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

gbm_tree<-vector(mode="numeric",length=100)
gbm_learn_rate<-vector(mode="numeric",length=100)
gbm_iter<-vector(mode="character",length=100)

for (i in 1:length(filename)){
  
  s1=paste("E:/SDP/Classifiers/xerces/s_train/",'/',filename[i],sep='')
  data_train<-read.csv(s1)
  data_train$label<-as.factor(data_train$label)
  data_train<-droplevels(data_train)
  s2=paste("E:/SDP/Classifiers/xerces/s_test/",'/',filename[i],sep='')
  data_test<-read.csv(s2)
  folds <- vfold_cv(data_train, v = 5)
  cell_pre_proc <-recipe(label ~ ., data = data_train) %>%step_downsample(label)
  GBM_mod<- boost_tree(mode = "classification", trees = tune(),learn_rate = tune()) %>%set_engine("xgboost")
  
  GBM_wflow <-workflow() %>%add_model(GBM_mod) %>%add_recipe(cell_pre_proc)
  
  GBM_set <-GBM_wflow %>%parameters()%>%update(trees = trees(c(100, 500)))
  GBM_search_2 <- tune_bayes(GBM_wflow, resamples = folds,iter = 100,metrics = metric_set(roc_auc),param_info = GBM_set,control = control_bayes(verbose = TRUE))
  GBM_best_model <- select_best(GBM_search_2, "roc_auc")
  gbm_tree[i]<-GBM_best_model$trees
  gbm_learn_rate[i]<-GBM_best_model$learn_rate
  gbm_iter[i]<-GBM_best_model$.config
  GBM_final_model <- finalize_model(GBM_mod, GBM_best_model)
  final_workflow    <- GBM_wflow %>% update_model(GBM_final_model)
  GBM_fit     <- fit(final_workflow, data = data_train)
  y_pred <- predict(GBM_fit, data_test)
  y_prob<-predict(GBM_fit,data_test,type = "prob")
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
write.table(df,"E:/SDP/TuneClassifiers/result/R/GBM/xerces.csv",row.names=FALSE,col.names=TRUE,sep=",")
va_para<-cbind(gbm_tree,gbm_learn_rate,gbm_iter)
names_para <- c("tree","learn_rate","iter")
df_para[[names_para[1]]] <- as.numeric()
df_para[[names_para[2]]] <- as.numeric()
df_para[[names_para[3]]] <- as.character()
df_para <- rbind(df_para,va_para)
write.table(df_para,"E:/SDP/TuneClassifiers/result/Parameters/GBM/xerces.csv",row.names=FALSE,col.names=TRUE,sep=",")