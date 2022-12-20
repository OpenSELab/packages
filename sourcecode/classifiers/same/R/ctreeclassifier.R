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
library(DEoptim)
library(parallel)
library(party)
library(doParallel)
library(foreach)


days<-c('camel','derby','eclipse','groovy','hbase','hive','ivy','jruby','log4j','lucene','poi','prop1','prop2','prop3','prop4','prop5','redaktor','synapse','tomcat','velocity','wicket','xalan','xerces')


for (k in 1:length(days)){
  s<-paste("E:/gln/C/mypackage/dataset/dataset/dataset/",days[k],'/',"s_train/",sep='')
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
  
  dt_mincriterion<-vector(mode="numeric",length=100)
  dt_maxdepth<-vector(mode="numeric",length=100)
  dt_minsplit<-vector(mode="numeric",length=100)
  
  
  for (i in 1:length(filename)){
    
    s1=paste("E:/gln/C/mypackage/dataset/dataset/dataset/",days[k],'/',"s_train/",filename[i],sep='')
    data_train<-read.csv(s1)
    data_train$label<-as.factor(data_train$label)
    levels(data_train$label)<-c('No','Yes')
    #data_train<-droplevels(data_train)
    
    s2=paste("E:/gln/C/mypackage/dataset/dataset/dataset/",days[k],'/',"s_test/",filename[i],sep='')
    data_test<-read.csv(s2)
    
    max_iter <- 5 # maximum number of iterations
    pop_size <- 10 # population size
    CV_folds <- 5 # number of folds
    CV_repeats <- 3 # number of repeats
    set.seed(123)
    n_cores <- detectCores()-1
    
    train_control <- caret::trainControl(method = "cv", number = CV_folds, repeats = CV_repeats,classProbs = TRUE) 
    
    eval_function_XGBoost_classifier <- function(x,data,train_settings) {
      
      x1 <- x[1]
      x2 <- x[2]
      x3 <- x[3]
      Traindata<-data[,-ncol(data)]
      Trainclass<-data$label
      #Trainclass[which(Trainclass==0)]<-'No'
      #Trainclass[which(Trainclass==1)]<-'Yes'
      #Trainclass<-factor(Trainclass)
      
      XGBoost_classifier_model <- caret::train(Traindata, Trainclass,
                                               method = "ctree",
                                               trControl =train_settings,
                                               metric='auc',
                                               tuneGrid = expand.grid(
                                                 mincriterion = round(x1), # number of boosting iterations
                                                 maxdepth=round(x2),
                                                 minsplit = x3
                                               ))
      resu=XGBoost_classifier_model$results$Accuracy
      return(resu)
    }
    
    
    mincriterion <- c(0.5,1)
    maxdepth<-c(2,40)
    minsplit<-c(2,40)
    
    lower = c(mincriterion[1],maxdepth[1],minsplit[1])
    upper = c(mincriterion[2],maxdepth[2],minsplit[2])
    
    DE_T0 <- Sys.time()
    # Run differential evolution algorithm
    DE_model_XGBoost_classifier <- DEoptim(eval_function_XGBoost_classifier, 
                                           lower =lower,
                                           upper = upper, 
                                           control = DEoptim.control(
                                             NP = pop_size, # population size
                                             itermax = max_iter, # maximum number of iterations
                                             CR = 0.5, # probability of crossover
                                             storepopfreq = 1, # store every population
                                             parallelType = 0 # run parallel processing
                                           ),
                                           data = data_train,
                                           train_settings = train_control
    )
    
    DE_solutions <- DE_model_XGBoost_classifier$optim$bestmem
    dt_mincriterion[i]<-DE_solutions[1]
    dt_maxdepth[i]<-DE_solutions[2]
    dt_minsplit[i]<-DE_solutions[3]
    
    # Grid of optimal hyperparameter values
    DE_XGBoost_classifier_grid <- expand.grid(
      mincriterion = round(DE_solutions[1]),  # learning rate, low value means model is more robust to overfitting
      maxdepth=round(DE_solutions[2]),
      minsplit = DE_solutions[3]
    )
    
    T0 <- Sys.time()
    cluster <- makeCluster(detectCores() - 1) # number of cores, convention to leave 1 core for OS
    registerDoParallel(cluster) # register the parallel processing
    
    set.seed(1)
    # Train model with optimal values
    DE_XGBoost_classifier_model <- caret::train(label ~., 
                                                data = data_train, 
                                                method = "ctree",
                                                tuneGrid = DE_XGBoost_classifier_grid
    )
    
    
    y_pred <- predict(DE_XGBoost_classifier_model, data_test)
    y_prob<-array(as.data.frame(predict(DE_XGBoost_classifier_model,data_test,type = "prob")))
    
    y_true<-data_test$label
    y_true[which(y_true==0)]<-'No'
    y_true[which(y_true==1)]<-'Yes'
    meas<-confusionMatrix(as.factor(y_pred),as.factor(y_true),positive='Yes', mode = "prec_recall")
    prec[i]<-meas$byClass['Precision']
    f1[i]<-meas$byClass['F1']
    recal[i]<-meas$byClass['Recall']
    balan[i]<-meas$byClass['Balanced Accuracy']
    brier[i]<-BrierScore(data_test$label, y_prob[,2])
    auc[i]<-auc(as.factor(data_test$label),y_prob[,2])
    y_pred1<-as.factor(as.numeric(y_prob[,2] > 0.5))
    mcc1[i]<-mccr(data_test$label,y_pred1)
    gmean[i]<-GMEAN(data_test$label,y_pred1, 0, 1)
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
  s_result<-paste("E:/gln/C/mypackage/result_new/RQ2/R/tune/DE/ctree/SDP/",days[k],".csv",sep='')
  write.table(df,s_result,row.names=FALSE,col.names=TRUE,,sep=",")
  va_para<-cbind(dt_mincriterion,dt_maxdepth,dt_minsplit)
  names_para <- c("mincriterion","maxdepth","minsplit")
  df_para<- data.frame()
  df_para[[names_para[1]]] <- as.numeric()
  df_para <- unlist(rbind(df_para,va_para))
  s_para=paste("E:/gln/C/mypackage/result_new/RQ2/R/tune/DE/ctree/SDP/parame/",days[k],".csv",sep='')
  write.table(df_para,s_para,row.names=FALSE,col.names=TRUE,,sep=",")
}

