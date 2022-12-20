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
library(doParallel)
library(foreach)


#days<-c('eclipse','groovy','hbase','hive','ivy','jruby','log4j','lucene','poi','prop1','prop2','prop3','prop4','prop5','redaktor','synapse','tomcat','velocity','wicket','xalan','xerces')
days<-c('180days')
files<-c('ofbiz','qpid')
for (j in 1:length(days)){
  for (k in 1:length(files)){
    s<-paste("E:/gln/C/mypackage/dataset/dataset/issueclosetime/",days[j],'/',files[k],'/',"train/",sep='')
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
    
    dt_nrounds<-vector(mode="numeric",length=100)
    dt_max_depth<-vector(mode="numeric",length=100)
    dt_eta<-vector(mode="numeric",length=100)
    
    
    for (i in 1:length(filename)){
      
      s1=paste("E:/gln/C/mypackage/dataset/dataset/issueclosetime/",days[j],'/',files[k],'/',"train/",filename[i],sep='')
      data_train<-read.csv(s1)
      data_train$issude_close_time<-as.factor(data_train$issude_close_time)
      levels(data_train$issude_close_time)<-c('No','Yes')
      #data_train<-droplevels(data_train)
      
      s2=paste("E:/gln/C/mypackage/dataset/dataset/issueclosetime/",days[j],'/',files[k],'/',"test/",filename[i],sep='')
      data_test<-read.csv(s2)
      
      max_iter <- 2 # maximum number of iterations
      pop_size <- 5 # population size
      CV_folds <- 5 # number of folds
      CV_repeats <- 3 # number of repeats
      set.seed(123)
      n_cores <- detectCores()-1
      
      train_control <- caret::trainControl(method = "cv", number = CV_folds, repeats = CV_repeats,classProbs = TRUE) 
      
      eval_function_XGBoost_classifier <- function(x,data_train,train_settings) {
        
        x1 <- x[1]
        x2 <- x[2]
        x3 <- x[3]
        x4 <- x[4]
        x5 <- x[5]
        x6 <- x[6]
        x7 <- x[7]
        Traindata<-data_train[,-ncol(data_train)]
        Trainclass<-data_train$issude_close_time
        #Trainclass[which(Trainclass==0)]<-'No'
        #Trainclass[which(Trainclass==1)]<-'Yes'
        #Trainclass<-factor(Trainclass)
        
        XGBoost_classifier_model <- caret::train(Traindata, data_train$issude_close_time,
                                                 method = "xgbTree",
                                                 trControl =train_settings,
                                                 metric='auc',
                                                 tuneGrid = expand.grid(
                                                   nrounds = round(x1), # number of boosting iterations
                                                   max_depth=round(x2),
                                                   eta = x3, # learning rate, low value means model is more robust to overfitting
                                                   gamma =x4, # L1 Regularization (equivalent to Lasso Regression) on weights
                                                   colsample_bytree=x5,
                                                   min_child_weight=x6,
                                                   subsample=x7
                                                 ))
        resu=XGBoost_classifier_model$results$Accuracy
        return(resu)
      }
      
      
      nrounds_min_max <- c(50,500)
      max_depth<-c(2,15)
      eta<-c(0.0001,1)
      gamma<-c(0.5,0.5)
      colsample_bytree<-c(1,1)
      min_child_weight<-c(1,1)
      subsample<-c(0.5,0.5)
      lower = c(nrounds_min_max[1],max_depth[1],eta[1],gamma[1],colsample_bytree[1],min_child_weight[1],subsample[1])
      upper = c(nrounds_min_max[2],max_depth[2],eta[2],gamma[2],colsample_bytree[2],min_child_weight[2],subsample[2])
      
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
                                             data_train= data_train,
                                             train_settings = train_control
      )
      
      DE_solutions <- DE_model_XGBoost_classifier$optim$bestmem
      dt_nrounds[i]<-DE_solutions[1]
      dt_max_depth[i]<-DE_solutions[2]
      dt_eta[i]<-DE_solutions[3]
      
      # Grid of optimal hyperparameter values
      DE_XGBoost_classifier_grid <- expand.grid(
        nrounds = round(DE_solutions[1]),  # learning rate, low value means model is more robust to overfitting
        max_depth=round(DE_solutions[2]),
        eta = DE_solutions[3], # learning rate, low value means model is more robust to overfitting
        gamma =DE_solutions[4], # L1 Regularization (equivalent to Lasso Regression) on weights
        colsample_bytree=DE_solutions[5],
        min_child_weight=DE_solutions[6],
        subsample=DE_solutions[7]
      )
      
      T0 <- Sys.time()
      cluster <- makeCluster(detectCores() - 1) # number of cores, convention to leave 1 core for OS
      registerDoParallel(cluster) # register the parallel processing
      
      set.seed(1)
      # Train model with optimal values
      DE_XGBoost_classifier_model <- caret::train(issude_close_time ~., 
                                                  data = data_train, 
                                                  method = "xgbTree",
                                                  trControl = train_control,
                                                  verbose = F, metric = "auc", maximize = FALSE,
                                                  silent = 1,
                                                  # tuneLength = 1
                                                  tuneGrid = DE_XGBoost_classifier_grid
      )
      
      
      y_pred <- predict(DE_XGBoost_classifier_model, data_test)
      y_prob<-array(as.data.frame(predict(DE_XGBoost_classifier_model,data_test,type = "prob")))
      
      y_true<-data_test$issude_close_time
      y_true[which(y_true==0)]<-'No'
      y_true[which(y_true==1)]<-'Yes'
      meas<-confusionMatrix(as.factor(y_pred),as.factor(y_true),positive='Yes', mode = "prec_recall")
      prec[i]<-meas$byClass['Precision']
      f1[i]<-meas$byClass['F1']
      recal[i]<-meas$byClass['Recall']
      balan[i]<-meas$byClass['Balanced Accuracy']
      brier[i]<-BrierScore(data_test$issude_close_time, y_prob[,2])
      auc[i]<-auc(as.factor(data_test$issude_close_time),y_prob[,2])
      y_pred1<-as.factor(as.numeric(y_prob[,2] > 0.5))
      mcc1[i]<-mccr(data_test$issude_close_time,y_pred1)
      gmean[i]<-GMEAN(data_test$issude_close_time,y_pred1, 0, 1)
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
    s_result<-paste("E:/gln/C/mypackage/result_new/RQ2/R/tune/DE/xgboost/issueclosetime/",days[j],'/',files[k],".csv",sep='')
    write.table(df,s_result,row.names=FALSE,col.names=TRUE,sep=",")
    va_para<-cbind(dt_nrounds,dt_max_depth,dt_eta)
    names_para <- c("nrounds","max_depth","eta")
    df_para<- data.frame()
    df_para[[names_para[1]]] <- as.numeric()
    df_para <- unlist(rbind(df_para,va_para))
    #colnames(df_para)<-c("nrounds","max_depth","eta")
    s_para=paste("E:/gln/C/mypackage/result_new/RQ2/R/tune/DE/param/xgboost/issueclosetime/",days[j],'/',files[k],".csv",sep='')
    write.table(df_para, s_para,row.names=FALSE,col.names=TRUE,sep=",")
  }

}