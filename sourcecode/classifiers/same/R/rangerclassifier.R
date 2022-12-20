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
library(ranger)
library(doParallel)
library(foreach)


#days<-c('DataClass','FeatureEnvy','GodClass','LongMethod')

#'eclipse','groovy','hbase','hive','ivy','jruby','log4j','lucene','poi','prop1','prop2','prop3','prop4','prop5','redaktor','synapse','tomcat','velocity','wicket','xalan','xerces'
days<-c('90days','180days','365days')
#'camel','cloudstack',
files<-c('camel','cloudstack','cocoon','deeplearning','hadoop','hive','node','ofbiz','qpid')
#files<-c('DataClass','FeatureEnvy','GodClass','LongMethod')
for (k in 1:length(days)){
  for (j in 1:length(files)){
    s<-paste("E:/gln/C/mypackage/dataset/dataset/issueclosetime/",days[k],'/',files[j],'/',"train/",sep='')
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
    
    dt_mtry<-vector(mode="numeric",length=100)
    
    
    for (i in 1:length(filename)){
      
      s1=paste("E:/gln/C/mypackage/dataset/dataset/issueclosetime/",days[k],'/',files[j],'/',"train/",filename[i],sep='')
      data_train<-read.csv(s1)
      data_train$issude_close_time<-as.factor(data_train$issude_close_time)
      levels(data_train$issude_close_time)<-c('No','Yes')
      #data_train<-droplevels(data_train)
      
      s2=paste("E:/gln/C/mypackage/dataset/dataset/issueclosetime/",days[k],'/',files[j],'/',"test/",filename[i],sep='')
      data_test<-read.csv(s2)
      
      max_iter <- 1 # maximum number of iterations
      pop_size <- 10 # population size
      CV_folds <- 5 # number of folds
      CV_repeats <- 3 # number of repeats
      set.seed(123)
      n_cores <- detectCores()-1
      
      train_control <- caret::trainControl(method = "cv", number = CV_folds, repeats = CV_repeats,classProbs = TRUE) 
      
      eval_function_XGBoost_classifier <- function(x,data,train_settings) {
        
        x1 <- x[1]
        x2 <-x[2]
        x3<-x[3]
        
        Traindata<-data[,-ncol(data)]
        Trainclass<-data$issude_close_time
        #Trainclass[which(Trainclass==0)]<-'No'
        #Trainclass[which(Trainclass==1)]<-'Yes'
        #Trainclass<-factor(Trainclass)
        
        XGBoost_classifier_model <- caret::train(Traindata, Trainclass,
                                                 method = "ranger",
                                                 trControl =train_control,
                                                 metric='auc',
                                                 tuneGrid = expand.grid(
                                                   mtry = round(x1), # number of boosting iterations
                                                   splitrule='gini',
                                                   min.node.size=round(x3)
                                                 ))
        resu=XGBoost_classifier_model$results$Accuracy
        return(resu)
      }
      
      
      mtry <- c(1,10)
      splitrule<-c('gini','gini')
      minnodesize<-c(1,1)
      lower = c(mtry[1],splitrule[1],minnodesize[1])
      upper = c(mtry[2],splitrule[2],minnodesize[2])
      
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
      dt_mtry [i]<-DE_solutions[1]
      
      
      # Grid of optimal hyperparameter values
      DE_XGBoost_classifier_grid <- expand.grid(
        mtry = round(DE_solutions[1]), 
        splitrule='gini',
        min.node.size=1
      )
      
      T0 <- Sys.time()
      cluster <- makeCluster(detectCores() - 1) # number of cores, convention to leave 1 core for OS
      registerDoParallel(cluster) # register the parallel processing
      
      set.seed(1)
      # Train model with optimal values
      DE_XGBoost_classifier_model <- caret::train(issude_close_time ~., 
                                                  data = data_train, 
                                                  method = "ranger",
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
    s_result<-paste("E:/gln/C/mypackage/result_new/RQ2/R/tune/DE/ranger/issueclosetime/",days[k],'/',files[j],".csv",sep='')
    write.table(df,s_result,row.names=FALSE,col.names=TRUE,sep=",")
    va_para<-cbind(dt_mtry)
    names_para <- c("mtry","splitrule")
    df_para<- data.frame()
    df_para[[names_para[1]]] <- as.numeric()
    df_para <- rbind(df_para,va_para)
    s_para=paste("E:/gln/C/mypackage/result_new/RQ2/R/tune/DE/param/ranger/issueclosetime/",days[k],'/',files[j],".csv",sep='')
    write.table(df_para, s_para,row.names=FALSE,col.names=TRUE,sep=",")
  }
}
