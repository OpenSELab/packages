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



#days<-c('camel','eclipse','groovy','hbase','hive','ivy','jruby','log4j','lucene','poi','prop1','prop2','prop3','prop4','prop5','redaktor','synapse','tomcat','velocity','wicket','xalan','xerces')
days<-c('30days')
files<-c('hive','node','ofbiz','qpid')
#'camel','cloudstack','cocoon','deeplearning',
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
    
    dt_cp<-vector(mode="numeric",length=100)
    
    dt_maxdepth<-vector(mode="numeric",length=100)
    dt_minsplit<-vector(mode="numeric",length=100)
    
    for (i in 1:length(filename)){
      
      s1=paste("E:/gln/C/mypackage/dataset/dataset/issueclosetime/",days[j],'/',files[k],'/',"train/",filename[i],sep='')
      data_train<-read.csv(s1)
      data_train$issude_close_time<-as.factor(data_train$issude_close_time)
      levels(data_train$issude_close_time)<-c('No','Yes')
      #data_train<-droplevels(data_train)
      Traindata<-data_train[,-ncol(data_train)]
      Trainclass<-data_train$issude_close_time
      
      s2=paste("E:/gln/C/mypackage/dataset/dataset/issueclosetime/",days[j],'/',files[k],'/',"test/",filename[i],sep='')
      data_test<-read.csv(s2)
      Testdata<-data_test[,-ncol(data)]
  
      
      max_iter <- 1 # maximum number of iterations
      pop_size <- 10 # population size
      CV_folds <- 5 # number of folds
      CV_repeats <- 3 # number of repeats
      set.seed(123)
      n_cores <- detectCores()-1
      
      train_control <- caret::trainControl(method = "cv", number = CV_folds, repeats = CV_repeats,classProbs = TRUE) 
      
      eval_function_XGBoost_classifier <- function(x,data,train_settings) {
        
        x1 <- x[1]
        x2 <- x[2]

        Traindata<-data[,-ncol(data)]
        Trainclass<-data$issude_close_time
        #Trainclass[which(Trainclass==0)]<-'No'
        #Trainclass[which(Trainclass==1)]<-'Yes'
        #Trainclass<-factor(Trainclass)
        
        XGBoost_classifier_model <- caret::train(Traindata, Trainclass,
                                                 method = "rpart",
                                                 trControl =train_settings,
                                                 metric='auc',
                                                 tuneGrid = expand.grid(
                                                   cp = x1
                                                 ))
        resu=XGBoost_classifier_model$results$Accuracy
        return(resu)
      }
      
      
      cp <- c(0.001,0.02)
     # maxdepth<-c(2,40)
    #  minsplit<-c(2,40)
      
      lower = c(cp[1])
      upper = c(cp[2])
      
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
      dt_cp[i]<-DE_solutions[1]
      
      
      # Grid of optimal hyperparameter values
      DE_XGBoost_classifier_grid <- expand.grid(
        cp = DE_solutions[1]
      )
      
      T0 <- Sys.time()
      cluster <- makeCluster(detectCores() - 1) # number of cores, convention to leave 1 core for OS
      registerDoParallel(cluster) # register the parallel processing
      
      set.seed(1)
      # Train model with optimal values
      DE_XGBoost_classifier_model <- caret::train(Traindata, Trainclass,  
                                                  method = "rpart",
    #                                             trControl = train_control,
                                                  tuneGrid = DE_XGBoost_classifier_grid
      )

      y_pred <- predict(DE_XGBoost_classifier_model, Testdata)
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
    s_result<-paste("E:/gln/C/mypackage/result_new/RQ2/R/tune/DE/rpart/issueclosetime/",days[j],'/',files[k],".csv",sep='')
    write.table(df,s_result,row.names=FALSE,col.names=TRUE,sep=",")
    va_para<-cbind(dt_cp)
    names_para <- c("cp")
    df_para<- data.frame()
    df_para[[names_para[1]]] <- as.numeric()
    df_para <- unlist(rbind(df_para,va_para))
    s_para=paste("E:/gln/C/mypackage/result_new/RQ2/R/tune/DE/param/rpart/issueclosetime/",days[j],'/',files[k],".csv",sep='')
    write.table(df_para, s_para,row.names=FALSE,col.names=TRUE,sep=",")
  }
}
 