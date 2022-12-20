import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from sklearn import metrics
import os
from sklearn.svm import LinearSVC
from hyperopt import fmin, tpe, hp
from hyperopt import Trials
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
   # file = ['DataClass','FeatureEnvy','GodClass','LongMethod']
    #file = ['eclipse', 'groovy','hbase','hive','ivy','jruby','log4j','lucene','poi','prop1','prop2','prop3','prop4','prop5','redaktor','synapse','tomcat','velocity','wicket','xalan','xerces']
   days = ['1days', '7days', '14days', '30days', '90days', '180days', '365days']
   file = ['camel', 'cloudstack', 'cocoon', 'deeplearning', 'hadoop', 'hive', 'node', 'ofbiz', 'qpid']
   for i in range(7):
       for j in range(9):
           s = 'C:/gln/mypackage/dataset/dataset/data_new/issue_close_time/' + days[i] + '/' + file[j] + '/train/'
           s1 = 'C:/gln/mypackage/dataset/dataset/data_new/issue_close_time/' + days[i] + '/' + file[j] + '/test/'
           g = os.walk(s)
           trials = Trials()
           auc_SVM = []
           mcc_SVM = []
           recall_SVM = []
           f1_SVM = []
           brier_SVM = []
           balance_SVM = []
           prec_SVM = []
           Gmean1_SVM = []
           C = []
           Kern = []
           Gamm = []
           for path, dir_list, file_list in g:
               print(file_list)
               for file_name in file_list:
                   s_train = os.path.join(path, file_name)
                   train = pd.read_csv(s_train)
                   s_test = os.path.join(s1, file_name)
                   test = pd.read_csv(s_test)

                   x_train = train.iloc[:, :train.shape[1] - 1].values
                   y_train = train.iloc[:, train.shape[1] - 1].values

                   x_test = test.iloc[:, :test.shape[1] - 1].values
                   y_test = test.iloc[:, test.shape[1] - 1].values


                   def hyperopt_model_score_SVM(params):
                       clf = LinearSVC(**params, max_iter=500)
                       return cross_val_score(clf, x_train, y_train, scoring="roc_auc").mean()


                   space_SVM = {
                       'C': hp.uniform('C', 0, 10),
                       # 'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'poly', 'rbf']),
                       # 'gamma': hp.uniform('gamma', 0, 5),
                   }


                   def fn_SVM(params):
                       acc = hyperopt_model_score_SVM(params)
                       return -acc


                   trials = Trials()

                   best = fmin(
                       fn=fn_SVM, space=space_SVM, algo=tpe.suggest, max_evals=10, trials=trials)
                   print("Best: {}".format(best))
                   # if (best['kernel'] == 0):
                   #     kernel = 'linear'
                   # else:
                   #     if (best['kernel'] == 1):
                   #         kernel = 'sigmoid'
                   #     else:
                   #         if (best['kernel'] == 2):
                   #             kernel = 'poly'
                   #         else:
                   #             kernel = 'rbf'

                   C.append(best['C'])
                   # Kern.append(kernel)
                   # Gamm.append(best['gamma'])

                   svm = LinearSVC(C=best['C'])
                   svm.fit(x_train, y_train)
                   svm.predict(x_test)
                   y_pred_svm = svm.predict(x_test)
                   y_prob_svm = svm.decision_function(x_test)
                   auc_SVM.append(metrics.roc_auc_score(y_test, y_prob_svm))
                   mcc_SVM.append(metrics.matthews_corrcoef(y_test, y_pred_svm))
                   recall_SVM.append(metrics.recall_score(y_test, y_pred_svm))
                   f1_SVM.append(metrics.f1_score(y_test, y_pred_svm))
                   brier_SVM.append(metrics.brier_score_loss(y_test, y_pred_svm))
                   balance_SVM.append(metrics.balanced_accuracy_score(y_test, y_pred_svm))
                   prec_SVM.append(metrics.precision_score(y_test, y_pred_svm))
                   Gmean1_SVM.append(geometric_mean_score(y_test, y_pred_svm))

           data_SVM = pd.DataFrame(
               np.column_stack((auc_SVM, mcc_SVM, recall_SVM, f1_SVM, brier_SVM, balance_SVM, prec_SVM, Gmean1_SVM)))
           data_SVM.columns = ['AUC', 'MCC', 'Recall', 'F1', 'Brier', 'Balance', 'Precision', 'G-Mean1']
           s3 = "C:/gln/mypackage/result/data_new_performance_py/T_performance/nb/LinearSVM/issue_close_time/"+days[i]+"/" + file[j] + ".csv"
           data_SVM.to_csv(s3, index=False)
           best_param = pd.DataFrame(np.column_stack((C)))
           # best_param.columns = ['C']
           s4 = "C:/gln/mypackage/result/data_new_performance_py/parameter/nb/LinearSVM/issue_close_time/"+days[i]+"/" + file[j] + ".csv"
           best_param.to_csv(s4, index=False)





