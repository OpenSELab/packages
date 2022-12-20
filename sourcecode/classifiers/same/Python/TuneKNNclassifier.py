import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from sklearn import metrics
import os
from sklearn.neighbors import KNeighborsClassifier
from hyperopt import Trials
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
   # file = ['eclipse',  'groovy', 'hbase', 'hive', 'ivy',
    #        'jruby', 'log4j', 'lucene', 'poi', 'prop1', 'prop2', 'prop3',
      #      'prop4', 'prop5', 'redaktor', 'synapse', 'tomcat', 'velocity', 'wicket', 'xalan', 'xerces']
   #file = ['camel', 'cloudstack', 'cocoon', 'deeplearning', 'hadoop', 'hive', 'node', 'ofbiz', 'qpid']
   #days = ['1days', '7days', '14days', '30days', '90days', '180days', '365days']
   file = ['camel', 'derby', 'eclipse', 'groovy', 'hbase', 'hive', 'ivy', 'jruby', 'log4j', 'lucene', 'poi', 'prop1',
           'prop2', 'prop3', 'prop4', 'prop5', 'redaktor', 'synapse', 'tomcat', 'velocity', 'wicket', 'xalan', 'xerces']
  # for i in range(7):
   for j in range(23):
           s = 'E:/gln/C/mypackage/dataset/dataset/dataset/' + file[j] + '/s_train/'
           s1 = 'E:/gln/C/mypackage/dataset/dataset/dataset/' + file[j] + '/s_test/'
           g = os.walk(s)
           trials = Trials()
           auc_KNN = []
           mcc_KNN = []
           recall_KNN = []
           f1_KNN = []
           brier_KNN = []
           balance_KNN = []
           prec_KNN = []
           Gmean1_KNN = []

           for path, dir_list, file_list in g:
               print(file_list)
               for file_name in file_list:
                #   print(file_name)
                   s_train = os.path.join(path, file_name)
                   train = pd.read_csv(s_train)
                   s_test = os.path.join(s1, file_name)
                   test = pd.read_csv(s_test)

                   x_train = train.iloc[:, :train.shape[1] - 1].values
                   y_train = train.iloc[:, train.shape[1] - 1].values

                   x_test = test.iloc[:, :test.shape[1] - 1].values
                   y_test = test.iloc[:, test.shape[1] - 1].values


                   rs = KNeighborsClassifier(n_neighbors=5)
                   rs.fit(x_train, y_train)

                   y_pred_knn = rs.predict(x_test)
                   y_prob_knn = rs.predict_proba(x_test)
                   auc_KNN.append(metrics.roc_auc_score(y_test, y_prob_knn[:, 1]))
                   mcc_KNN.append(metrics.matthews_corrcoef(y_test, y_pred_knn))
                   recall_KNN.append(metrics.recall_score(y_test, y_pred_knn))
                   f1_KNN.append(metrics.f1_score(y_test, y_pred_knn))
                   brier_KNN.append(metrics.brier_score_loss(y_test, y_pred_knn))
                   balance_KNN.append(metrics.balanced_accuracy_score(y_test, y_pred_knn))
                   prec_KNN.append(metrics.precision_score(y_test, y_pred_knn))
                   Gmean1_KNN.append(geometric_mean_score(y_test, y_pred_knn))

           data_KNN = pd.DataFrame(
               np.column_stack((auc_KNN, mcc_KNN, recall_KNN, f1_KNN, brier_KNN, balance_KNN, prec_KNN, Gmean1_KNN)))
           data_KNN.columns = ['AUC', 'MCC', 'Recall', 'F1', 'Brier', 'Balance', 'Precision', 'G-Mean1']
           s3 = "E:/gln/C/mypackage/result_new/RQ1/defaultsetting/KNN/issueclosetime/"+days[i]+'/KNN/'+ file[i] + ".csv"
           data_KNN.to_csv(s3, index=False)




