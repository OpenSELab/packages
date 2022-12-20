import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from sklearn import metrics
import os
from sklearn.tree import DecisionTreeClassifier
from hyperopt import Trials
from scipy.optimize import differential_evolution


if __name__ == '__main__':
    #file = ['camel', 'derby', 'eclipse', 'groovy', 'hbase', 'hive', 'ivy', 'jruby', 'log4j', 'lucene', 'poi', 'prop1', 'prop2', 'prop3','prop4', 'prop5', 'redaktor', 'synapse', 'tomcat', 'velocity', 'wicket', 'xalan', 'xerces']
    # file=['DataClass','FeatureEnvy','GodClass','LongMethod']
   file = ['camel', 'cloudstack', 'cocoon', 'deeplearning', 'hadoop', 'hive', 'node', 'ofbiz', 'qpid']
   days = ['1days', '7days', '14days', '30days', '90days', '180days', '365days']
   for i in range(7):
      for j in range(9):
           s = 'E:/gln/C/mypackage/dataset/dataset/issueclosetime/'+days[i]+'/'+ file[j] + '/train/'
           s1 = 'E:/gln/C/mypackage/dataset/dataset/issueclosetime/' +days[i]+'/'+ file[j] + '/test/'
           g = os.walk(s)
           trials = Trials()
           auc_DT = []
           mcc_DT = []
           recall_DT = []
           f1_DT = []
           brier_DT = []
           balance_DT = []
           prec_DT = []
           Gmean1_DT = []

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


                   rs = DecisionTreeClassifier(max_depth=30,min_samples_split=20)
                   rs.fit(x_train, y_train)

                   y_pred_dt = rs.predict(x_test)
                   y_prob_dt = rs.predict_proba(x_test)
                   auc_DT.append(metrics.roc_auc_score(y_test, y_prob_dt[:, 1]))
                   mcc_DT.append(metrics.matthews_corrcoef(y_test, y_pred_dt))
                   recall_DT.append(metrics.recall_score(y_test, y_pred_dt))
                   f1_DT.append(metrics.f1_score(y_test, y_pred_dt))
                   brier_DT.append(metrics.brier_score_loss(y_test, y_pred_dt))
                   balance_DT.append(metrics.balanced_accuracy_score(y_test, y_pred_dt))
                   prec_DT.append(metrics.precision_score(y_test, y_pred_dt))
                   Gmean1_DT.append(geometric_mean_score(y_test, y_pred_dt))

           data_DT = pd.DataFrame(
               np.column_stack((auc_DT, mcc_DT, recall_DT, f1_DT, brier_DT, balance_DT, prec_DT, Gmean1_DT)))
           data_DT.columns = ['AUC', 'MCC', 'Recall', 'F1', 'Brier', 'Balance', 'Precision', 'G-Mean1']
           s3 = "E:/gln/C/mypackage/result_new/RQ1/samesetting/DT/issueclosetime/"+days[i]+"/decisiontree/"+ file[j] + ".csv"
           data_DT.to_csv(s3, index=False)




