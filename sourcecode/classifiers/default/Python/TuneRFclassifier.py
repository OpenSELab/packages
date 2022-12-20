import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from sklearn import metrics
import os
from sklearn.ensemble import RandomForestClassifier
from hyperopt import fmin, tpe, hp
from hyperopt import Trials

if __name__ == '__main__':
    # file = ['groovy', 'hbase', 'hive', 'ivy',
    #         'jruby', 'log4j', 'lucene', 'poi', 'prop1', 'prop2', 'prop3',
    #         'prop4', 'prop5', 'redaktor', 'synapse', 'tomcat', 'velocity', 'wicket', 'xalan', 'xerces','eclipse']
    #file=['camel','cloudstack','cocoon','deeplearning','hadoop','hive','node','ofbiz','qpid']
   # days=['1days','7days','14days','30days','90days','180days','365days']
    file = ['camel', 'derby', 'eclipse', 'groovy', 'hbase', 'hive', 'ivy', 'jruby', 'log4j', 'lucene', 'poi', 'prop1',
            'prop2', 'prop3', 'prop4', 'prop5', 'redaktor', 'synapse', 'tomcat', 'velocity', 'wicket', 'xalan',
            'xerces']
    #for i in range(7):
    for j in range(23):
            s = 'E:/gln/C/mypackage/dataset/dataset/dataset/' + file[j] + '/s_train/'
            s1 = 'E:/gln/C/mypackage/dataset/dataset/dataset/' + file[j] + '/s_test/'
            g = os.walk(s)
            trials = Trials()
            auc_RF = []
            mcc_RF = []
            recall_RF = []
            f1_RF = []
            brier_RF = []
            balance_RF = []
            prec_RF = []
            Gmean1_RF = []


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


                    rs = RandomForestClassifier()
                    rs.fit(x_train, y_train)

                    y_pred_rf = rs.predict(x_test)
                    y_prob_rf = rs.predict_proba(x_test)
                    auc_RF.append(metrics.roc_auc_score(y_test, y_prob_rf[:, 1]))
                    mcc_RF.append(metrics.matthews_corrcoef(y_test, y_pred_rf))
                    recall_RF.append(metrics.recall_score(y_test, y_pred_rf))
                    f1_RF.append(metrics.f1_score(y_test, y_pred_rf))
                    brier_RF.append(metrics.brier_score_loss(y_test, y_pred_rf))
                    balance_RF.append(metrics.balanced_accuracy_score(y_test, y_pred_rf))
                    prec_RF.append(metrics.precision_score(y_test, y_pred_rf))
                    Gmean1_RF.append(geometric_mean_score(y_test, y_pred_rf))

            data_RF = pd.DataFrame(
                np.column_stack((auc_RF, mcc_RF, recall_RF, f1_RF, brier_RF, balance_RF, prec_RF, Gmean1_RF)))
            data_RF.columns = ['AUC', 'MCC', 'Recall', 'F1', 'Brier', 'Balance', 'Precision', 'G-Mean1']
            s3 = "E:/gln/C/mypackage/result_new/RQ1/defaultsetting/RF/SDP/RandomForestClassifier/" +file[j] + ".csv"
            data_RF.to_csv(s3, index=False)