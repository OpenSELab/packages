import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from sklearn import metrics
import os
from hyperopt import Trials
from xgboost import XGBClassifier
import warnings

if __name__ == '__main__':
    # file = ['camel','derby','eclipse', 'groovy', 'hbase', 'hive', 'ivy',
    #         'jruby', 'log4j', 'lucene', 'poi', 'prop1', 'prop2', 'prop3',
    #         'prop4', 'prop5', 'redaktor', 'synapse', 'tomcat', 'velocity', 'wicket', 'xalan', 'xerces']
    #file=['DataClass','FeatureEnvy','GodClass','LongMethod']
    file = ['camel', 'cloudstack', 'cocoon', 'deeplearning', 'hadoop', 'hive', 'node', 'ofbiz', 'qpid']
    days = ['1days','7days','14days', '30days', '90days', '180days', '365days']
    for i in range(7):
        for j in range(9):
            s = 'E:/gln/C/mypackage/dataset/dataset/issueclosetime/' + days[i] + '/' + file[j] + '/train/'
            s1 = 'E:/gln/C/mypackage/dataset/dataset/issueclosetime/' + days[i] + '/' + file[j] + '/test/'
            g = os.walk(s)
            trials = Trials()
            auc_GBM = []
            mcc_GBM = []
            recall_GBM = []
            f1_GBM = []
            brier_GBM = []
            balance_GBM = []
            prec_GBM = []
            Gmean1_GBM = []


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

                    rs = XGBClassifier(nrounds=100,eta=0.3,max_depth=6)
                    rs.fit(x_train, y_train)

                    y_pred_gbm = rs.predict(x_test)
                    y_prob_gbm = rs.predict_proba(x_test)
                    auc_GBM.append(metrics.roc_auc_score(y_test, y_prob_gbm[:, 1]))
                    mcc_GBM.append(metrics.matthews_corrcoef(y_test, y_pred_gbm))
                    recall_GBM.append(metrics.recall_score(y_test, y_pred_gbm))
                    f1_GBM.append(metrics.f1_score(y_test, y_pred_gbm))
                    brier_GBM.append(metrics.brier_score_loss(y_test, y_pred_gbm))
                    balance_GBM.append(metrics.balanced_accuracy_score(y_test, y_pred_gbm))
                    prec_GBM.append(metrics.precision_score(y_test, y_pred_gbm))
                    Gmean1_GBM.append(geometric_mean_score(y_test, y_pred_gbm))

            data_GBM = pd.DataFrame(
                np.column_stack((auc_GBM, mcc_GBM, recall_GBM, f1_GBM, brier_GBM, balance_GBM, prec_GBM, Gmean1_GBM)))
            data_GBM.columns = ['AUC', 'MCC', 'Recall', 'F1', 'Brier', 'Balance', 'Precision', 'G-Mean1']
            s3 = "E:/gln/C/mypackage/result_new/RQ1/defaultsetting/xgboost/issueclosetime/"+days[i]+'/xgboost/' + file[j] + ".csv"
            data_GBM.to_csv(s3, index=False)

