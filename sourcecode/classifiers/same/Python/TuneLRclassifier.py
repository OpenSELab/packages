import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from sklearn import metrics
import os
from sklearn.linear_model import LogisticRegression
from hyperopt import fmin, tpe, hp
from hyperopt import Trials
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    #file = ['DataClass','FeatureEnvy','GodClass','LongMethod']
    file = ['camel', 'cloudstack', 'cocoon', 'deeplearning', 'hadoop', 'hive', 'node', 'ofbiz', 'qpid']
    days = ['1days', '7days', '14days', '30days', '90days', '180days', '365days']
    for i in range(7):
        for j in range(9):
            s = 'E:/gln/C/mypackage/dataset/dataset/issueclosetime/' + days[i] + '/' + file[j] + '/train/'
            s1 = 'E:/gln/C/mypackage/dataset/dataset/issueclosetime/' + days[i] + '/' + file[j] + '/test/'
            g = os.walk(s)
            trials = Trials()
            auc_LR = []
            mcc_LR = []
            recall_LR = []
            f1_LR = []
            brier_LR = []
            balance_LR = []
            prec_LR = []
            Gmean1_LR = []

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


                    rs = LogisticRegression()
                    rs.fit(x_train, y_train)



                    y_pred_lr = rs.predict(x_test)
                    y_prob_lr = rs.predict_proba(x_test)
                    auc_LR.append(metrics.roc_auc_score(y_test, y_prob_lr[:, 1]))
                    mcc_LR.append(metrics.matthews_corrcoef(y_test, y_pred_lr))
                    recall_LR.append(metrics.recall_score(y_test, y_pred_lr))
                    f1_LR.append(metrics.f1_score(y_test, y_pred_lr))
                    brier_LR.append(metrics.brier_score_loss(y_test, y_pred_lr))
                    balance_LR.append(metrics.balanced_accuracy_score(y_test, y_pred_lr))
                    prec_LR.append(metrics.precision_score(y_test, y_pred_lr))
                    Gmean1_LR.append(geometric_mean_score(y_test, y_pred_lr))

            data_LR = pd.DataFrame(
                np.column_stack((auc_LR, mcc_LR, recall_LR, f1_LR, brier_LR, balance_LR, prec_LR, Gmean1_LR)))
            data_LR.columns = ['AUC', 'MCC', 'Recall', 'F1', 'Brier', 'Balance', 'Precision', 'G-Mean1']
            s3 = "E:/gln/C/mypackage/result_new/RQ1/defaultsetting/LR/issueclosetime/"+days[i]+'/LR/' + file[j] + ".csv"
            data_LR.to_csv(s3, index=False)



