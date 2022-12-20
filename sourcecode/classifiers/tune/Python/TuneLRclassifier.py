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
    s = 'C:/gln/mypackage/train_test/train/camel/'
    s1 = 'C:/gln/mypackage/train_test/test/camel/'
    g = os.walk(s)
    trials = Trials()
    auc_LR = []
    mcc_LR = []
    recall_LR = []
    f1_LR= []
    brier_LR= []
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


            def hyperopt_model_score_LR(params):
                clf = LogisticRegression(**params)
                return cross_val_score(clf, x_train, y_train, scoring="roc_auc").mean()


            space_LR = {
                'tol': hp.choice('tol', [1e-5,1e-4,1e-3]),
                'C': hp.choice('C', [1.0,0.1,0.001]),
                'penalty': hp.choice('penalty', ["l1", "l2"]),
            }


            def fn_LR(params):
                acc = hyperopt_model_score_LR(params)
                return -acc


            trials = Trials()

            best = fmin(
                fn=fn_LR, space=space_LR, algo=tpe.suggest, max_evals=1000, trials=trials)
            print("Best: {}".format(best))
            if (best['penalty'] == 0):
                penalty = "l1"
            else:
                penalty = "l2"

            if (best['tol'] == 0):
                tol = 1e-5
            else:
                if(best['tol'] == 1):
                    tol = 1e-4
                else:
                    tol = 1e-3

            if (best['C'] == 0):
                C = 1.0
            else:
                if(best['C'] == 1):
                    C = 0.1
                else:
                    C = 0.01
            print(penalty,tol,C)

            lr =LogisticRegression(penalty=penalty,tol=tol,C=C)
            lr.fit(x_train, y_train)
            lr.predict(x_test)
            y_pred_lr = lr.predict(x_test)
            y_prob_lr = lr.predict_proba(x_test)
            auc_LR.append(metrics.roc_auc_score(y_test, y_prob_lr[:, 1]))
            mcc_LR.append(metrics.matthews_corrcoef(y_test, y_pred_lr))
            recall_LR.append(metrics.recall_score(y_test, y_pred_lr))
            f1_LR.append(metrics.f1_score(y_test, y_pred_lr))
            brier_LR.append(metrics.brier_score_loss(y_test, y_pred_lr))
            balance_LR.append(metrics.balanced_accuracy_score(y_test, y_pred_lr))
            prec_LR.append(metrics.precision_score(y_test, y_pred_lr))
            Gmean1_LR.append(geometric_mean_score(y_test, y_pred_lr))

    data_LR = pd.DataFrame(np.column_stack((auc_LR, mcc_LR, recall_LR, f1_LR, brier_LR, balance_LR, prec_LR, Gmean1_LR)))
    data_LR.columns = ['AUC', 'MCC', 'Recall', 'F1', 'Brier', 'Balance', 'Precision', 'G-Mean1']
    data_LR.to_csv("C:/gln/mypackage/result/performance/tune/LR/camel.csv", index=False)


