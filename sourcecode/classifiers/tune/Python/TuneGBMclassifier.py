import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from sklearn import metrics
import os
from sklearn.ensemble import GradientBoostingClassifier
from hyperopt import fmin, tpe, hp
from hyperopt import Trials
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier


if __name__ == '__main__':
    s = 'C:/gln/mypackage/train_test/train/camel/'
    s1 = 'C:/gln/mypackage/train_test/test/camel/'
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


            def hyperopt_model_score_GBM(params):
                clf = XGBClassifier(**params)
                return cross_val_score(clf, x_train, y_train, scoring="roc_auc").mean()


            space_GBM = {
                    'max_depth': hp.choice('max_depth',range(3,10)),
                    'n_estimators': hp.choice('n_estimators', range(100,500)),

            }

            def fn_GBM(params):
                acc = hyperopt_model_score_GBM(params)
                return -acc


            trials = Trials()

            best = fmin(
                fn=fn_GBM, space=space_GBM, algo=tpe.suggest, max_evals=1000, trials=trials)
            print("Best: {}".format(best))

            gbm = GradientBoostingClassifier(n_estimators=best['n_estimators'],max_depth=best['max_depth'])
            gbm.fit(x_train, y_train)
            gbm.predict(x_test)
            y_pred_gbm = gbm.predict(x_test)
            y_prob_gbm = gbm.predict_proba(x_test)
            auc_GBM.append(metrics.roc_auc_score(y_test, y_prob_gbm[:, 1]))
            mcc_GBM.append(metrics.matthews_corrcoef(y_test, y_pred_gbm))
            recall_GBM.append(metrics.recall_score(y_test, y_pred_gbm))
            f1_GBM.append(metrics.f1_score(y_test, y_pred_gbm))
            brier_GBM.append(metrics.brier_score_loss(y_test, y_pred_gbm))
            balance_GBM.append(metrics.balanced_accuracy_score(y_test, y_pred_gbm))
            prec_GBM.append(metrics.precision_score(y_test, y_pred_gbm))
            Gmean1_GBM.append(geometric_mean_score(y_test, y_pred_gbm))

    data_GBM = pd.DataFrame(np.column_stack((auc_GBM, mcc_GBM, recall_GBM, f1_GBM, brier_GBM, balance_GBM, prec_GBM, Gmean1_GBM)))
    data_GBM.columns = ['AUC', 'MCC', 'Recall', 'F1', 'Brier', 'Balance', 'Precision', 'G-Mean1']
    data_GBM.to_csv("C:/gln/mypackage/result/performance/tune/GBM/camel.csv", index=False)


