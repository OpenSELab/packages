import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from sklearn import metrics
import os
from sklearn.ensemble import RandomForestClassifier
from hyperopt import fmin, tpe, hp
from hyperopt import Trials
from sklearn.model_selection import cross_val_score


if __name__ == '__main__':
    s = 'C:/gln/mypackage/train_test/train/camel/'
    s1 = 'C:/gln/mypackage/train_test/test/camel/'
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


            def hyperopt_model_score_RF(params):
                clf = RandomForestClassifier(**params)
                return cross_val_score(clf, x_train, y_train, scoring="roc_auc").mean()


            space_RF = {
                'max_depth': 1+hp.choice('max_depth', range(1,20)),
                'max_features':1+hp.randint("max_features",15),
                'n_estimators': hp.choice('n_estimators',range(100,500)),
                'criterion': hp.choice('criterion', ["gini", "entropy"])
            }

            def fn_RF(params):
                acc = hyperopt_model_score_RF(params)
                return -acc


            trials = Trials()

            best = fmin(
                fn=fn_RF, space=space_RF, algo=tpe.suggest, max_evals=100, trials=trials)
            print("Best: {}".format(best))
            if (best['criterion'] == 0):
                criterion = "gini"
            else:
                criterion = "entropy"


            rf = RandomForestClassifier(criterion=criterion, max_depth=best['max_depth'],
                                            max_features=best['max_features'], n_estimators=best['n_estimators'])

            rf.fit(x_train, y_train)
            rf.predict(x_test)
            y_pred_rf = rf.predict(x_test)
            y_prob_rf = rf.predict_proba(x_test)
            auc_RF.append(metrics.roc_auc_score(y_test, y_prob_rf[:, 1]))
            mcc_RF.append(metrics.matthews_corrcoef(y_test, y_pred_rf))
            recall_RF.append(metrics.recall_score(y_test, y_pred_rf))
            f1_RF.append(metrics.f1_score(y_test, y_pred_rf))
            brier_RF.append(metrics.brier_score_loss(y_test, y_pred_rf))
            balance_RF.append(metrics.balanced_accuracy_score(y_test, y_pred_rf))
            prec_RF.append(metrics.precision_score(y_test, y_pred_rf))
            Gmean1_RF.append(geometric_mean_score(y_test, y_pred_rf))

    data_RF = pd.DataFrame(np.column_stack((auc_RF, mcc_RF, recall_RF, f1_RF, brier_RF, balance_RF, prec_RF, Gmean1_RF)))
    data_RF.columns = ['AUC', 'MCC', 'Recall', 'F1', 'Brier', 'Balance', 'Precision', 'G-Mean1']
    data_RF.to_csv("C:/gln/mypackage/result/performance/tune/RF/camel.csv", index=False)


