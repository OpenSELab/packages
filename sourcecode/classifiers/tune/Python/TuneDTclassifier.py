import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from sklearn import metrics
import os
from sklearn.tree import DecisionTreeClassifier
from hyperopt import fmin, tpe, hp
from hyperopt import Trials
from sklearn.model_selection import cross_val_score


if __name__ == '__main__':
    s = 'C:/gln/mypackage/train_test/train/camel/'
    s1 = 'C:/gln/mypackage/train_test/test/camel/'
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
    max_depth=[]
    max_features=[]
    crites=[]

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


            def hyperopt_model_score_dtree(params):
                clf = DecisionTreeClassifier(**params)
                return cross_val_score(clf, x_train, y_train, scoring="roc_auc").mean()


            space_dtree = {
                'max_depth': 1 +hp.randint("max_depth",15),
                'max_features': hp.choice('max_features', [3,5,7,10,12]),
                'criterion': hp.choice('criterion', ["gini", "entropy"])
            }


            def fn_dtree(params):
                acc = hyperopt_model_score_dtree(params)
                return -acc


            trials = Trials()

            best = fmin(
                fn=fn_dtree, space=space_dtree, algo=tpe.suggest, max_evals=100, trials=trials)
            print("Best: {}".format(best))
            if (best['criterion'] == 0):
                criterion = "gini"
            else:
                criterion = "entropy"

            if (best['max_features'] == 0):
                feature = 3
            else:
                if (best['max_features'] == 1):
                    feature = 5
                else:
                    if (best['max_features'] == 2):
                        feature = 7
                    else:
                        if (best['max_features'] == 3):
                            feature = 10
                        else:
                            feature = 12


            dt = DecisionTreeClassifier(criterion=criterion, max_depth=best['max_depth'],
                                            max_features=feature)


            max_depth.append(best['max_depth'])
            max_features.append(best['max_features'])
            crites.append(criterion)

            dt.fit(x_train, y_train)
            dt.predict(x_test)
            y_pred_dt = dt.predict(x_test)
            y_prob_dt = dt.predict_proba(x_test)
            auc_DT.append(metrics.roc_auc_score(y_test, y_prob_dt[:, 1]))
            mcc_DT.append(metrics.matthews_corrcoef(y_test, y_pred_dt))
            recall_DT.append(metrics.recall_score(y_test, y_pred_dt))
            f1_DT.append(metrics.f1_score(y_test, y_pred_dt))
            brier_DT.append(metrics.brier_score_loss(y_test, y_pred_dt))
            balance_DT.append(metrics.balanced_accuracy_score(y_test, y_pred_dt))
            prec_DT.append(metrics.precision_score(y_test, y_pred_dt))
            Gmean1_DT.append(geometric_mean_score(y_test, y_pred_dt))

    data_DT = pd.DataFrame(np.column_stack((auc_DT, mcc_DT, recall_DT, f1_DT, brier_DT, balance_DT, prec_DT, Gmean1_DT)))
    data_DT.columns = ['AUC', 'MCC', 'Recall', 'F1', 'Brier', 'Balance', 'Precision', 'G-Mean1']
    data_DT.to_csv("C:/gln/mypackage/result/performance/tune/DT/camel.csv", index=False)
    data_para = pd.DataFrame(
        np.column_stack((max_depth,max_features,crites)))
    data_para.columns = ['max_depth', 'max_features', 'criterion']
    data_para.to_csv("C:/gln/mypackage/result/para/DT/camel.csv", index=False)


