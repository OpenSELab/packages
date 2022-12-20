import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from sklearn import metrics
import os
from sklearn.neighbors import KNeighborsClassifier
from hyperopt import fmin, tpe, hp
from hyperopt import Trials
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")
import math


if __name__ == '__main__':
    s = 'C:/gln/mypackage/train_test/train/camel/'
    s1 = 'C:/gln/mypackage/train_test/test/camel/'
    g = os.walk(s)
    trials = Trials()
    auc_KNN = []
    mcc_KNN = []
    recall_KNN = []
    f1_KNN= []
    brier_KNN= []
    balance_KNN = []
    prec_KNN = []
    Gmean1_KNN = []
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


            def hyperopt_model_score_KNN(params):
                clf = KNeighborsClassifier(**params)
                return cross_val_score(clf, x_train, y_train, scoring="roc_auc").mean()


            space_KNN = {
                'n_neighbors': hp.choice('max_features', [1, 30])
            }


            def fn_KNN(params):
                acc = hyperopt_model_score_KNN(params)
                return -acc


            trials = Trials()

            best = fmin(
                fn=fn_KNN, space=space_KNN, algo=tpe.suggest, max_evals=100, trials=trials)
            print("Best: {}".format(best))
            if(best['n_neighbors']==0):
                knn = KNeighborsClassifier(n_neighbors=(1+best['n_neighbors']))
            else:
                knn = KNeighborsClassifier(n_neighbors=best['n_neighbors'])

            knn.fit(x_train, y_train)
            knn.predict(x_test)
            y_pred_knn = knn.predict(x_test)
            y_prob_knn = knn.predict_proba(x_test)
            auc_KNN.append(metrics.roc_auc_score(y_test, y_prob_knn[:, 1]))
            mcc_KNN.append(metrics.matthews_corrcoef(y_test, y_pred_knn))
            recall_KNN.append(metrics.recall_score(y_test, y_pred_knn))
            f1_KNN.append(metrics.f1_score(y_test, y_pred_knn))
            brier_KNN.append(metrics.brier_score_loss(y_test, y_pred_knn))
            balance_KNN.append(metrics.balanced_accuracy_score(y_test, y_pred_knn))
            prec_KNN.append(metrics.precision_score(y_test, y_pred_knn))
            Gmean1_KNN.append(geometric_mean_score(y_test, y_pred_knn))

    data_KNN = pd.DataFrame(np.column_stack((auc_KNN, mcc_KNN, recall_KNN, f1_KNN, brier_KNN, balance_KNN, prec_KNN, Gmean1_KNN)))
    data_KNN.columns = ['AUC', 'MCC', 'Recall', 'F1', 'Brier', 'Balance', 'Precision', 'G-Mean1']
    data_KNN.to_csv("C:/gln/mypackage/result/performance/tune/KNN/camel.csv", index=False)


