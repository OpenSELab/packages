import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from sklearn import metrics
import os
from sklearn.svm import SVC
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
    auc_SVM = []
    mcc_SVM = []
    recall_SVM = []
    f1_SVM= []
    brier_SVM= []
    balance_SVM = []
    prec_SVM = []
    Gmean1_SVM = []
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


            def hyperopt_model_score_SVM(params):
                clf = SVC(**params)
                return cross_val_score(clf, x_train, y_train, scoring="roc_auc").mean()


            space_SVM = {
                'C': hp.uniform('C', 0, 10),
                'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'poly', 'rbf']),
                'gamma': hp.uniform('gamma', 0, 10),
            }

            def fn_SVM(params):
                acc = hyperopt_model_score_SVM(params)
                return -acc

            trials = Trials()

            best = fmin(
                fn=fn_SVM, space=space_SVM, algo=tpe.suggest, max_evals=1000, trials=trials)
            print("Best: {}".format(best))
            if (best['kernel'] == 0):
                kernel = 'linear'
            else:
                if(best['kernel']==1):
                    kernel = 'sigmoid'
                else:
                    if(best['kernel']==2):
                        kernel = 'poly'
                    else:
                        kernel = 'rbf'

            svm =SVC(C=best['C'],kernel=kernel,gamma=best['gamma'])
            svm.fit(x_train, y_train)
            svm.predict(x_test)
            y_pred_svm = svm.predict(x_test)
            y_prob_svm = svm.predict_proba(x_test)
            auc_SVM.append(metrics.roc_auc_score(y_test, y_prob_svm[:, 1]))
            mcc_SVM.append(metrics.matthews_corrcoef(y_test, y_pred_svm))
            recall_SVM.append(metrics.recall_score(y_test, y_pred_svm))
            f1_SVM.append(metrics.f1_score(y_test, y_pred_svm))
            brier_SVM.append(metrics.brier_score_loss(y_test, y_pred_svm))
            balance_SVM.append(metrics.balanced_accuracy_score(y_test, y_pred_svm))
            prec_SVM.append(metrics.precision_score(y_test, y_pred_svm))
            Gmean1_SVM.append(geometric_mean_score(y_test, y_pred_svm))

    data_SVM = pd.DataFrame(np.column_stack((auc_SVM, mcc_SVM, recall_SVM, f1_SVM, brier_SVM, balance_SVM, prec_SVM, Gmean1_SVM)))
    data_SVM.columns = ['AUC', 'MCC', 'Recall', 'F1', 'Brier', 'Balance', 'Precision', 'G-Mean1']
    data_SVM.to_csv("C:/gln/mypackage/result/performance/tune/SVM/camel.csv", index=False)


