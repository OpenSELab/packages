import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from sklearn import metrics
import os
from sklearn.svm import LinearSVC
from hyperopt import Trials
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    #file =['DataClass','FeatureEnvy','GodClass','LongMethod']
   # file = ['camel', 'cloudstack', 'cocoon', 'deeplearning', 'hadoop', 'hive', 'node', 'ofbiz', 'qpid']
    #days = ['1days', '7days', '14days', '30days', '90days', '180days', '365days']
    file = ['camel', 'derby', 'eclipse', 'groovy', 'hbase', 'hive', 'ivy', 'jruby', 'log4j', 'lucene', 'poi', 'prop1',
            'prop2', 'prop3', 'prop4', 'prop5', 'redaktor', 'synapse', 'tomcat', 'velocity', 'wicket', 'xalan',
            'xerces']
    #for i in range(7):
    for j in range(23):
            s = 'E:/gln/C/mypackage/dataset/dataset/dataset/' + file[j] + '/s_train/'
            s1 = 'E:/gln/C/mypackage/dataset/dataset/dataset/' + file[j] + '/s_test/'
            g = os.walk(s)
            trials = Trials()
            auc_SVM = []
            mcc_SVM = []
            recall_SVM = []
            f1_SVM = []
            brier_SVM = []
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


                    rs = LinearSVC()
                    rs.fit(x_train, y_train)


                    y_pred_svm = rs.predict(x_test)
                    y_prob_svm = rs.decision_function(x_test)
                    auc_SVM.append(metrics.roc_auc_score(y_test, y_prob_svm))
                    mcc_SVM.append(metrics.matthews_corrcoef(y_test, y_pred_svm))
                    recall_SVM.append(metrics.recall_score(y_test, y_pred_svm))
                    f1_SVM.append(metrics.f1_score(y_test, y_pred_svm))
                    brier_SVM.append(metrics.brier_score_loss(y_test, y_pred_svm))
                    balance_SVM.append(metrics.balanced_accuracy_score(y_test, y_pred_svm))
                    prec_SVM.append(metrics.precision_score(y_test, y_pred_svm))
                    Gmean1_SVM.append(geometric_mean_score(y_test, y_pred_svm))

            data_SVM = pd.DataFrame(
                np.column_stack((auc_SVM, mcc_SVM, recall_SVM, f1_SVM, brier_SVM, balance_SVM, prec_SVM, Gmean1_SVM)))
            data_SVM.columns = ['AUC', 'MCC', 'Recall', 'F1', 'Brier', 'Balance', 'Precision', 'G-Mean1']
            s3 = "E:/gln/C/mypackage/result_new/RQ1/defaultsetting/SVM/SDP/"+'LinearSVC/' +file[j] + ".csv"
            data_SVM.to_csv(s3, index=False)




