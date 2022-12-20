from eli5.sklearn import PermutationImportance
import numpy as np
import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.metrics import geometric_mean_score
from sklearn import metrics
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


if __name__ == '__main__':
    file = ['camel', 'ceylon', 'derby', 'eclipse', 'elasticsearch', 'groovy', 'hazelcast', 'hbase', 'hive', 'ivy',
            'jruby', 'log4j', 'lucene', 'mcMMO', 'neo4j', 'netty', 'orientdb', 'poi', 'prop1', 'prop2', 'prop3',
            'prop4', 'prop5', 'synapse', 'tomcat', 'velocity', 'wicket', 'xalan', 'xerces']
    for i in range(1):
        s = 'C:/gln/mypackage/train_test/train/' + file[i]
        s1 = 'C:/gln/mypackage/train_test/test/' + file[i]
        g = os.walk(s)
        s_feature="C:/gln/mypackage/autospearmandataset/"+ file[i] + '.csv'
        df_feature=pd.read_csv(s_feature)
        DT_feature_train=[]
        RF_feature_train=[]
        SVM_feature_train=[]
        LR_feature_train=[]
        GBM_feature_train=[]
        KNN_feature_train=[]

        DT_feature_test=[]
        RF_feature_test=[]
        SVM_feature_test=[]
        LR_feature_test=[]
        GBM_feature_test=[]
        KNN_feature_test=[]


        p_feature = []
        s_SM = df_feature.columns.astype(str)

        for k in range(0, df_feature.shape[1] - 2):
            p_feature.append(s_SM[k])
        print(p_feature)

        for path, dir_list, file_list in g:
            print(file_list)
            for file_name in file_list:
                s_train = os.path.join(path, file_name)
                train = pd.read_csv(s_train)
                s_test = os.path.join(s1, file_name)
                test = pd.read_csv(s_test)

                x_train = train[p_feature].values
                y_train = train.iloc[:, train.shape[1] - 1].values

                x_test = test[p_feature].values
                y_test = test.iloc[:, test.shape[1] - 1].values
                print(x_train.shape)
                print(p_feature)

                dt=DecisionTreeClassifier(min_samples_leaf=1,max_depth=30,ccp_alpha=0.01).fit(x_train,y_train)
                r_dt=PermutationImportance(dt, n_iter=5, random_state=1024, cv=5)
                r_dt.fit(x_test, y_test)
                print(r_dt.feature_importances_)
                DT_feature_test.append(r_dt.feature_importances_)

                r_dt_train= PermutationImportance(dt, n_iter=5, random_state=1024, cv=5)
                r_dt_train.fit(x_train, y_train)
                # print(r_dt_train.feature_importances_)
                DT_feature_train.append(r_dt_train.feature_importances_)

                rf = RandomForestClassifier(n_estimators=100).fit(x_train, y_train)
                r_rf=PermutationImportance(rf, n_iter=5, random_state=1024, cv=5)
                r_rf.fit(x_test, y_test)
                RF_feature_test.append(r_rf.feature_importances_)

                r_rf_train = PermutationImportance(rf, n_iter=5, random_state=1024, cv=5)
                r_rf_train.fit(x_train, y_train)
                RF_feature_train.append(r_rf_train.feature_importances_)


                lr = LogisticRegression(tol=1e-8, max_iter=100,solver='liblinear').fit(x_train, y_train)
                r_lr = PermutationImportance(lr, n_iter=5, random_state=1024, cv=5)
                r_lr.fit(x_test, y_test)
                LR_feature_test.append(r_lr.feature_importances_)

                r_lr_train = PermutationImportance(lr, n_iter=5, random_state=1024, cv=5)
                r_lr_train.fit(x_train, y_train)
                LR_feature_train.append(r_lr_train.feature_importances_)



                knn = KNeighborsClassifier(n_neighbors=5).fit(x_train, y_train).fit(x_train, y_train)
                r_knn = PermutationImportance(knn, n_iter=5, random_state=1024, cv=5)
                r_knn.fit(x_test, y_test)
                KNN_feature_test.append(r_knn.feature_importances_)

                r_knn_train = PermutationImportance(knn, n_iter=5, random_state=1024, cv=5)
                r_knn_train.fit(x_train, y_train)
                KNN_feature_train.append(r_knn_train.feature_importances_)

                svc = SVC(kernel='sigmoid', cache_size=200, probability=True).fit(x_train, y_train)
                #svc.fit(x_train, y_train)
                r_svc = PermutationImportance(svc, n_iter=5, random_state=1024, cv=5)
                r_svc.fit(x_test, y_test)
                SVM_feature_test.append(r_svc.feature_importances_)

                r_svc_train = PermutationImportance(svc, n_iter=5, random_state=1024, cv=5)
                r_svc_train.fit(x_train, y_train)
                SVM_feature_train.append(r_svc_train.feature_importances_)

                gbm = XGBClassifier(learning_rate=0.3, n_estimators=100, max_depth=10).fit(x_train, y_train)
                #gbm.fit(x_train, y_train)
                r_gbm = PermutationImportance(gbm, n_iter=5, random_state=1024, cv=5)
                r_gbm.fit(x_test, y_test)
                GBM_feature_test.append(r_gbm.feature_importances_)

                r_gbm_train = PermutationImportance(gbm, n_iter=5, random_state=1024, cv=5)
                r_gbm_train.fit(x_train, y_train)
                GBM_feature_train.append(r_gbm_train.feature_importances_)

        df_RF_test=pd.DataFrame(RF_feature_test)
        df_DT_test=pd.DataFrame(DT_feature_test)
        df_SVM_test=pd.DataFrame(SVM_feature_test)
        df_LR_test=pd.DataFrame(LR_feature_test)
        df_KNN_test=pd.DataFrame(KNN_feature_test)
        df_GBM_test=pd.DataFrame(GBM_feature_test)
        df_RF_test.columns=p_feature
        df_DT_test.columns =p_feature
        df_SVM_test.columns =p_feature
        df_LR_test.columns =p_feature
        df_KNN_test.columns =p_feature
        df_GBM_test.columns =p_feature
        s_RF_test="C:/gln/mypackage/result/featureimportant/python/NT_tune/test/RF/"+ file[i] + '.csv'
        s_DT_test = "C:/gln/mypackage/result/featureimportant/python/NT_tune/test/DT/" + file[i] + '.csv'
        s_SVM_test = "C:/gln/mypackage/result/featureimportant/python/NT_tune/test/SVM/" + file[i] + '.csv'
        s_LR_test = "C:/gln/mypackage/result/featureimportant/python/NT_tune/test/LR/" + file[i] + '.csv'
        s_KNN_test = "C:/gln/mypackage/result/featureimportant/python/NT_tune/test/KNN/" + file[i] + '.csv'
        s_GBM_test = "C:/gln/mypackage/result/featureimportant/python/NT_tune/test/GBM/" + file[i] + '.csv'

        df_RF_test.to_csv(s_RF_test,index=False)
        df_DT_test.to_csv(s_RF_test, index=False)
        df_SVM_test.to_csv(s_RF_test, index=False)
        df_KNN_test.to_csv(s_RF_test, index=False)
        df_GBM_test.to_csv(s_RF_test, index=False)
        df_LR_test.to_csv(s_RF_test, index=False)

        df_RF_train = pd.DataFrame(RF_feature_train)
        df_DT_train = pd.DataFrame(DT_feature_train)
        df_SVM_train = pd.DataFrame(SVM_feature_train)
        df_LR_train = pd.DataFrame(LR_feature_train)
        df_KNN_train = pd.DataFrame(KNN_feature_train)
        df_GBM_train = pd.DataFrame(GBM_feature_train)
        df_RF_train.columns = p_feature
        df_DT_train.columns = p_feature
        df_SVM_train.columns = p_feature
        df_LR_train.columns = p_feature
        df_KNN_train.columns = p_feature
        df_GBM_train.columns = p_feature
        s_RF_train = "C:/gln/mypackage/result/featureimportant/python/NT_tune/train/RF/" + file[i] + '.csv'
        s_DT_train = "C:/gln/mypackage/result/featureimportant/python/NT_tune/train/DT/" + file[i] + '.csv'
        s_SVM_train = "C:/gln/mypackage/result/featureimportant/python/NT_tune/train/SVM/" + file[i] + '.csv'
        s_LR_train = "C:/gln/mypackage/result/featureimportant/python/NT_tune/train/LR/" + file[i] + '.csv'
        s_KNN_train = "C:/gln/mypackage/result/featureimportant/python/NT_tune/train/KNN/" + file[i] + '.csv'
        s_GBM_train = "C:/gln/mypackage/result/featureimportant/python/NT_tune/train/GBM/" + file[i] + '.csv'

        df_RF_train.to_csv(s_RF_train, index=False)
        df_DT_train.to_csv(s_RF_train, index=False)
        df_SVM_train.to_csv(s_RF_train, index=False)
        df_KNN_train.to_csv(s_RF_train, index=False)
        df_GBM_train.to_csv(s_RF_train, index=False)
        df_LR_train.to_csv(s_RF_train, index=False)














