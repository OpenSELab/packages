from eli5.sklearn import PermutationImportance
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import multiprocessing as mp
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from imblearn import pipeline
from sklearn.neighbors import NearestNeighbors
from  sklearn.ensemble import GradientBoostingClassifier

def removedup(x,y):
    # print(x.shape,y.shape)
    df=pd.DataFrame(np.column_stack((x,y)))
    df=df.drop_duplicates()
    # print(df.shape)
    x_train=df.iloc[:,:df.shape[1]-1].values
    y_train=df.iloc[:,df.shape[1]-1].values
    return x_train,y_train
def rand(df):
    boot = np.random.choice(df.shape[0], df.shape[0], replace=True)
    oob = [x for x in [i for i in range(0, df.shape[0])] if x not in boot]
    df1=df.iloc[boot]
    df2=df.iloc[oob]

    defe = df1[df1["label"] == 1]
    clean = df1[df1["label"] == 0]

    defe1 = df2[df2["label"] == 1]
    clean1 = df2[df2["label"] == 0]

    if (defe.shape[0]!=0 and clean.shape[0]!=0 and defe1.shape[0]!=0 and clean1.shape[0]!=0):
        return boot
    else:
        rand(df)

    return boot
def __populate(nnarray,n, y,y_label):
    # T=0
    number=0
    for i in range(n):
        label=y[nnarray[i]]
        if(y_label==label):
            number=number+1

    return number

def addoverlaplabe(df):
    laplabel = np.zeros(shape=[df.shape[0], 1])
    # print(df.columns)
    df1=df
    X = df1.iloc[:, :df1.shape[1] - 1].values
    # print(X)
    y = df1.iloc[:, df1.shape[1] - 1].values

    nbrs1 = NearestNeighbors(n_neighbors=5, algorithm="auto",metric="euclidean")
    nbrs1.fit(X)
    nnarray1 = nbrs1.kneighbors(X)[1]
    for i in range(nnarray1.shape[0]):
        y_label = y[i]
        num = __populate(nnarray1[i], 5, y, y_label)
        if (num < 3 ):
            laplabel[i] = 1
    df1["laplabel"] = laplabel
    return df1[df1["laplabel"]==0]

def getclean(nnarray,n,x,y):
    test_data=pd.DataFrame([])
    for i in range(n):
        if y[nnarray[i]]==0:
            new_data = pd.DataFrame(x[nnarray[i] - 1:nnarray[i], :])
            new_data["label"]=y[nnarray[i]]
            # print(new_data)
            test_data=pd.concat([test_data,new_data],axis=0)
    return test_data

def NCL(df):
    X = df.iloc[:, :df.shape[1] - 1].values
    y = df.iloc[:, df.shape[1] - 1].values
    # print(y)
    df_bug=df[df["label"] == 1]
    df_clean=df[df["label"]==0]
    # print(df_clean)
    X_bug=df_bug.iloc[:,:df_bug.shape[1]-1].values
    nbrs1 = NearestNeighbors(n_neighbors=3, algorithm="auto",metric="euclidean")
    nbrs1.fit(X)
    nnarray1 = nbrs1.kneighbors(X_bug)[1]
    test_data=pd.DataFrame([])
    for i in range(nnarray1.shape[0]):
        new_data=getclean(nnarray1[i],3,X,y)
        # print(new_data)
        test_data=pd.concat([test_data,new_data],axis=0)

    df_clean=pd.concat([df_clean,test_data],axis=0)
    df1=df_clean.drop_duplicates(keep=False)
    df2=pd.concat([df1,df_bug],axis=0)
    return df2

def calcluster(df):
    k = int(df.shape[0] / 10)
    x = df.iloc[:, :df.shape[1]-1]
    kmeans = KMeans(n_clusters=k, random_state=0).fit(x)
    list=kmeans.labels_
    # print(k)
    # print(list)
    df["clusterlabel"] = list
    return df

def KMCL(df):
    df_bug = df[df["label"] == 1]
    df_clean = df[df["label"] == 0]
    p=df_bug.shape[0]/df_clean.shape[0]
    print(p)
    df_new=calcluster(df)
    k = int(df.shape[0] / 10)
    df_train = pd.DataFrame([])
    for i in range(k):
        arr = df_new[df_new['clusterlabel'] == i]
        n0 = arr[arr['label'] == 0].shape[0]
        n1 = arr[arr['label'] == 1].shape[0]
        # print(n0, n1)
        if n0 == 0 or n1 / n0 >= p:
            # if n0 == 0 or n1 / n0 >= 2:
            data = arr[arr['label'] == 1]
            df_train = pd.concat([df_train, data], axis=0)
        else:
            data = arr[arr['label'] == 0]
            df_train = pd.concat([df_train, data], axis=0)

    return df_train

def tunparameter(x_train,y_train):
    param_range_learning_rate = [0.05, 0.01, 0.1]
    param_range_max_depth = range(5, 15, 5)
    # param_range_max_features = [5, 8, 10, 15]
    param_grid = [{'clf__learning_rate': param_range_learning_rate, 'clf__max_depth': param_range_max_depth}]
    pipe_rf = Pipeline([('clf', GradientBoostingClassifier())])
    best_score = 0
    for k in range(1, 20, 5):
        sm = SMOTE(k_neighbors=k)
        x_res, y_res = sm.fit_resample(x_train, y_train)
        gs = GridSearchCV(estimator=pipe_rf, param_grid=param_grid, scoring='roc_auc')
        gs.fit(x_res, y_res)
        score = np.mean(gs.cv_results_['mean_test_score'])
        if (score > best_score):
            best_gs = gs
    return best_gs


def tunoverlapparameter(x,y):
    param_range_learning_rate = [0.05, 0.01, 0.1]
    param_range_max_depth = range(5, 15, 5)
    # param_range_max_features = [5, 8, 10, 15]

    param_grid = [{'clf__learning_rate': param_range_learning_rate, 'clf__max_depth': param_range_max_depth}]
    best_score = 0
    pipe_rf = Pipeline([('clf', GradientBoostingClassifier())])
    for k in range(1, 20, 5):
        sm = SMOTE(k_neighbors=k)
        x_res, y_res = sm.fit_resample(x, y)
        df = pd.DataFrame(x_res)
        df["label"] = y_res
        df_new = addoverlaplabe(df)
        x_new = df_new.iloc[:, :df_new.shape[1] - 2].values
        y_new = df_new.iloc[:, df_new.shape[1] - 2].values

        gs = GridSearchCV(estimator=pipe_rf, param_grid=param_grid, scoring='roc_auc')
        gs.fit(x_new, y_new)
        score = np.mean(gs.cv_results_['mean_test_score'])
        if (score > best_score):
            best_gs = gs
    return best_gs


def PermutationImportance_(clf, X, y):
    perm = PermutationImportance(clf, n_iter=5, random_state=1024, cv=5)

    perm.fit(X, y)

    # result_ = {'var': var
    #     , 'feature_importances_': perm.feature_importances_
    #     , 'feature_importances_std_': perm.feature_importances_std_}
    # feature_importances_ = pd.DataFrame(result_, columns=['var', 'feature_importances_', 'feature_importances_std_'])
    # feature_importances_ = feature_importances_.sort_values('feature_importances_', ascending=False)
    return perm.feature_importances_


def fun(file):
    df = pd.read_csv('C:/gln/myclassoverlap/Data/new/gooddata/data_clean/'+file)
    scaler = StandardScaler()
    # scaler1 = StandardScaler()
    # scaler2 = StandardScaler()


    feature_original=[]
    feature_remove=[]
    feature_SMOTE=[]
    feature_ovim=[]
    feature_imov=[]
    feature_NCL=[]
    feature_IKMCCA=[]
    # feature_separating = []

    for t in range(100):
        boot = rand(df)
        train = df.iloc[boot]
        oob = [x for x in [i for i in range(0, df.shape[0])] if x not in boot]  # testing data
        test = df.iloc[oob]

        x_train_original = train.iloc[:, :train.shape[1] - 3]
        y_train_original = train.iloc[:, train.shape[1] - 2].values
        # print(y_train_original)
        var = x_train_original.columns.values

        x_train_original = scaler.fit_transform(x_train_original)

        train_original = pd.DataFrame(x_train_original)
        train_original["label"] = y_train_original

        # print(file)
        # print(train_original)
        train_NCL = NCL(train_original)
        x_NCL = train_NCL.iloc[:, :train_NCL.shape[1] - 1].values
        y_NCL = train_NCL.iloc[:, train_NCL.shape[1] - 1].values
        # print(y_NCL)

        print(file)
        # print(train_original)
        train_IKMCCA = KMCL(train_original)
        # print(train_IKMCCA)
        x_IKMCCA = train_IKMCCA.iloc[:, :train_IKMCCA.shape[1] - 2].values
        y_IKMCCA = train_IKMCCA.iloc[:, train_IKMCCA.shape[1] - 2].values
        # print(y_IKMCCA)

        train_original1 = pd.DataFrame(x_train_original)
        train_original1["label"] = y_train_original
        train_remove = addoverlaplabe(train_original1)
        x_train_overlap = train_remove.iloc[:, :train_remove.shape[1] - 2].values
        y_train_overlap = train_remove.iloc[:, train_remove.shape[1] - 2].values

        # SL = test['loc'].values
        # defnum = test['bugs'].values

        x_test = test.iloc[:, :test.shape[1] - 3].values
        y_test = test.iloc[:, test.shape[1] - 2].values
        x_test = scaler.transform(x_test)

        pipe_rf = Pipeline([('clf', GradientBoostingClassifier())])
        param_range_learning_rate = [0.05, 0.01, 0.1]
        param_range_max_depth = range(5, 15, 5)

        param_grid = [{'clf__learning_rate': param_range_learning_rate, 'clf__max_depth': param_range_max_depth}]
        # Perform grid search cross validation on the parameters listed in param_grid, using accuracy as the measure of fit and number of folds (CV) = 5
        gs = GridSearchCV(estimator=pipe_rf, param_grid=param_grid)
        gs.fit(x_train_original, y_train_original)
        # best_rf_params = gs.best_params_
        # clf = DecisionTreeClassifier(criterion=gs.best_params_['clf__criterion'],max_depth=gs.best_params_['clf__max_depth'],min_samples_leaf=gs.best_params_['clf__min_samples_leaf'])
        # clf.fit(x_train_original, y_train_original)
        feature_importances_1= PermutationImportance_(gs, x_test, y_test)
        feature_original.append(feature_importances_1)
        # print(feature_original.shape)

        print(feature_importances_1)
        # x_train_remove1,y_train_remove1=removedup(x_train_remove,y_train_remove)
        gs1 = GridSearchCV(estimator=pipe_rf, param_grid=param_grid)
        gs1.fit(x_train_overlap, y_train_overlap)
        # best_rf_params = gs.best_params_
        # clf1 = DecisionTreeClassifier(criterion=gs1.best_params_['clf__criterion'],max_depth=gs1.best_params_['clf__max_depth'],min_samples_leaf=gs1.best_params_['clf__min_samples_leaf'])
        #
        # clf1.fit(x_train_overlap, y_train_overlap)
        # x_test2=scaler1.transform(x_test)
        feature_importances_2= PermutationImportance_(gs1, x_test, y_test)
        print(feature_importances_2)
        feature_remove.append(feature_importances_2)


        clf2=tunparameter(x_train_original,y_train_original)
        feature_importances_3 = PermutationImportance_(clf2, x_test, y_test)
        print(feature_importances_3)
        feature_SMOTE.append(feature_importances_3)

        clf3 = tunparameter(x_train_overlap, y_train_overlap)
        feature_importances_4 = PermutationImportance_(clf3, x_test, y_test)
        print(feature_importances_4)
        feature_ovim.append(feature_importances_4)

        clf4 = tunoverlapparameter(x_train_original,y_train_original)
        feature_importances_5 = PermutationImportance_(clf4, x_test, y_test)
        print(feature_importances_5)
        feature_imov.append(feature_importances_5)

        gs5 = GridSearchCV(estimator=pipe_rf, param_grid=param_grid, scoring='roc_auc', cv=5)
        gs5.fit(x_NCL, y_NCL)
        # clf5 = DecisionTreeClassifier(criterion=gs5.best_params_['clf__criterion'],
        #                              max_depth=gs5.best_params_['clf__max_depth'],
        #                              min_samples_leaf=gs5.best_params_['clf__min_samples_leaf'])
        # clf5.fit(x_NCL, y_NCL)
        feature_importances_6 = PermutationImportance_(gs5, x_test, y_test)
        feature_NCL.append(feature_importances_6)

        gs6 = GridSearchCV(estimator=pipe_rf, param_grid=param_grid, scoring='roc_auc', cv=5)
        gs6.fit(x_IKMCCA, y_IKMCCA)
        # clf6 = DecisionTreeClassifier(criterion=gs6.best_params_['clf__criterion'],
        #                               max_depth=gs6.best_params_['clf__max_depth'],
        #                               min_samples_leaf=gs6.best_params_['clf__min_samples_leaf'])
        # clf6.fit(x_IKMCCA, y_IKMCCA)
        feature_importances_7 = PermutationImportance_(gs6, x_test, y_test)
        feature_IKMCCA.append(feature_importances_7)

    feature_original=pd.DataFrame(feature_original)
    print(feature_original.shape)
    feature_original.columns=var
    feature_remove=pd.DataFrame(feature_remove)
    feature_remove.columns=var
    feature_SMOTE=pd.DataFrame(feature_SMOTE)
    feature_SMOTE.columns=var
    feature_ovim = pd.DataFrame(feature_ovim)
    feature_ovim.columns = var
    feature_imov = pd.DataFrame(feature_imov)
    feature_imov.columns = var
    feature_NCL = pd.DataFrame(feature_NCL)
    feature_NCL.columns = var
    feature_IKMCCA = pd.DataFrame(feature_IKMCCA)
    feature_IKMCCA.columns = var
    ss='C:/gln/myclassoverlap/imbalanceoverlap/result/feature/GBM/original/original_'+file
    feature_original.to_csv(ss,index=False)
    ss1='C:/gln/myclassoverlap/imbalanceoverlap/result/feature/GBM/remove/remove_'+file
    feature_remove.to_csv(ss1,index=False)
    ss2 = 'C:/gln/myclassoverlap/imbalanceoverlap/result/feature/GBM/SMOTE/SMOTE_' + file
    feature_SMOTE.to_csv(ss2, index=False)
    ss3 = 'C:/gln/myclassoverlap/imbalanceoverlap/result/feature/GBM/ovim/ovim_' + file
    feature_ovim.to_csv(ss3, index=False)
    ss4 = 'C:/gln/myclassoverlap/imbalanceoverlap/result/feature/GBM/imov/imov_' + file
    feature_imov.to_csv(ss4, index=False)
    ss5 = 'C:/gln/myclassoverlap/imbalanceoverlap/result/feature/GBM/NCL/NCL_' + file
    feature_NCL.to_csv(ss5, index=False)
    ss6 = 'C:/gln/myclassoverlap/imbalanceoverlap/result/feature/GBM/IKMCCA/IKMCCA_' + file
    feature_IKMCCA.to_csv(ss6, index=False)

if __name__ == '__main__':
    N =18
    g1 = os.scandir(r"/Data/new/gooddata/data_clean/")

    with mp.Pool(processes=N) as p:
        results = p.map(fun, [file.name for file in g1])
