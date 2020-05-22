import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns

def get_cleaned_data(file):
    data = pd.read_csv(file, index_col=0)
    data = data.drop([89,524,636,706,1299])
    data = clean_data(data)
    data = pd.get_dummies(data, drop_first=True)
    target = data.SalePrice
    data = data.drop('SalePrice', axis=1)
    return data, target

def get_eda_data(file):
    data = pd.read_csv(file, index_col=0)
    target = data.SalePrice
    data = data.drop('SalePrice', axis=1)
    data, _ , _ , _ = train_test_split(data, target, test_size=.3, random_state=14)
    return data

def clean_data(df):
    #filling na
    df.Alley        = df.Alley.fillna(value = 'NoAlley')
    df.BsmtCond     = df.BsmtCond.fillna(value = 'NoBsmt')
    df.BsmtQual     = df.BsmtQual.fillna(value = 'NoBsmt')
    df.BsmtExposure = df.BsmtExposure.fillna(value= 'NoBsmt')
    df.BsmtFinType1 = df.BsmtFinType1.fillna(value= 'NoBsmt')
    df.BsmtFinType2 = df.BsmtFinType2.fillna(value= 'NoBsmt')
    df.FireplaceQu  = df.FireplaceQu.fillna(value = 'Nofireplace')
    df.GarageType   = df.GarageType.fillna(value = 'NoGarage')
    df.GarageCond   = df.GarageCond.fillna(value = 'NoGarage')
    df.GarageFinish = df.GarageFinish.fillna(value = 'NoGarage')
    df.GarageQual   = df.GarageQual.fillna(value = 'NoGarage')
    df.PoolQC       = df.PoolQC.fillna(value = 'NoPool')
    df.Fence        = df.Fence.fillna(value = 'NoFence')
    df.MiscFeature  = df.MiscFeature.fillna(value = 'NoMisc')
    df.LotFrontage  = df.LotFrontage.fillna(df.groupby('LotConfig')['LotFrontage'].transform('mean'))
    df.GarageYrBlt  = df.GarageYrBlt.fillna(df.YearBuilt)
    df.GarageYrBlt  = df.GarageYrBlt - df.YearBuilt
    df.MasVnrType   = df.MasVnrType.fillna(value = 'None')
    df.MasVnrArea   = df.MasVnrArea.fillna(value = 0)
    df              = df.dropna()
    
    #dropping uniform features
    df = df.drop('Street', axis=1)
    df = df.drop('Utilities', axis=1)
    df = df.drop('Condition2', axis=1)
    df = df.drop ('RoofMatl', axis=1)
    df = df.drop('Heating', axis=1)
    df = df.drop('LowQualFinSF', axis=1)
    df = df.drop('3SsnPorch', axis=1)
    df = df.drop('PoolArea', axis=1)
    df = df.drop('MiscFeature', axis=1)
    df = df.drop('MoSold', axis=1)

    #converting to numeric
    df.BsmtCond     = df.BsmtCond.map({'Ex':5 ,'Gd':4 , 'TA':3 ,'Fa':2 ,'Po':1 , 'NoBsmt':0})
    df.BsmtQual     = df.BsmtQual.map({'Ex':5 ,'Gd':4 , 'TA':3 ,'Fa':2 ,'Po':1 , 'NoBsmt':0})
    df.BsmtExposure = df.BsmtExposure.map({'Gd':4, 'Av':3, 'Mn':2, 'No':1, 'NoBsmt':0})
    df.BsmtFinType1 = df.BsmtFinType1.map({'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NoBsmt':0})
    df.BsmtFinType2 = df.BsmtFinType2.map({'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NoBsmt':0})

    df.GarageCond   = df.GarageCond.map({'NoGarage':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
    df.GarageQual   = df.GarageQual.map({'NoGarage':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
    df.GarageFinish = df.GarageFinish.map({'Fin':3, 'RFn':2, 'Unf':1, 'NoGarage':0})
    df.PavedDrive   = df.PavedDrive.map({'Y':3,'P':2, 'N':1 })

    df.MSZoning     = df.MSZoning.map({'FV':1,'C (all)':2,"RL":3,'RM':4,'RH':5})
    df.Alley        = df.Alley.map({'NoAlley':0,'Grvl':1, 'Pave':1})
    df.LotShape     = df.LotShape.map({'Reg':4, 'IR1':3, 'IR2':2, 'IR3':1})
    df.ExterCond    = df.ExterCond.map({"Ex":5,'Gd':4,'TA':3,'Fa':2,'Po':1})
    df.ExterQual    = df.ExterQual.map({"Ex":5,'Gd':4,'TA':3,'Fa':2,'Po':1})

    df.CentralAir   = df.CentralAir.map({'Y':1, 'N':0})
    df.HeatingQC    = df.HeatingQC.map({"Ex":5,'Gd':4,'TA':3,'Fa':2,'Po':1})
    df.FireplaceQu  = df.FireplaceQu.map({"Ex":5,'Gd':4,'TA':3,'Fa':2,'Po':1, 'Nofireplace':0})
    df.KitchenQual  = df.KitchenQual.map({"Ex":5,'Gd':4,'TA':3,'Fa':2,'Po':1})
    df.PoolQC       = df.PoolQC.map({"Ex":4,'Gd':3,'TA':2,'Fa':1,'NoPool':0})
    df.Fence        = df.Fence.map({'GdPrv':4 , 'MnPrv':3 , 'GdWo':2 , 'MnWw':1 , 'NoFence':0})

    df.LandSlope    = df.LandSlope.map({'Sev':2, 'Mod':1, 'Gtl':0})
    df.Functional   = df.Functional.map({'Typ':7,'Min1':6,'Min2':5,'Mod':4,'Maj1':3,'Maj2':2,'Sev':1})
    df.LandContour  = df.LandContour.map({'Lvl':0,'Bnk':1, 'Low':1, 'HLS':1})
    df.Electrical   = df.Electrical.map({'FuseA':3, 'FuseF':2, 'FuseP':1, 'Mix':0, 'SBrkr':4})

    df.Functional   = df.Functional.map({'Maj1':3, 'Maj2':3, 'Min1':1, 'Min2':1, 'Mod':2, 'Sev':4, 'Typ':0})
    df.PoolQC       = df.PoolQC.map({'Ex':4, 'Fa':1, 'Gd':3, 'NoPool':0})

    #Dropping uncorrelated data or duplicate columns
    df = df.drop('Functional', axis=1)
    df = df.drop('PoolQC', axis=1)
    df = df.drop('BsmtFinType2', axis=1)
    df = df.drop('GarageQual', axis=1)
    df = df.drop('GarageCars', axis=1)
    df = df.drop('FireplaceQu', axis=1)
    df = df.drop('BsmtHalfBath', axis=1)
    df = df.drop('BsmtFinSF2', axis=1)
    
    return df

def hist_LandContour(X_train_uncleaned):
    plt.figure(figsize=(8,4))
    X_train_uncleaned.LandContour.hist();
    plt.title('Land contour Hist');

def plot_price_distribution(y_train, y_train_log):
    plt.figure(figsize=(16,4))
    plt.subplot(1,2,1)
    y_train.plot(kind='hist', bins=30)
    plt.title('SalePrice Distribution')
    plt.xlabel('Dollars')
    
    plt.subplot(1,2,2)
    y_train_log.plot(kind='hist', bins=30)
    plt.title('log(SalePrice) Distribution')
    plt.xlabel('log of Dollars')


def plot_pca_and_lda(X_train, y_train):
    scaler = StandardScaler()
    data_std = scaler.fit_transform(X_train)
    label = ['0-120,000 dollars' if price <= 120000 else '120,000-200,000 dollars' if price <= 200000 else '200,000+ dollars' for price in y_train]

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_std)

    lda = LinearDiscriminantAnalysis()
    data_lda = lda.fit_transform(data_std, label)

    plt.figure(figsize=(16,5))

    plt.subplot(1,2,1)
    sns.scatterplot(data_pca[:, 0], data_pca[:, 1], hue=label)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA')

    plt.subplot(1,2,2)
    sns.scatterplot(data_lda[:, 0], data_lda[:, 1], hue=label)
    plt.xlabel('LDA Component 1')
    plt.ylabel('LDA Component 2')
    plt.title('LDA')

def get_features_importances_df(X_train, y_train):
    
    columns = X_train.columns

    GBR = GradientBoostingRegressor(learning_rate= 0.1, n_estimators=500, max_depth=2)
    GBR.fit(X_train, np.log(y_train))
    feature_importances_gbr = sorted(zip(columns, GBR.feature_importances_), reverse=True, key=lambda x: x[1])
    feature_df_gbr = pd.DataFrame(feature_importances_gbr, columns=['feature', 'importance'])

    rf = RandomForestClassifier(max_depth=35, min_samples_leaf=2, min_samples_split=4, n_estimators=250)
    rf.fit(X_train, y_train)
    feature_importances_rf = sorted(zip(columns, rf.feature_importances_), reverse=True, key=lambda x: x[1])
    feature_df_rf = pd.DataFrame(feature_importances_rf, columns=['feature', 'importance'])

    thresholder = VarianceThreshold()
    thresholder.fit(X_train)
    feature_importances_var = sorted(zip(columns, thresholder.variances_), reverse=True, key=lambda x: x[1])
    feature_df_var = pd.DataFrame(feature_importances_var, columns=['feature', 'importance'])

    features_compare = pd.DataFrame()
    features_compare['var'] = feature_df_var.feature
    features_compare['gbr'] = feature_df_gbr.feature
    features_compare['rf'] = feature_df_rf.feature

    return features_compare

def graph_feature_importances(X_train, y_train):
    features_compare = get_features_importances_df(X_train, y_train)
    top_n_features = features_compare.iloc[:10, :].stack().value_counts()
    
    plt.figure(figsize=(16,4))
    top_n_features[:40].plot(kind='bar')
    plt.title('Number of models that feature is in list of top 10 features')
    plt.ylabel('Number of models')
    plt.xticks(rotation=45);
    plt.yticks([1,2,3]);

def graph_SVC_score_increasing_number_features(X_train, y_train, y_train_binned):
    feature_df = get_features_importances_df(X_train, y_train)
    max_num_features = 31
    best_params_svc = {'C': 0.01, 'gamma': 0.0001, 'kernel': 'linear'}

    f1_macro = []
    accuracy = []
    recall = []
    precision = []

    for num_features in range(1,max_num_features):
        features = set(feature_df.iloc[:num_features, :].stack())
        data_partial = X_train.loc[:, features]

        pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', SVC(**best_params_svc))])
        scores = cross_validate(pipeline, data_partial, y_train_binned, cv=10, n_jobs=-1, return_train_score=True,
                                scoring=['f1_macro', 'recall_macro', 'precision_macro', 'accuracy'])

        f1_macro.append(scores['test_f1_macro'].mean())
        accuracy.append(scores['test_accuracy'].mean())
        recall.append(scores['test_recall_macro'].mean())
        precision.append(scores['test_precision_macro'].mean())
        
    plt.figure(figsize=(16,6))
    plt.plot(range(1, max_num_features), f1_macro, label='f1_macro')
    plt.plot(range(1, max_num_features), accuracy, label='accuracy')
    plt.plot(range(1, max_num_features), recall, label='recall')
    plt.plot(range(1, max_num_features), precision, label='precision')
    plt.xticks(range(1, max_num_features))
    plt.legend()
    plt.xlabel('Top N Features from all feature ratings')
    plt.ylabel('metric score %')
    plt.title('SVM Classifier')
    
def graph_RFC_score_increasing_number_features(X_train, y_train, y_train_binned):
    feature_df = get_features_importances_df(X_train, y_train)
    max_num_features = 31
    best_params_rfc={'max_depth': 20, 'min_samples_leaf': 3, 'min_samples_split': 8, 'n_estimators': 100}

    f1_macro = []
    accuracy = []
    recall = []
    precision = []

    for num_features in range(1,max_num_features):
        features = set(feature_df.iloc[:num_features, :].stack())
        data_partial = X_train.loc[:, features]

        pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', RandomForestClassifier(**best_params_rfc))])
        scores = cross_validate(pipeline, data_partial, y_train_binned, cv=10, n_jobs=-1, return_train_score=True,
                                scoring=['f1_macro', 'recall_macro', 'precision_macro', 'accuracy'])

        f1_macro.append(scores['test_f1_macro'].mean())
        accuracy.append(scores['test_accuracy'].mean())
        recall.append(scores['test_recall_macro'].mean())
        precision.append(scores['test_precision_macro'].mean())
        
    plt.figure(figsize=(16,6))
    plt.plot(range(1, max_num_features), f1_macro, label='f1_macro')
    plt.plot(range(1, max_num_features), accuracy, label='accuracy')
    plt.plot(range(1, max_num_features), recall, label='recall')
    plt.plot(range(1, max_num_features), precision, label='precision')
    plt.xticks(range(1, max_num_features))
    plt.legend()
    plt.xlabel('Top N Features from all feature ratings')
    plt.ylabel('metric score %')
    plt.title('Random Forest Classifier')
    
def graph_ridge_score_increasing_number_features(X_train, y_train):
    feature_df = get_features_importances_df(X_train, y_train)
    max_num_features = 31

    train = []
    test = []

    for num_features in range(1,max_num_features):
        features = set(feature_df.iloc[:num_features, :].stack())
        data_partial = X_train.loc[:, features]

        pipeline = Pipeline([('scaler', StandardScaler()), ('estimator', Ridge(alpha=.05))])
        scores = cross_validate(pipeline, data_partial, np.log(y_train), cv=10, n_jobs=-1, return_train_score=True)

        train.append(scores['train_score'].mean())
        test.append(scores['test_score'].mean())
        
    plt.figure(figsize=(16,6))
    plt.plot(range(1, max_num_features), train, label='Train')
    plt.plot(range(1, max_num_features), test, label='Test')
    plt.xticks(range(1, max_num_features))
    plt.legend()
    plt.xlabel('Top N Features from all feature ratings')
    plt.ylabel('metric score %')
    plt.title('Linear Regression (Ridge Regularization)')
    
def graph_lasso_score_increasing_number_features(X_train, y_train):
    feature_df = get_features_importances_df(X_train, y_train)
    max_num_features = 31

    train = []
    test = []

    for num_features in range(1,max_num_features):
        features = set(feature_df.iloc[:num_features, :].stack())
        data_partial = X_train.loc[:, features]

        pipeline = Pipeline([('scaler', StandardScaler()), ('estimator', Lasso(alpha=.0005))])
        scores = cross_validate(pipeline, data_partial, np.log(y_train), cv=10, n_jobs=-1, return_train_score=True)

        train.append(scores['train_score'].mean())
        test.append(scores['test_score'].mean())
        
    plt.figure(figsize=(16,6))
    plt.plot(range(1, max_num_features), train, label='Train')
    plt.plot(range(1, max_num_features), test, label='Test')
    plt.xticks(range(1, max_num_features))
    plt.legend()
    plt.xlabel('Top N Features from all feature ratings')
    plt.ylabel('metric score %')
    plt.title('Linear Regression (Lasso Regularization)')
    
def graph_rf_score_increasing_number_features(X_train, y_train):
    feature_df = get_features_importances_df(X_train, y_train)
    max_num_features = 21

    train = []
    test = []
    
    best_params_rf = {'max_depth': 30, 'min_samples_leaf': 1,'min_samples_split': 2, 'n_estimators': 300}

    for num_features in range(1,max_num_features):
        features = set(feature_df.iloc[:num_features, :].stack())
        data_partial = X_train.loc[:, features]

        pipeline = RandomForestRegressor(**best_params_rf)
        scores = cross_validate(pipeline, data_partial, np.log(y_train), cv=10, n_jobs=-1, return_train_score=True)

        train.append(scores['train_score'].mean())
        test.append(scores['test_score'].mean())
        
    plt.figure(figsize=(16,6))
    plt.plot(range(1, max_num_features), train, label='Train')
    plt.plot(range(1, max_num_features), test, label='Test')
    plt.xticks(range(1, max_num_features))
    plt.legend()
    plt.xlabel('Top N Features from all feature ratings')
    plt.ylabel('metric score %')
    plt.title('Random Forest Regression')
    
def graph_svr_score_increasing_number_features(X_train, y_train):
    feature_df = get_features_importances_df(X_train, y_train)
    max_num_features = 31

    train = []
    test = []
    
    best_params_svr = {'C': 0.1, 'kernel': 'linear'}

    for num_features in range(1,max_num_features):
        features = set(feature_df.iloc[:num_features, :].stack())
        data_partial = X_train.loc[:, features]

        pipeline = Pipeline([('scaler', StandardScaler()), ('estimator', SVR(**best_params_svr))])
        scores = cross_validate(pipeline, data_partial, np.log(y_train), cv=10, n_jobs=-1, return_train_score=True)

        train.append(scores['train_score'].mean())
        test.append(scores['test_score'].mean())
        
    plt.figure(figsize=(16,6))
    plt.plot(range(1, max_num_features), train, label='Train')
    plt.plot(range(1, max_num_features), test, label='Test')
    plt.xticks(range(1, max_num_features))
    plt.legend()
    plt.xlabel('Top N Features from all feature ratings')
    plt.ylabel('metric score %')
    plt.title('SVM Regression')

def graph_clf_scores():
    
    plt.figure(figsize=(16,8))
    barWidth = 0.25

    #f1
    plt.subplot(2,2,1)
    val_error = [0.0006733632111376462, 0.04285034378236204, 0.051259169328772315]

    # set height of bar
    train = [0.2275475220267404,
             0.9362245609295536,
             0.8341622749061172]
    validation = [0.2275496270802373,
                  0.8097853991726909,
                  0.817839868932561]
    test = [0.23184079601990049, 0.8110689920948051, 0.8252475673813829]

    # Set position of bar on X axis
    r1 = np.arange(len(train))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    plt.bar(r1, train, color='#7f6d5f', width=barWidth, edgecolor='white', label='train')
    plt.bar(r2, validation, color='#557f2d', width=barWidth, edgecolor='white', label='validate', yerr=val_error)
    plt.bar(r3, test, color='#2d7f5e', width=barWidth, edgecolor='white', label='test')

    # Add xticks on the middle of the group bars
    plt.title('f1 score', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(train))], ['Dummy', 'RandomForest', 'SVM'])
    plt.legend()

    #accuracy
    plt.subplot(2,2,2)
    val_error = [0.0023250079343804337, 0.038329483446028524, 0.03565814067546885]

    # set height of bar
    train = [0.5181908924967281,
             0.9424256713219938,
             0.8529465011867259]
    validation = [0.5182016914427022,
                  0.8280306774252525,
                  0.8408254519341141]
    test = [0.5331807780320366, 0.8283752860411899, 0.8489702517162472]

    # Set position of bar on X axis
    r1 = np.arange(len(train))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    plt.bar(r1, train, color='#7f6d5f', width=barWidth, edgecolor='white', label='train')
    plt.bar(r2, validation, color='#557f2d', width=barWidth, edgecolor='white', label='validate', yerr=val_error)
    plt.bar(r3, test, color='#2d7f5e', width=barWidth, edgecolor='white', label='test')

    # Add xticks on the middle of the group bars
    plt.title('accuracy score', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(train))], ['Dummy', 'RandomForest', 'SVM'])
    plt.legend()

    #recall
    plt.subplot(2,2,3)
    val_error = [5.551115123125783e-17, 0.04469248484639341, 0.05576784743462415]

    # set height of bar
    train = [0.33333333333333337,
             0.9232344328264668,
             0.8100834099270703]
    validation = [0.33333333333333337,
                  0.7903445662551277,
                  0.7956379833400368]
    test = [0.3333333333333333, 0.8053906541157235, 0.8069318714610308]

    # Set position of bar on X axis
    r1 = np.arange(len(train))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    plt.bar(r1, train, color='#7f6d5f', width=barWidth, edgecolor='white', label='train')
    plt.bar(r2, validation, color='#557f2d', width=barWidth, edgecolor='white', label='validate', yerr=val_error)
    plt.bar(r3, test, color='#2d7f5e', width=barWidth, edgecolor='white', label='test')

    # Add xticks on the middle of the group bars
    plt.title('recall score', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(train))], ['Dummy', 'RandomForest', 'SVM'])
    plt.legend()

    #precision
    plt.subplot(2,2,4)
    val_error = [0.0007750026447934849, 0.046503431828439266, 0.03785522743464524]

    # set height of bar
    train = [0.1727302974989094,
             0.9522937528358334,
             0.8738345240133493]
    validation = [0.17273389714756743,
                  0.8433565655151808,
                  0.8618894432429327]
    test = [0.17772692601067885, 0.8214717224918137, 0.8567973540572374]

    # Set position of bar on X axis
    r1 = np.arange(len(train))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    plt.bar(r1, train, color='#7f6d5f', width=barWidth, edgecolor='white', label='train');
    plt.bar(r2, validation, color='#557f2d', width=barWidth, edgecolor='white', label='validate', yerr=val_error);
    plt.bar(r3, test, color='#2d7f5e', width=barWidth, edgecolor='white', label='test');

    # Add xticks on the middle of the group bars
    plt.title('precision score', fontweight='bold');
    plt.xticks([r + barWidth for r in range(len(train))], ['Dummy', 'RandomForest', 'SVM']);
    plt.legend();
    
    plt.tight_layout()

def graph_reg_scores():
    plt.figure(figsize=(10,5))
    barWidth = 0.25
    val_error = [0.02581770078453299, 0.0259824771586376, 0.022769871586745077, 0.024863216391633164]

    # set height of bar
    train = [0.9073082736423019,
             0.9074269743267204,
             0.982299857026382,
             0.9057892422310181]
    validation = [0.8993150446798763,
                  0.8992762271487311,
                  0.8653374918145496,
                  0.8987882583891276]
    test = [0.8938701438797594, 0.892623556439032, 0.8653296343750956, 0.8954935569599229]

    # Set position of bar on X axis
    r1 = np.arange(len(train))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    plt.bar(r1, train, color='#7f6d5f', width=barWidth, edgecolor='white', label='train')
    plt.bar(r2, validation, color='#557f2d', width=barWidth, edgecolor='white', label='validate', yerr=val_error)
    plt.bar(r3, test, color='#2d7f5e', width=barWidth, edgecolor='white', label='test')

    # Add xticks on the middle of the group bars
    plt.title('R-Squared', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(train))], ['Linear Regression (lasso)',
                                                           'Linear Regression (ridge)', 
                                                           'Random Forest',
                                                           'SVM'])
    plt.legend(loc=4);


def graph_cum_variance(X_train, y_train):

        scaler = StandardScaler()
        data_std = scaler.fit_transform(X_train)

        pca = PCA()
        pca.fit(data_std)
        cum_var = np.cumsum(pca.explained_variance_ratio_)

        plt.figure(figsize=(16,4))
        plt.bar(range(len(cum_var)), cum_var)    
    
def get_scores_rfc(X_train, X_test, y_train, y_test):
    """prints f1 train_score and test_score for random forest classifier"""
    best_params_rfc={'max_depth': 20,
                     'min_samples_leaf': 3,
                     'min_samples_split': 8,
                     'n_estimators': 100}
    
    rfc = RandomForestClassifier(**best_params_rfc)
    
    rfc_features = ['GrLivArea', 'OverallQual', 'GarageArea', 'BsmtUnfSF', 'KitchenQual',
                    'MiscVal', '2ndFlrSF', 'LotArea', 'GarageFinish', '1stFlrSF',
                    'TotalBsmtSF']
    
    rfc.fit(data.loc[:,rfc_features], target_binned)
    print(rfc.score(data.loc[:,rfc_features], target_binned),
          rfc.score(data_test.loc[:,rfc_features], target_test_binned))
    
def get_scores_rf(X_train, X_test, y_train, y_test):
    """prints r2 train_score and test_score for random forest regressor"""
    best_params_rf = {'max_depth': 30,
                      'min_samples_leaf': 1,
                      'min_samples_split': 2,
                      'n_estimators': 300}
    
    rf = RandomForestRegressor(**best_params_rf)
    
    rf_features = ['GrLivArea', '1stFlrSF', 'GarageArea', 'TotalBsmtSF', 'BsmtFinSF1',
                   'OverallQual', '2ndFlrSF', 'LotArea', 'YearBuilt', 'BsmtUnfSF',
                   'MasVnrArea', 'BsmtQual', 'OverallCond', 'MiscVal', 'KitchenQual',
                   'GarageFinish']
    
    rf.fit(data.loc[:,rf_features], target_log)
    print(rf.score(data.loc[:,rf_features], target_log),
          rf.score(data_test.loc[:,rf_features], target_test_log))
    
def get_scores_lasso(X_train, X_test, y_train, y_test):
    """prints r2 train_score and test_score for random forest regressor"""
    
    lr_features = ['BsmtFinType1', 'LotArea', 'YearRemodAdd', 'GrLivArea', '2ndFlrSF',
                   '1stFlrSF', 'GarageArea', 'TotalBsmtSF', 'YearBuilt', 'BsmtFinSF1',
                   'OpenPorchSF', 'Fireplaces', 'MSZoning', 'LotFrontage', 'OverallQual',
                   'OverallCond', 'BsmtQual', 'BsmtUnfSF', 'CentralAir', 'GarageFinish',
                   'Neighborhood_Crawfor', 'EnclosedPorch', 'WoodDeckSF', 'MSSubClass',
                   'GarageYrBlt', 'ScreenPorch', 'MiscVal', 'KitchenQual', 'MasVnrArea']
    
    pipeline_lasso = Pipeline([('scaler', StandardScaler()), ('estimator', Lasso(alpha=.0005))])
    pipeline_lasso.fit(data.loc[:,lr_features], target_log)
    
    print(pipeline_lasso.score(data.loc[:,lr_features], target_log),
          pipeline_lasso.score(data_test.loc[:,lr_features], target_test_log))
    
def get_scores_ridge(X_train, X_test, y_train, y_test):
    """prints r2 train_score and test_score for random forest regressor"""
    
    lr_features = ['BsmtFinType1', 'LotArea', 'YearRemodAdd', 'GrLivArea', '2ndFlrSF',
                   '1stFlrSF', 'GarageArea', 'TotalBsmtSF', 'YearBuilt', 'BsmtFinSF1',
                   'OpenPorchSF', 'Fireplaces', 'MSZoning', 'LotFrontage', 'OverallQual',
                   'OverallCond', 'BsmtQual', 'BsmtUnfSF', 'CentralAir', 'GarageFinish',
                   'Neighborhood_Crawfor', 'EnclosedPorch', 'WoodDeckSF', 'MSSubClass',
                   'GarageYrBlt', 'ScreenPorch', 'MiscVal', 'KitchenQual', 'MasVnrArea']
    
    pipeline_ridge = Pipeline([('scaler', StandardScaler()), ('estimator', Ridge(alpha=.05))])
    pipeline_ridge.fit(data.loc[:,lr_features], target_log)
    print(pipeline_ridge.score(data.loc[:,lr_features], target_log),
          pipeline_ridge.score(data_test.loc[:,lr_features], target_test_log))


def hist_data_columns(data, columns, ncols=4):
    nrows = len(columns) // ncols + 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(18,nrows*3))
    for ax, column in zip(axs.ravel(), columns):
        data[column].hist(ax=ax)
        ax.set_title(column)
    
    fig.tight_layout()

    
def classification_target(data):
    return [0 if price <= 120000 else 1 if price <= 200000 else 2 for price in data['SalePrice']]


def Heatmap(df):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(12,12))
    sns.heatmap(corr, ax=ax, cmap='seismic', center= 0.0, square = True);

def top10Heatmap(df):
    corr = df.corr()
    cols = corr.nlargest(11, 'SalePrice')['SalePrice'].index 
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(12,12))
    sns.heatmap(corr, ax=ax, xticklabels=cols, yticklabels=cols, fmt='.2f',annot=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.xticks(rotation=45) 
    plt.yticks(rotation=45); 
    
def get_train_test_split(data, log=False):
    if log == True:
        y = np.log(data['SalePrice'])
    else:
        y = data['SalePrice']
    x = data.drop(labels = ['SalePrice', 'Id'], axis=1).astype("float64")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)
    return x_train, x_test, y_train, y_test

def get_binned_train_test_split(data):
    target = [0 if price <= 120000 else 1 if price <= 200000 else 2 for price in data['SalePrice']]
    x = data.drop(labels = 'SalePrice', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, target, test_size = 0.3, random_state = 42)
    return x_train, x_test, y_train, y_test
