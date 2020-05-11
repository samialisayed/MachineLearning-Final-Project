import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


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

def handling_missing_values(df):
    df.Alley = df.Alley.fillna(value = 'NoAlley')
    df.BsmtCond = df.BsmtCond.fillna(value = 'NoBsmt')
    df.BsmtQual = df.BsmtQual.fillna(value = 'NoBsmt')
    df.BsmtExposure = df.BsmtExposure.fillna(value= 'NoBsmt')
    df.BsmtFinType1 = df.BsmtFinType1.fillna(value= 'NoBsmt')
    df.BsmtFinType2 = df.BsmtFinType2.fillna(value= 'NoBsmt')
    df.FireplaceQu = df.FireplaceQu.fillna(value = 'Nofireplace')
    df.GarageType = df.GarageType.fillna(value = 'NoGarage')
    df.GarageCond = df.GarageCond.fillna(value = 'NoGarage')
    df.GarageFinish = df.GarageFinish.fillna(value = 'NoGarage')
    df.GarageQual = df.GarageQual.fillna(value = 'NoGarage')
    df.PoolQC = df.PoolQC.fillna(value = 'NoPool')
    df.Fence = df.Fence.fillna(value = 'NoFence')
    df.MiscFeature = df.MiscFeature.fillna(value = 'NoMisc')
    df.LotFrontage.fillna(df.groupby('LotConfig')['LotFrontage'].transform('mean'), inplace=True)
    df.GarageYrBlt = df.GarageYrBlt.fillna(df.YearBuilt)
    df.GarageYrBlt = df.GarageYrBlt - df.YearBuilt
    df.MasVnrType = df.MasVnrType.fillna(value = 'None')
    df.MasVnrArea = df.MasVnrArea.fillna(value = 0)
    df.dropna(inplace=True)
    
def classification_target(data):
    return [0 if price <= 120000 else 1 if price <= 200000 else 2 for price in data['SalePrice']]

def preprocessed_data(data):
    """returns preprocessed data as numpy array ready for decision tree"""
    
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    
    if 'SalePrice' in data.columns:
        data=data.drop('SalePrice', axis=1)
    if 'Id' in data.columns:
        data=data.drop('Id', axis=1)
        
    data=data.astype({'LotFrontage': 'int64', 'MasVnrArea': 'int64', 'GarageYrBlt': 'int64'})
    
    categorical_data = ['MSSubClass', 'MSZoning', 'Alley', 'LandContour', 'LotConfig',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'Foundation', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'Functional', 'GarageType',
       'Fence', 'MiscFeature', 'YrSold', 'SaleType', 'SaleCondition']
    
    numeric_data = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
                'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
                'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
                'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt',
                'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
                'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold']
    
    boolean_data = ['Street', 'CentralAir']
    
    ordinal_data = ['LotShape', 'Utilities', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual',
                'BsmtCond', 'BsmtExposure', 'HeatingQC', 'Electrical', 'KitchenQual', 'FireplaceQu',
                'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', dtype='int8'), categorical_data),
            ('ord', OrdinalEncoder(dtype='int8'), ordinal_data+boolean_data),
            ('num', 'passthrough', numeric_data)])
    
    return preprocessor.fit_transform(data)
    
def restoring_data(data):
    columns = data[0].split(',')
    data = pd.DataFrame(data[1:])[0].str.split(',',expand=True)
    data.columns = columns
    for col, dtype in zip(columns, data_types):
        data[col] = data[col].astype(dtype)
    return data

def Heatmap(df):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(12,12))
    sns.heatmap(corr, ax=ax, cmap='seismic', center= 0.0, square = True);

def top10Heatmap(df):
    corr = df.corr()
    cols = corr.nlargest(11, 'SalePrice')['SalePrice'].index 
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(12,12))
    sns.heatmap(corr, ax=ax, xticklabels=cols, yticklabels=cols, annot=True);

def LabelEncoder(df):
    df.BsmtCond = df.BsmtCond.map({'Ex':5 ,'Gd':4 , 'TA':3 ,'Fa':2 ,'Po':1 , 'NoBsmt':0})
    df.BsmtQual = df.BsmtQual.map({'Ex':5 ,'Gd':4 , 'TA':3 ,'Fa':2 ,'Po':1 , 'NoBsmt':0})
    df.BsmtExposure = df.BsmtExposure.map({'Gd':4, 'Av':3, 'Mn':2, 'No':1, 'NoBsmt':0})
    df.BsmtFinType1 = df.BsmtFinType1.map({'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NoBsmt':0})
    df.BsmtFinType2 = df.BsmtFinType2.map({'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NoBsmt':0})
    
    df.GarageType = df.GarageType.map({'BuiltIn':6, 'Attchd': 5, '2Types':4 , 'Basment':3 , 'Detchd':2, 'CarPort' :1 , 'NoGarage': 0})
    df.GarageCond = df.GarageCond.map({'NoGarage':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
    df.GarageQual = df.GarageQual.map({'NoGarage':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
    df.GarageFinish = df.GarageFinish.map({'Fin':3, 'RFn':2, 'Unf':1, 'NoGarage':0})
    df.PavedDrive = df.PavedDrive.map({'Y':3,'P':2, 'N':1 })
    
    df.MSZoning = df.MSZoning.map({'FV':1,'C (all)':2,"RL":3,'RM':4,'RH':5})
    df.Street = df.Street.map({'Pave':2,'Grvl':1})
    df.Alley = df.Alley.map({'NoAlley':0,'Grvl':1, 'Pave':2})
    df.LotShape = df.LotShape.map({'Reg':4, 'IR1':3, 'IR2':2, 'IR3':1})
    df.ExterCond = df.ExterCond.map({"Ex":5,'Gd':4,'TA':3,'Fa':2,'Po':1})
    df.ExterQual = df.ExterQual.map({"Ex":5,'Gd':4,'TA':3,'Fa':2,'Po':1})
    
    df.CentralAir = df.CentralAir.map({'Y':1, 'N':0})
    df.HeatingQC = df.HeatingQC.map({"Ex":5,'Gd':4,'TA':3,'Fa':2,'Po':1})
    df.FireplaceQu = df.FireplaceQu.map({"Ex":5,'Gd':4,'TA':3,'Fa':2,'Po':1, 'Nofireplace':0})
    df.KitchenQual = df.KitchenQual.map({"Ex":5,'Gd':4,'TA':3,'Fa':2,'Po':1})
    df.PoolQC = df.PoolQC.map({"Ex":4,'Gd':3,'TA':2,'Fa':1,'NoPool':0})
    df.Fence = df.Fence.map({'GdPrv':4 , 'MnPrv':3 , 'GdWo':2 , 'MnWw':1 , 'NoFence':0})
    
    df.LandSlope = df.LandSlope.map({'Sev':3,'Mod':2, 'Gtl':1 })
    df.Functional = df.Functional.map({'Typ':7,'Min1':6,'Min2':5,'Mod':4,'Maj1':3,'Maj2':2,'Sev':1})
    df.SaleCondition = df.SaleCondition.map({'Normal':6,'Abnorml':5,'AdjLand':4,'Alloca':3,'Family':2,'Partial':1})

    pd.get_dummies(df, drop_first= True)
    
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