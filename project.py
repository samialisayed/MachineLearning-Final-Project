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
    df.PavedDrive = df.PavedDrive.map({'Y':2,'P':1, 'N':0 })

    df.MSZoning = df.MSZoning.map({"RL":4,'RM':3,'RH':2,'C (all)':1,'FV':0})
    df.Street = df.Street.map({'Pave':0,'Grvl':1})
    df.Alley = df.Alley.map({'NoAlley':0,'Grvl':1, 'Pave':0})
    df.LotShape = df.LotShape.map({'Reg':0, 'IR1':1, 'IR2':2, 'IR3':3})
    df.ExterCond = df.ExterCond.map({"Ex":4,'Gd':3,'TA':2,'Fa':1,'Po':0})
    df.ExterQual = df.ExterQual.map({"Ex":4,'Gd':3,'TA':2,'Fa':1,'Po':0})

    df.CentralAir = df.CentralAir.map({'Y':1, 'N':0})
    df.HeatingQC = df.HeatingQC.map({"Ex":4,'Gd':3,'TA':2,'Fa':1,'Po':0})
    df.FireplaceQu = df.FireplaceQu.map({"Ex":5,'Gd':4,'TA':3,'Fa':2,'Po':1, 'Nofireplace':0})
    df.KitchenQual = df.KitchenQual.map({"Ex":4,'Gd':3,'TA':2,'Fa':1,'Po':0})
    df.PoolQC = df.PoolQC.map({"Ex":4,'Gd':3,'TA':2,'Fa':1,'NoPool':0})
    df.Fence = df.Fence.map({'GdPrv':4 , 'MnPrv':3 , 'GdWo':2 , 'MnWw':1 , 'NoFence':0})
