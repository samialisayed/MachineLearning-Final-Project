import pandas as pd
import sys

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
    return df


def dropping_uniform_features(data):
    data = data.drop('Street', axis=1)
    data = data.drop('Utilities', axis=1)
    data = data.drop('Condition2', axis=1)
    data = data.drop ('RoofMatl', axis=1)
    data = data.drop('Heating', axis=1)
    data = data.drop('LowQualFinSF', axis=1)
    data = data.drop('3SsnPorch', axis=1)
    data = data.drop('PoolArea', axis=1)
    data = data.drop('MiscFeature', axis=1)
    data = data.drop('MoSold', axis=1)
    return data

def convering_to_numeric(data):
    data.BsmtCond = data.BsmtCond.map({'Ex':5 ,'Gd':4 , 'TA':3 ,'Fa':2 ,'Po':1 , 'NoBsmt':0})
    data.BsmtQual = data.BsmtQual.map({'Ex':5 ,'Gd':4 , 'TA':3 ,'Fa':2 ,'Po':1 , 'NoBsmt':0})
    data.BsmtExposure = data.BsmtExposure.map({'Gd':4, 'Av':3, 'Mn':2, 'No':1, 'NoBsmt':0})
    data.BsmtFinType1 = data.BsmtFinType1.map({'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NoBsmt':0})
    data.BsmtFinType2 = data.BsmtFinType2.map({'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NoBsmt':0})

    data.GarageCond = data.GarageCond.map({'NoGarage':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
    data.GarageQual = data.GarageQual.map({'NoGarage':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
    data.GarageFinish = data.GarageFinish.map({'Fin':3, 'RFn':2, 'Unf':1, 'NoGarage':0})
    data.PavedDrive = data.PavedDrive.map({'Y':3,'P':2, 'N':1 })

    data.MSZoning = data.MSZoning.map({'FV':1,'C (all)':2,"RL":3,'RM':4,'RH':5})
    data.Alley = data.Alley.map({'NoAlley':0,'Grvl':1, 'Pave':1})
    data.LotShape = data.LotShape.map({'Reg':4, 'IR1':3, 'IR2':2, 'IR3':1})
    data.ExterCond = data.ExterCond.map({"Ex":5,'Gd':4,'TA':3,'Fa':2,'Po':1})
    data.ExterQual = data.ExterQual.map({"Ex":5,'Gd':4,'TA':3,'Fa':2,'Po':1})

    data.CentralAir = data.CentralAir.map({'Y':1, 'N':0})
    data.HeatingQC = data.HeatingQC.map({"Ex":5,'Gd':4,'TA':3,'Fa':2,'Po':1})
    data.FireplaceQu = data.FireplaceQu.map({"Ex":5,'Gd':4,'TA':3,'Fa':2,'Po':1, 'Nofireplace':0})
    data.KitchenQual = data.KitchenQual.map({"Ex":5,'Gd':4,'TA':3,'Fa':2,'Po':1})
    data.PoolQC = data.PoolQC.map({"Ex":4,'Gd':3,'TA':2,'Fa':1,'NoPool':0})
    data.Fence = data.Fence.map({'GdPrv':4 , 'MnPrv':3 , 'GdWo':2 , 'MnWw':1 , 'NoFence':0})

    data.LandSlope = data.LandSlope.map({'Sev':2, 'Mod':1, 'Gtl':0})
    data.Functional = data.Functional.map({'Typ':7,'Min1':6,'Min2':5,'Mod':4,'Maj1':3,'Maj2':2,'Sev':1})
    data.LandContour = data.LandContour.map({'Lvl':0,'Bnk':1, 'Low':1, 'HLS':1})
    data.Electrical = data.Electrical.map({'FuseA':3, 'FuseF':2, 'FuseP':1, 'Mix':0, 'SBrkr':4})

    data.Functional = data.Functional.map({'Maj1':3, 'Maj2':3, 'Min1':1, 'Min2':1, 'Mod':2, 'Sev':4, 'Typ':0})
    data.PoolQC = data.PoolQC.map({'Ex':4, 'Fa':1, 'Gd':3, 'NoPool':0})
    return data

def dropping_uncorrelated_features(data):
    data = data.drop('Functional', axis=1)
    data = data.drop('PoolQC', axis=1)
    data = data.drop('BsmtFinType2', axis=1)
    data = data.drop('GarageQual', axis=1)
    data = data.drop('GarageCars', axis=1)
    data = data.drop('FireplaceQu', axis=1)
    data = data.drop('BsmtHalfBath', axis=1)
    data = data.drop('BsmtFinSF2', axis=1)

    data = data.drop('Id', axis=1)
    
    return data

def process_data(data):
    data = pd.read_csv(sys.argv[1])
    
    data = handling_missing_values(data)
    data = dropping_uniform_features(data)
    data = convering_to_numeric(data)
    data = dropping_uncorrelated_features(data)
    
    numerical_data = pd.get_dummies(data, drop_first=True)
    return numerical_data
    
if __name__ == '__main__':
    process_data(data).to_csv(sys.argv[1])
    
