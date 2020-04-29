import pandas as pd
import sys

def fill_missing_values(df):
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

def get_data(input_file):
    df = pd.read_csv(input_file)
    fill_missing_values(df)
    return df
    
if __name__ == '__main__':
    get_data(sys.argv[1]).to_csv(sys.stdout,index=False)
    
