def hist_data_columns(data, columns=data.columns, ncols=4):
    nrows = len(columns) // ncols + 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(18,nrows*3))
    for ax, column in zip(axs.ravel(), columns):
        data[column].hist(ax=ax)
        ax.set_title(column)
    
    fig.tight_layout()

def handling(df):
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
