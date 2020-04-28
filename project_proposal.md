Regression on Ames Housing Prices
Financial & Commerce

Sami Ali
Tova Schwartz
Nur Deniz Turhan

Motivation: 
Real estate is at the crossroads of several sectors. The financial sector is greatly invested in the real estate market, private citizens are the mainstay of the mortgage industry, and housing, in general, is a great public welfare concern. If there’s one lesson to be learned from the 2007-2008 housing crisis it is the significant relevance of the real estate market within a modern economy.
Our data can be found on kaggle at: https://www.kaggle.com/c/house-prices-advancedregression-techniques. It was constructed by Dean De Cock. The data has approximately 3000 rows of information on house listings in Ames, Texas. The columns of the data are highly interpretable and there are almost 80 variables, thus it will be good for applying machine learning algorithms. 
We hope to accomplish two or three general goals towards gaining a greater understanding of the housing market. Firstly, we hope to create a model that can predict house price; either a regression to predict the exact price or a classification model to predict a bin. Then we want to look at specific trends based on location. Lastly, we hope to analyze the relationship between variables and their relationship with the target variable price.

Method: 
We can do some exploratory analysis such as an investigation of features distribution and checking the correlation between the features and the target or between the features with each other. The first step will be data preprocessing such as dealing with missing values, using summary statistics; encoding categorical variables, using one-hot encoding; and dealing with outliers. Secondly, we have too many features, thus dimension reduction will be one of our main points whether by feature selection or creating new features. Example techniques we may use are lasso regression, decision tree, gradient boosting, forward or backward feature selection, and correlation analysis. Lastly, we will do necessary standardization. We will then model a linear regression with different regularization techniques; we will do this on the data as a whole and on the data grouped by neighborhood. Lastly, we will run several experiments on our data to understand the relationship between the different variables and our target variable of price.

Intended experiments: 
We will run several experiments on our data to understand the relationship between the different variables and our target variable of price. We will look at both: the regression coefficients and a heatmap of the correlation between the variables to understand the big picture. Then, we want to create a constant price range to see the effects and dependencies of the different variables. This can be done by running a PCA and ICA and then analyzing the principle components. 
Our data is split into two sets: test set and train set. We will evaluate our model using statistical tools for measurement such as R-squared, and similar metric scores from the sklearn library. We will also use cross validation when creating our model to reduce bias.
