import numpy as np, pandas as pd, rough_oversampler
from sklearn.experimental import enable_iterative_imputer # Must keep this line to enable MICE Imputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model
def rough_preprocessor (x):

# 1. Missing Data
# The dataset contains many entries with missing data. There are two types of missing data: 
# partially missing (some data points missing in a column)
# and wholly missing (entire column has no data, as is the case of TBG)
# remove TSH_measured, T3_measured, TT4_measured, T4U_measured, FTI_measured, TBG_measured and TBG as they give no useful information
# Also helpful to remove columns related to thyroxine, antithyroid medication, surgery, lithium, and goitre to prevent data leakeage.

    x = x.drop(['TSH_measured','T3_measured','TT4_measured','T4U_measured','FTI_measured','TBG_measured', 'TBG', 'thyroxine', 
'query_thyroxine', 'query_hyperthyroid', 'query_hypothyroid','antithyroid_meds', 'thyroid_surgery', 'lithium', 'goitre'], axis= 1)

# Now we deal with columns with only partially missing data. First we replace `?` with np.NaN and correctly recast data types:

    x = x.replace('?',np.nan)
    x[['age', 'TSH', 'T3', 'TT4', 'T4U','FTI']] = x[['age', 'TSH', 'T3', 'TT4', 'T4U','FTI']].apply(pd.to_numeric)

# Then we split the reminaing dataset into categorical and numerical columns and use SimpleImputer to fill in missing numerical values

    categorical_cols = [cname for cname in x.columns if x[cname].dtype == "object"]
    numerical_cols = [cname for cname in x.columns if x[cname].dtype in ['int64', 'float64']]
    x_num = x[numerical_cols].copy()
    x_cat = x[categorical_cols].copy()

# Because there is quite a number of data missing in each numerical column (100+ ~ 500+) 
# I decided on using MICE (Multivariate Imputation by Chained Equation) based on a Bayesian Ridge model 
# to maintain random variability and capture the relationship between thyroid hormone levels and age
    
    mice_imputer = IterativeImputer(estimator= linear_model.BayesianRidge(), n_nearest_features= None, imputation_order= 'ascending')
    x_num = pd.DataFrame(mice_imputer.fit_transform(x_num), columns = numerical_cols)
    
# Only a few data points were missing in the categorical columns
# Considering that women are more at risk of thyroid issues than men, more women are likely to be surveyed for this dataset
    
    x_cat['sex'] = x_cat['sex'].fillna('F')
    
# Transform categorical variables
    
    x_cat_onehot = pd.get_dummies(x_cat, columns = categorical_cols)
    
# Join numerical and categorical sub dataframes together and overwrite original dataframe
    
    x_num.index = x_cat.index
    x = pd.concat([x_num, x_cat_onehot], axis = 1)
    x_for_smote = pd.concat([x_num, x_cat], axis = 1)
    return (x, x_for_smote, categorical_cols)