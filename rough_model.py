import pandas as pd
import numpy as np

# import csv dataset
thyroid_df = pd.read_csv('hypothyroid_ver2.csv')

# check composition of dataset (number of sick cases vs. non-sick cases)
percent_sick = thyroid_df.thyroid_class.value_counts()['hypothyroid']/len(thyroid_df.index) * 100
print(thyroid_df.sick.value_counts(), f"{percent_sick:.3}%")

#(Probably bad practice but idc) Convert the outcome (thyroid_class) into numeric variables
thyroid_df['thyroid_class'].replace(['hypothyroid', 'negative'], [1,0], inplace= True)

# well we know the dataset is moderately unbalanced (4% entries sick). 
# Thus, our results could be significantly improved if we employ oversampling on the training set.
# Let's first split up the dataset into training and testing sets

from sklearn.model_selection import train_test_split

thyroid_features, X_test, thyroid_result, Y_test = train_test_split(thyroid_df.drop(['thyroid_class'], axis= 1),
    thyroid_df['thyroid_class'], test_size = .1, random_state = 123)

# Further split up non-testing set into training and validation sets

X_train, X_val, Y_train, Y_val = train_test_split(thyroid_features, thyroid_result, test_size= .1, random_state= 123)

# Before we employ oversampling, the dataset contains a lot of a. missing points and b. categorical variables that need to be preprocessed.
# 1. Missing Data
# The dataset contains many entries with missing data. There are two types of missing data: 
# partially missing (some data points missing in a column)
# and wholly missing (entire column has no data, as is the case of TBG)
# remove TSH_measured, T3_measured, TT4_measured, T4U_measured, FTI_measured, TBG_measured and TBG as they give no useful information
# Also helpful to remove columns related to thyroxine, antithyroid medication, surgery, lithium, and goitre to prevent data leakeage.  

X_train = X_train.drop(['TSH_measured','T3_measured','TT4_measured','T4U_measured','FTI_measured','TBG_measured', 'TBG', 'thyroxine', 
'query_thyroxine', 'query_hyperthyroid', 'query_hypothyroid','antithyroid_meds', 'thyroid_surgery', 'lithium', 'goitre'], axis= 1)

# Now we deal with columns with only partially missing data. First we replace `?` with np.NaN and correctly recast data types:

X_train = X_train.replace('?',np.nan)
X_train[['age', 'TSH', 'T3', 'TT4', 'T4U','FTI']] = X_train[['age', 'TSH', 'T3', 'TT4', 'T4U','FTI']].apply(pd.to_numeric)

# Then we split the reminaing dataset into categorical and numerical columns and use SimpleImputer to fill in missing numerical values

categorical_cols = [cname for cname in X_train.columns if X_train[cname].dtype == "object"]
numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]
X_train_num = X_train[numerical_cols].copy()
X_train_cat = X_train[categorical_cols].copy()

# Because there is quite a number of data missing in each numerical column (100+ ~ 500+) 
# I decided on using MICE (Multivariate Imputation by Chained Equation) based on a Bayesian Ridge model 
# to maintain random variability and capture the relationship between thyroid hormone levels and age

from sklearn.experimental import enable_iterative_imputer # Must keep this line to enable MICE Imputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model

mice_imputer = IterativeImputer(estimator= linear_model.BayesianRidge(), n_nearest_features= None, imputation_order= 'ascending')
X_train_num = pd.DataFrame(mice_imputer.fit_transform(X_train_num), columns = numerical_cols)

# Only a few data points were missing in the categorical columns
# Considering that women are more at risk of thyroid issues than men, more women are likely to be surveyed for this dataset

X_train_cat['sex'] = X_train_cat['sex'].fillna('F')

# Transform categorical variables

X_train_cat_onehot = pd.get_dummies(X_train_cat, columns = categorical_cols)

# Join numerical and categorical sub dataframes together and overwrite original dataframe

X_train_num.index = X_train_cat.index
X_train = pd.concat([X_train_num, X_train_cat], axis = 1)
X_train_one = pd.concat([X_train_num, X_train_cat_onehot], axis = 1)

# Now the data has been properly preprocessed and ready for sample and feature selection.
# 1. Random oversampling method:
# import packages

from imblearn.over_sampling import RandomOverSampler

random_os = RandomOverSampler(sampling_strategy = 0.3) # we just want minority (hypothyroid) class to appear more often
# oversampling minority class until it equals majority class can lead to model overfitting
X_train_rand, Y_train_rand = random_os.fit_resample(X_train_one, Y_train)

# 2. SMOTE oversampling method:

from imblearn.over_sampling import SMOTENC

smote_os = SMOTENC(categorical_features= [6,7,8,9], random_state = 123, sampling_strategy= 0.3) # categorical data in last 4 columns 
X_train_smote, Y_train_smote = smote_os.fit_resample(X_train, Y_train)
X_train_smote = pd.get_dummies(X_train_smote, columns = categorical_cols)
