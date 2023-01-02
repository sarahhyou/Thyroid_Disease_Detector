import pandas as pd

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
    thyroid_df['thyroid_class'], test_size = .15, random_state = 123)

# Further split up non-testing set into training and validation sets

X_train, Y_train, X_val, Y_val = train_test_split(thyroid_features, thyroid_result, test_size= .15, random_state= 123)

# Before we employ oversampling, the dataset contains a lot of a. missing points and b. categorical variables that need to be preprocessed.
# 1. Missing Data
# The dataset contains many entries with missing data. There are two types of missing data: 
# partially missing (some data points missing in a column)
# and wholly missing (entire column has no data, as is the case of TBG)
# remove TBG measured & TBG as they give no useful information
# Also helpful to remove columns related to thyroxine, antithyroid medication, surgery, lithium, and goitre to prevent data leakeage.

print(list(X_train.columns))
X_train = X_train.drop(['TBG_measured', 'TBG', 'thyroxine', 'query_thyroxine', 'antithyroid_meds', 'thyroid_surgery', 'lithium', 'goitre'], axis= 1)
print(list(X_train.columns))

# 1. Random oversampling method:
# import packages
