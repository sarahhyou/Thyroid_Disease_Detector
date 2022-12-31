import pandas as pd

# import csv dataset
thyroid_df = pd.read_csv('hypothyroid.csv')

# check composition of dataset (number of sick cases vs. non-sick cases)
percent_sick = thyroid_df.sick.value_counts()['t']/len(thyroid_df.index) * 100
print(thyroid_df.sick.value_counts(), f"{percent_sick:.3}%")

# well we know the dataset is moderately unbalanced (4% entries sick). Thus, our results could be significantly improved if we employ oversampling on the training set.
# Let's first split up the dataset into training and testing sets
from sklearn.model_selection import train_test_split

thyroid_features, thyroid_result, X_test, Y_test = train_test_split(thyroid_df.drop(['sick'], axis= 1),
    thyroid_df['sick'], test_size = .15, random_state = 123)
# Further split up training set into 
X_train, Y_train, X_val, Y_val = train_test_split(thyroid_features, thyroid_result, test_size= .15, random_state= 123)

# Before we employ oversampling, the dataset contains a lot of a. missing points and b. categorical variables that need to be preprocessed.
# 

# 1. Random oversampling method:
# import packages
