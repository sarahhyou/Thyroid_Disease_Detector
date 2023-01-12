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

X_train, X_test, Y_train, Y_test = train_test_split(thyroid_df.drop(['thyroid_class'], axis= 1),
    thyroid_df['thyroid_class'], test_size = .3, random_state = 123)

# Before we employ oversampling, the dataset contains a lot of a. missing points and b. categorical variables that need to be preprocessed.

import rough_preprocessor, rough_oversampler
X_train, X_train_smote, catcols = rough_preprocessor.rough_preprocessor(X_train)

# Now the data has been properly preprocessed and ready for sample selection.

X_train_rand, Y_train_rand = rough_oversampler.rough_oversampler_random(X_train, Y_train)
X_train_smote, Y_train_smote = rough_oversampler.rough_oversampler_smote(X_train_smote, Y_train, catcols)

# Preprocessing done! Time to model the datasets:
# There are three general categories of classification models we can use: Traditional Classification, Artificial NN, and Gradient Boosting

# 1. For Traditional Classification we can test Logistic Regression:

# Implementation: 

import rough_model_trainer

trad_model, trad_model_2 = rough_model_trainer.logistic_classifier(X_train_rand, Y_train_rand, X_train_smote, Y_train_smote)

# 2. For Artificial Neural Networks we can test Multilayer Perceptron:

# 3. For Gradient Boosting LightGBM reports to have good results: 

# Validation:

# Logistic Regression:

import rough_validate
rough_validate.rough_validate(X_test, Y_test, trad_model, trad_model_2)
crossval_1 = rough_validate.crossval(trad_model, X_train_rand, Y_train_rand)
crossval_2 = rough_validate.crossval(trad_model_2, X_train_smote, Y_train_smote)
print(rough_validate.model_comp(crossval_1, crossval_2))