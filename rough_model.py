#TODO: Why is there NaN in entries???
import pandas as pd
import warnings 

warnings.filterwarnings("ignore")

# import csv dataset
thyroid_df = pd.read_csv('hypothyroid_ver2.csv')

# check composition of dataset (number of sick cases vs. non-sick cases)
#percent_sick = thyroid_df.thyroid_class.value_counts()['hypothyroid']/len(thyroid_df.index) * 100
#print(thyroid_df.thyroid_class.value_counts(), f"{percent_sick:.3}%")

#(Probably bad practice but idc) Convert the outcome (thyroid_class) into numeric variables
thyroid_df['thyroid_class'].replace(['hypothyroid', 'negative'], [1,0], inplace= True)

# well we know the dataset is moderately unbalanced (4% entries sick). 
# Thus, our results could be significantly improved if we employ oversampling on the training set.
# Let's first split up the dataset into training and testing sets

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

X, X_test, Y, Y_test = train_test_split(thyroid_df.drop(['thyroid_class'], axis= 1),
    thyroid_df['thyroid_class'], test_size = .3, random_state = 123)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = .3, random_state = 123)


# Preprocess data (includes data cleaning, feature engineering and oversampling for minority hypothyroid class)

import rough_preprocessor
categorical_cols = rough_preprocessor.rough_preprocessor(X_train)[1]
X_train_rand, Y_train_rand, X_train_smote, Y_train_smote = rough_preprocessor.training_pipeline(X_train, Y_train, categorical_cols)
#print(X_train_rand)

# OnehotEncoder to convert categorical variables into dummy variables
ohe = OneHotEncoder(categories='auto', drop=None, sparse= True, handle_unknown='ignore')
step = Pipeline(steps = [('onehot', ohe)])
tf = ColumnTransformer(transformers=[('categorical', step, categorical_cols)], remainder='passthrough')

X_train_rand = tf.fit_transform(X_train_rand)
X_train_smote = tf.fit_transform(X_train_smote)

#Add features

# Time to model the datasets:
# There are three general categories of classification models we can use: Traditional Classification, Artificial NN, and Gradient Boosting

# 1. For Traditional Classification we can test Logistic Regression:

# Implementation: 

import rough_model_trainer

trad_model, trad_model_2 = rough_model_trainer.logistic_classifier(
    X_train_rand, Y_train_rand, X_train_smote, Y_train_smote)

forest_model, forest_model_2 = rough_model_trainer.tree_classifier(
    X_train_rand, Y_train_rand, X_train_smote, Y_train_smote
)

# 2. For Artificial Neural Networks we can test Multilayer Perceptron:

mlp_model, mlp_model_2 = rough_model_trainer.mlp_classifier(
    X_train_rand, Y_train_rand, X_train_smote, Y_train_smote)

# 3. For Gradient Boosting LightGBM reports to have good results: 

grad_model, grad_model_2 = rough_model_trainer.lgbt_classifier(X_train_rand, Y_train_rand, X_train_smote, Y_train_smote)

# Model comparison:

import rough_validate

X_val = rough_preprocessor.val_test_pipeline(X_val)
X_val = tf.transform(X_val)

models = [trad_model, trad_model_2, grad_model, grad_model_2, forest_model, forest_model_2, mlp_model, mlp_model_2]
print(rough_validate.comp_f1(models, X_val, Y_val)) # The best model is Random Forest Classifier with SMOTE oversampling

# Hyperparameter Tuning:


