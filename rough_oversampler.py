# 1. Random oversampling method:
# import packages
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTENC

def rough_oversampler_random (x, y):
    random_os = RandomOverSampler(sampling_strategy = 0.3, random_state = 123) # we just want minority (hypothyroid) class to appear more often
    # oversampling minority class until it equals majority class can lead to model overfitting
    x_rand, y_rand = random_os.fit_resample(x, y)
    return (x_rand, y_rand)

# 2. SMOTE oversampling method:

def rough_oversampler_smote(x, y, cols):
    smote_os = SMOTENC(categorical_features= [6,7,8,9], random_state = 123, sampling_strategy= 0.3) # categorical data in last 4 columns 
    x_smote, y_smote = smote_os.fit_resample(x, y)
    x_smote = pd.get_dummies(x_smote, columns = cols)
    return (x_smote, y_smote)
