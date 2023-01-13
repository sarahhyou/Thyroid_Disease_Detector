from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.stats import wilcoxon
import rough_preprocessor
def rough_validate(x, y, model, model_2):
    X_val, X_val_smote, cat_cols = rough_preprocessor.rough_preprocessor(x) 
    predict_1 = model.predict(X_val)
    predict_2 = model_2.predict(X_val)
    acc1 = accuracy_score(y, predict_1)
    acc2 = accuracy_score(y, predict_2)

    print(acc1, acc2)

def crossval(model, x, y):
    skf = StratifiedKFold(n_splits = 20)
    results = cross_val_score(model, x, y, cv = skf)
    return(results)

def model_comp(cv1, cv2):
    stat, p = wilcoxon(cv1, cv2, zero_method='zsplit'); p
    return ([stat, p]) 
