from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.stats import wilcoxon
import rough_preprocessor

#def rough_validate(x, y, model):
#    predict = model.predict(x)
#    acc = accuracy_score(y, predict)
#    print(acc)

def crossval(model, x, y):
    skf = StratifiedKFold(n_splits = 20)
    results = cross_val_score(model, x, y, cv = skf)
    return(results)

def model_comp(cv1, cv2):
    stat, p = wilcoxon(cv1, cv2, zero_method='zsplit'); p
    return ([stat, p]) 

def better_model(m1, x1, y1, m2, x2, y2):
    cv1 = crossval(m1, x1, y1)
    cv2 = crossval(m2, x2, y2)
    stats = model_comp(cv1, cv2)
    if stats[1] < 0.05 and cv1.all() > cv2.all(): print("Random oversampling is better.")
    elif stats[1] < 0.05 and cv2.all() > cv1.all(): print("SMOTE oversampling is better.")
    else: print ("There is no difference in the two models.")