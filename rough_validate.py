from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.stats import wilcoxon

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

def find_bigger_vector(v1, v2):
    counter = 0
    for i, j in zip(v1, v2):
        if i > j: counter += 1
    if counter > len(v1)/2 : return(v1) 
    # if at least half of the cross-validation scores for one model is larger
    # we consider the difference to be consistent enough for the model to be better
    else: return(v2) 

def better_model(m1, x1, y1, m2, x2, y2):
    cv1 = crossval(m1, x1, y1)
    cv2 = crossval(m2, x2, y2)
    if find_bigger_vector(cv1, cv2).all() == cv1.all(): print("Model 1 is better.")
    elif find_bigger_vector(cv1, cv2).all() == cv2.all(): print("Model 2 is better.")
    else: print ("There is no difference in the two models.")