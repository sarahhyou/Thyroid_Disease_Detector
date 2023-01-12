from sklearn.metrics import accuracy_score
import rough_preprocessor
def rough_validate(x, y, model, model_2):
    X_val, X_val_smote, cat_cols = rough_preprocessor.rough_preprocessor(x) 
    predict_1 = model.predict(X_val)
    predict_2 = model_2.predict(X_val)
    acc1 = accuracy_score(y, predict_1)
    acc2 = accuracy_score(y, predict_2)

    print(acc1, acc2)