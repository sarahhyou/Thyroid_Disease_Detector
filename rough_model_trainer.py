def logistic_classifier (x1, y1, x2, y2):
    from sklearn.linear_model import LogisticRegression
    model_1 = LogisticRegression()
    model_1.fit(x1, y1)
    model_2 = LogisticRegression()
    model_2.fit(x2, y2)
    return(model_1, model_2)

def lgbt_classifier (x1, y1, x2, y2):
    import lightgbm as lgbt
    model_1 = lgbt.LGBMClassifier()
    model_1.fit(x1, y1)
    model_2 = lgbt.LGBMClassifier()
    model_2.fit(x2, y2)
    return (model_1, model_2)
