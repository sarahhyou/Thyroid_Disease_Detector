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

def mlp_classifier(x1, y1, x2, y2):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import SGD
    from keras.wrappers.scikit_learn import KerasClassifier

    # Create basic model:

    def create_model(): #TODO: the MLP model isn't fitting on, figure out why
        clf = Sequential()
        clf.add(Dense(9, activation='relu', input_shape = (None, 10, 14)))
        clf.add(Dense(9, activation='relu'))
        clf.add(Dense(3, activation='relu'))
        clf.add(Dense(1, activation = 'sigmoid'))
        clf.compile(loss='binary_crossentropy', optimizer=SGD(), metrics=["accuracy"])
        return clf
    
    comp_model_1 = KerasClassifier(create_model, epochs=20, batch_size=10, verbose=0)
    comp_model_2 = KerasClassifier(create_model, epochs=20, batch_size=10, verbose=0)
    
    return(comp_model_1, comp_model_2)