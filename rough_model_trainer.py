def logistic_classifier (x1, y1, x2, y2):
    from sklearn.linear_model import LogisticRegression
    model_1 = LogisticRegression()
    model_1.fit(x1, y1)
    model_2 = LogisticRegression()
    model_2.fit(x2, y2)
    return(model_1, model_2)

def tree_classifier(x1, y1, x2, y2):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()
    clf.fit(x1, y1)
    clf2 = RandomForestClassifier()
    clf2.fit(x2, y2)
    return(clf, clf2)

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

    x_numcols = len(x1.columns)

    def create_model():
      model = Sequential()
      model.add(Dense(units=80,activation='relu'))
      model.add(Dense(units=20,activation='relu'))
      model.add(Dense(units=10,activation='relu'))
      model.add(Dense(units=5,activation='relu'))
      model.add(Dense(units=1,activation='sigmoid'))
      model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
      return model
    
    comp_model_1 = KerasClassifier(build_fn = create_model, epochs=20, batch_size=10, verbose = 0)
    comp_model_2 = KerasClassifier(build_fn = create_model, epochs=20, batch_size=10, verbose = 0)

    comp_model_1.fit(x1, y1)
    comp_model_2.fit(x2, y2)
    
    return(comp_model_1, comp_model_2)