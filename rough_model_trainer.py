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
    from tensorflow import keras
    from keras.wrappers.scikit_learn import KerasClassifier

    # Create basic model:

    def create_model(): #TODO: the MLP model isn't fitting on, figure out why
        model = keras.models.Sequential([
            keras.layers.Dense(300, activation='relu'),
            keras.layers.Dense(100, activation='relu'),
            keras.layers.Dense(2, activation='softmax')
        ])

        model.compile(loss='sparse_categorical_crossentropy', optimizer = 'sgd', metrics=['accuracy'])
        return model
    
    comp_model_1 = KerasClassifier(model=create_model, epochs=150, batch_size=10, verbose=0)
    actual_model_1 = create_model().fit(x1, y1)
    comp_model_2 = KerasClassifier(model=create_model, epochs=150, batch_size=10, verbose=0)
    actual_model_2 = create_model().fit(x2, y2)
    return(comp_model_1, actual_model_1, comp_model_2, actual_model_2)