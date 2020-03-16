""" 
A script to perform hyperparameter tuning of an RNN network on the MIT-BIH dataset.
""" 

import pandas as pd
import numpy as np
import os
from datetime import datetime

from sklearn.model_selection import train_test_split
import sklearn.decomposition as decomposition
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, accuracy_score, make_scorer, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

import tensorflow as tf
from tensorflow.keras import Model, Input, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM, GRU, BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau

from metrics import f1_m

df_train = pd.read_csv("../data/mitbih_train.csv", header=None)
df_train = df_train.sample(frac=1)
df_test = pd.read_csv("../data/mitbih_test.csv", header=None)

Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

n_class = np.unique(Y).size

def build_gru(n_class=5, dropout=0.3, rnn_sizes = [128, 128], fc_sizes=[64], batch_norm=True):
    nclass = 5
    model = Sequential()
    model.add(Input(shape=(187, 1)))
    
    if batch_norm:
        model.add(BatchNormalization())
        
    for index, dim in enumerate(rnn_sizes):
        model.add(GRU(dim, dropout=dropout, return_sequences=(index != len(rnn_sizes) - 1)))
        
        if batch_norm:
            model.add(BatchNormalization())
    
    for index, dim in enumerate(fc_sizes):
        model.add(Dense(dim, activation="relu"))
        model.add(Dropout(dropout))
        
        if batch_norm:
            model.add(BatchNormalization())
            
    model.add(Dense(nclass, activation="softmax"))

    return model

class CustomRNN(BaseEstimator):

    def __init__(self,  epochs=100, 
                        batch_size=64, 
                        learning_rate=1e-3, 
                        dropout=0.3, 
                        rnn_sizes=[128, 128],
                        fc_sizes=[64],
                        batch_norm=True):
        #print("Initializing model")
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.rnn_sizes = rnn_sizes
        self.fc_sizes = fc_sizes
        self.batch_norm = batch_norm

    def fit(self, train_X, train_y, **kwargs):
        tf.keras.backend.clear_session()
        #early = EarlyStopping(monitor="val_loss", mode="min", patience=5, verbose=1)
        redonplat = ReduceLROnPlateau(monitor="val_loss", mode="min", patience=3, verbose=2)
        callbacks_list = [redonplat]
        self.build_model()

        self.model.fit(train_X, train_y, validation_split=0.1, 
                                         epochs=self.epochs, 
                                         batch_size=self.batch_size, 
                                         callbacks=callbacks_list)
    
    def predict(self, eval_X):
        return np.argmax(self.model.predict(eval_X), axis=1)
    
    def set_params(self, epochs=100, 
                         batch_size=64, 
                         learning_rate=1e-3, 
                         dropout=0.3, 
                         rnn_sizes=[128, 128],
                         fc_sizes=[64],
                         batch_norm=True):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.rnn_sizes = rnn_sizes
        self.fc_sizes = fc_sizes
        self.batch_norm = batch_norm
                
        return self

    def get_params(self, deep=False):
      params = self.__dict__
      params.pop('model', None)
      return params
    
    """def score(self, eval_X, eval_y):
        predicted_y = np.argmax(self.model.predict(eval_X), axis=1)
        f1_score_ = f1_score(predicted_y, eval_y, average='macro')
        return f1_score_ """
        
    def build_model(self):
        print("Building model")
        self.model = build_gru(n_class=5, dropout=self.dropout, 
                          rnn_sizes=self.rnn_sizes, fc_sizes=self.fc_sizes, batch_norm=self.batch_norm)
        opt = optimizers.Adam(self.learning_rate)
            
        self.model.compile(optimizer=opt, 
                      loss="sparse_categorical_crossentropy", 
                      metrics=['accuracy'])
        #self.model.summary()       

params = {
    'epochs': [100],
    'batch_size': [128],
    'learning_rate': [1e-4],
    'dropout': [0.2],
    'rnn_sizes': [[128, 128, 64, 64], [128, 128, 128], [256, 256, 128]],
    'fc_sizes': [[64], [64, 64]],
    'batch_norm': [True, False]
}

model = CustomRNN()
search = GridSearchCV(model, 
                      params,
                      n_jobs=1,
                      cv=5,
                      return_train_score=True,
                      scoring={'f1_score': make_scorer(f1_score, average='macro'),
                               'accuracy': 'accuracy'}, 
                      refit='f1_score',   
                      verbose=10,
                      error_score='raise')
best = search.fit(X[:, :, :], Y[:])
print(best.__dict__)
print("BEST PARAMS: ", best.best_params_)

model_dir = 'models'
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

best_estimator = best.best_estimator_
best_estimator.fit(X, Y)
predicted_y = best_estimator.predict(X_test)
print("TEST EVALUATION")
print("F1-SCORE: ", f1_score(Y_test, predicted_y, average='macro'))
print("ACCURACY: ", accuracy_score(Y_test, predicted_y))
print(confusion_matrix(Y_test, predicted_y))

best_estimator.model.save(os.path.join(model_dir, str(datetime.now().strftime("%Y%m%d-%H%M%S"))))



