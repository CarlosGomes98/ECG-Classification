import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import sklearn.decomposition as decomposition
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, accuracy_score, make_scorer
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

import matplotlib.pyplot as plt
import seaborn as sb
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

df_train = pd.read_csv("data/mitbih_train.csv", header=None)
df_train = df_train.sample(frac=1)
df_test = pd.read_csv("data/mitbih_test.csv", header=None)

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
    def fit(self, train_X, train_y, **kwargs):
        
        self.build_model()
        
        # early = EarlyStopping(monitor="val_accuracy", mode="max", patience=5, verbose=1)
        # redonplat = ReduceLROnPlateau(monitor="val_accuracy", mode="max", patience=3, verbose=2)
        # callbacks_list = [checkpoint, early, redonplat]  # early
        self.model.fit(train_X, train_y, validation_split=0.1, epochs=self.epochs, batch_size=self.batch_size)
    
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
    
    def score(self, eval_X, eval_y):
        predicted_y = np.argmax(self.model.predict(eval_X), axis=1)
        f1_score_ = f1_score(predicted_y, eval_y, average='macro')
        print("f1 score: ", f1_score_)
        return f1_score_
        
    def build_model(self):
        self.model = build_gru(n_class=5, dropout=self.dropout, 
                          rnn_sizes=self.rnn_sizes, fc_sizes=self.fc_sizes, batch_norm=self.batch_norm)
        opt = optimizers.Adam(self.learning_rate)
            
        self.model.compile(optimizer=opt, 
                      loss="sparse_categorical_crossentropy", 
                      metrics=['accuracy'])        

params = {
    'epochs': [2],
    'batch_size': [64],
    'learning_rate': [1e-3],
    'dropout': [0.1],
    'rnn_sizes': [[128, 128], [128, 128, 128]],
    'fc_sizes': [[64], [64, 64], [64, 32]],
    'batch_norm': [True, False]
}

dummy_params = {
    'epochs': [45],
    'batch_size': [128],
    'learning_rate': [1e-4],
    'dropout': [0.1],
    'rnn_sizes': [[128, 128, 128]],
    'fc_sizes': [[64, 32]],
    'batch_norm': [True]
}

model = CustomRNN()
search = GridSearchCV(estimator=model, 
                      param_grid=dummy_params,
                      n_jobs=1,
                      cv=5,
                      return_train_score=True, 
                      refit=False, 
                      verbose=10,
                      error_score='raise')
best = search.fit(X[:, :, :], Y[:])
print(best.__dict__)

