import pandas as pd
import numpy as np
import os
from datetime import datetime

from sklearn.model_selection import train_test_split
import sklearn.decomposition as decomposition
from sklearn.manifold import TSNE
from sklearn.metrics import make_scorer, f1_score, accuracy_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
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

df_1 = pd.read_csv("data/ptbdb_normal.csv", header=None)
df_2 = pd.read_csv("data/ptbdb_abnormal.csv", header=None)
df = pd.concat([df_1, df_2])

df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])

Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

n_class = 1

def build_gru(n_class=5, dropout=0.3, rnn_sizes = [128, 128], fc_sizes=[64], batch_norm=True):
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
            
    model.add(Dense(n_class, activation="sigmoid"))

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
        self.class_weight = compute_class_weight('balanced', [0, 1], train_y)
        early = EarlyStopping(monitor="val_loss", mode="min", patience=5, verbose=1)
        redonplat = ReduceLROnPlateau(monitor="val_loss", mode="min", patience=3, verbose=2)
        callbacks_list = [early, redonplat]
        self.build_model()

        self.model.fit(train_X, train_y, validation_split=0.1, 
                                         epochs=self.epochs, 
                                         batch_size=self.batch_size, 
                                         callbacks=callbacks_list)
    
    def predict(self, eval_X):
        return np.argmax(self.model.predict(eval_X), axis=1)
    
    def set_params(self, epochs=1, 
                         batch_size=64, 
                         learning_rate=1e-3, 
                         dropout=0.2, 
                         rnn_sizes=[128, 128],
                         fc_sizes=[64],
                         batch_norm=True,
                         balanced=True):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.rnn_sizes = rnn_sizes
        self.fc_sizes = fc_sizes
        self.batch_norm = batch_norm
        self.balanced = balanced
                
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
        self.model = build_gru(n_class=n_class, dropout=self.dropout, 
                          rnn_sizes=self.rnn_sizes, fc_sizes=self.fc_sizes, batch_norm=self.batch_norm)
        opt = optimizers.Adam(self.learning_rate)
        
        if self.balanced:
            self.model.compile(optimizer=opt, 
                      loss="binary_crossentropy", 
                      metrics=['accuracy'],
                      class_weight = self.class_weight)
        else:
            self.model.compile(optimizer=opt, 
                        loss="binary_crossentropy", 
                        metrics=['accuracy'])
        #self.model.summary()       

params = {
    'epochs': [2],
    'batch_size': [64],
    'learning_rate': [1e-3],
    'dropout': [0.1],
    'rnn_sizes': [[128, 128], [128, 128, 128]],
    'fc_sizes': [[64], [64, 64], [64, 32]],
    'batch_norm': [False],
    'balanced': [True]
}

dummy_params = {
    'epochs': [1],
    'batch_size': [32],
    'learning_rate': [1e-4],
    'dropout': [0.2],
    'rnn_sizes': [[8]],
    'fc_sizes': [[8]],
    'batch_norm': [False],
    'balanced': [True]
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
auc_roc = roc_auc_score(Y_test, predicted_y)
print("AUROC score : %s "% acc)
precision, recall, _ = precision_recall_curve(Y_test, predicted_y)
auc_prc = auc(recall, precision)
print("AUPRC score : %s "% auc_prc)
print(confusion_matrix(Y_test, predicted_y))

best_estimator.model.save(os.path.join(model_dir, str(datetime.now().strftime("%Y%m%d-%H%M%S"))))