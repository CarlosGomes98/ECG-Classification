import pandas as pd
import numpy as np

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

df_train = pd.read_csv("data/mitbih_train.csv", header=None)
df_train = df_train.sample(frac=1)
df_test = pd.read_csv("data/mitbih_test.csv", header=None)

Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

X_train, X_eval, y_train, y_eval = train_test_split(X, Y,
                                                    stratify=Y, 
                                                    test_size=0.15)

print(X_train.shape, X_eval.shape)

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

class F1_Metric(tf.keras.callbacks.Callback):

	 def on_epoch_end(self, batch, logs={}):
	    predicted_y = np.argmax(self.model.predict(X_eval), axis=1)
	    score = f1_score(predicted_y, y_eval, average='macro')
	    print("f1_score: ", score)

model = build_gru(n_class=5, 
				  dropout=0.2, 
				  rnn_sizes=[128, 128, 128], 
				  fc_sizes=[64, 64], 
				  batch_norm=True)
opt = optimizers.Adam(1e-4)

model.compile(optimizer=opt, 
		           loss="sparse_categorical_crossentropy", 
		           metrics=['accuracy'])

model.fit(X_train[:, :, :], y_train[:], 
		  validation_data=(X_eval, y_eval),
		  epochs=100, 
		  batch_size=128,
		  callbacks=[F1_Metric()])
