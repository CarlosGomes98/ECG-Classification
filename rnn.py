import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import sklearn.decomposition as decomposition
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, accuracy_score

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import load_model

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

X.shape, Y.shape

seq_len = 187

def get_model():
    n_class = 5
    model = Sequential()
    model.add(LSTM(128, 
                   input_shape=(seq_len, 1), 
                   activation='relu', 
                   return_sequences=True,
                   dropout=0.2))
    model.add(LSTM(128, 
                   activation='relu',
                   return_sequences=False, 
                   dropout=0.1))
    
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_class, activation='softmax'))
    
    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
    
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'],
    )
    return model

model = get_model()
model.summary()

model.fit(X[:10000 :, :], Y[:10000], 
          epochs=5, 
          verbose=1,
          batch_size=32,
          validation_split=0.1)

pred_test = model.predict(X_test)
pred_test = np.argmax(pred_test, axis=-1)

f1 = f1_score(Y_test, pred_test, average="macro")

print("Test f1 score : %s "% f1)

acc = accuracy_score(Y_test, pred_test)

print("Test accuracy score : %s "% acc)

