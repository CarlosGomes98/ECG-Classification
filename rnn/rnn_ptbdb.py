# Test f1 score : 0.8985821037563442
# Test accuracy score : 0.9828247761739448
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import sklearn.decomposition as decomposition
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau

import matplotlib.pyplot as plt
import seaborn as sb
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches


df_1 = pd.read_csv("data/ptbdb_normal.csv", header=None)
df_2 = pd.read_csv("data/ptbdb_abnormal.csv", header=None)
df = pd.concat([df_1, df_2])

df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])

Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

# sample_weight = compute_sample_weight('balanced', np.unique(Y), Y)
# print(np.unique(Y))
# print(sample_weight)
# sample_weights = np.array([sample_weight[i] for i in Y.tolist()])


X.shape, Y.shape

seq_len = 187

def get_model():
    n_class = 1
    model = Sequential()
    model.add(GRU(128, 
                   input_shape=(seq_len, 1),
                   return_sequences=True,
                   dropout=0.2))
    model.add(GRU(128,
                   return_sequences=True, 
                   dropout=0.2))
    model.add(GRU(128,
                   return_sequences=False, 
                   dropout=0.2))
    
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_class, activation='sigmoid'))
    
    opt = tf.keras.optimizers.Adam(lr=0.001)
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=opt,
        metrics=['accuracy'],
    )
    return model

model = get_model()
model.summary()

file_path = "rnn_ptbdb.h5"
checkpoint = ModelCheckpoint(file_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_accuracy", mode="max", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_accuracy", mode="max", patience=3, verbose=2)
callbacks_list = [checkpoint, early, redonplat]  # early

model.fit(X, Y,
          epochs=1000, 
          verbose=1,
          batch_size=64,
          validation_split=0.1,
         # sample_weight=sample_weights,
          callbacks=callbacks_list)

pred_test = model.predict(X_test)
pred_test = (pred_test>0.5).astype(np.int8)

f1 = f1_score(Y_test, pred_test, average="macro")

print("Test f1 score : %s "% f1)

acc = accuracy_score(Y_test, pred_test)

print("Test accuracy score : %s "% acc)

auc_roc = roc_auc_score(Y_test, pred_test)

print("AUROC score : %s "% acc)

precision, recall, _ = precision_recall_curve(Y_test, pred_test)

auc_prc = auc(recall, precision)
print("AUPRC score : %s "% auc_prc)

print(confusion_matrix(Y_test, pred_test))
