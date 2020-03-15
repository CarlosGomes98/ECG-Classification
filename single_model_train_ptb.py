import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

from sklearn.model_selection import train_test_split
import sklearn.decomposition as decomposition
from sklearn.manifold import TSNE
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix, make_scorer

import tensorflow as tf
from tensorflow.keras import Model, Input, optimizers, losses
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM, GRU, BatchNormalization, Bidirectional
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, TensorBoard

df_1 = pd.read_csv("data/ptbdb_normal.csv", header=None)
df_2 = pd.read_csv("data/ptbdb_abnormal.csv", header=None)
df = pd.concat([df_1, df_2])

df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])

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

def build_bilstm(n_class=5, dropout=0.3, rnn_sizes = [128, 128], fc_sizes=[64], batch_norm=True):
    nclass = 5
    model = Sequential()
    model.add(Input(shape=(187, 1)))
    
    if batch_norm:
        model.add(BatchNormalization())
        
    for index, dim in enumerate(rnn_sizes):
        model.add(Bidirectional(GRU(dim, dropout=dropout, return_sequences=(index != len(rnn_sizes) - 1))))
        
        if batch_norm:
            model.add(BatchNormalization())
    
    for index, dim in enumerate(fc_sizes):
        model.add(Dense(dim, activation="relu"))
        model.add(Dropout(dropout))
        
        if batch_norm:
            model.add(BatchNormalization())
            
    model.add(Dense(nclass, activation="softmax"))

    return model

if sys.argv[1] == 'gru':
	model = build_gru(n_class=n_class, 
					  dropout=0.2, 
					  rnn_sizes=[128, 128, 128], 
					  fc_sizes=[64, 64], 
					  batch_norm=True)
elif sys.argv[1] == 'bilstm':
	model = build_bilstm(n_class=n_class, 
					  	 dropout=0.2, 
					  	 rnn_sizes=[128, 128], 
					  	 fc_sizes=[64], 
					  	 batch_norm=True)
else:
	print("Invalid argument")
	sys.exit()

opt = optimizers.Adam(1e-4)

early = EarlyStopping(monitor="val_loss", mode="min", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_loss", mode="min", patience=3, verbose=2)
model.compile(optimizer=opt, 
	          loss=losses.binary_crossentropy, 
	          metrics=['accuracy'])

logdir = os.path.join("logs", "scalars", str(datetime.now().strftime("%Y%m%d-%H%M%S")))
if not os.path.exists(logdir):
	os.makedirs(logdir)
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()
tensorboard_callback = TensorBoard(log_dir=logdir)
early = EarlyStopping(monitor="val_loss", mode="min", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_loss", mode="min", patience=3, verbose=2)

model.fit(X_train[:, :, :], y_train[:], 
		  validation_data=(X_eval, y_eval),
		  epochs=50, 
		  batch_size=64,
		  callbacks=[tensorboard_callback,
                     early,
                     redonplat])

predicted_y = np.argmax(model.predict(X_test), axis=1)
print("TEST EVALUATION")
print("F1-SCORE: ", f1_score(Y_test, predicted_y, average='macro'))
print("ACCURACY: ", accuracy_score(Y_test, predicted_y))
auc_roc = roc_auc_score(Y_test, predicted_y)
print("AUROC score : %s "% acc)
precision, recall, _ = precision_recall_curve(Y_test, predicted_y)
auc_prc = auc(recall, precision)
print("AUPRC score : %s "% auc_prc)
print(confusion_matrix(Y_test, predicted_y))
model_dir = 'models'
if not os.path.exists(model_dir):
  os.makedirs(model_dir)
model.save(os.path.join(model_dir, str(datetime.now().strftime("%Y%m%d-%H%M%S"))))
