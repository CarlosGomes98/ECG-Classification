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
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import Model, Input, optimizers, losses
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM, GRU, BatchNormalization, Bidirectional
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.utils import plot_model

df_1 = pd.read_csv("../data/ptbdb_normal.csv", header=None)
df_2 = pd.read_csv("../data/ptbdb_abnormal.csv", header=None)
df = pd.concat([df_1, df_2])

df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])

Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

X_train, X_eval, y_train, y_eval = train_test_split(X, Y,
                                                    stratify=Y, 
                                                    test_size=0.1)

print(X_train.shape, X_eval.shape)
print(np.unique(y_train, return_counts=True))
print(np.unique(y_eval, return_counts=True))
print(np.unique(Y_test, return_counts=True))
n_class = 1

def build_gru(n_class=2, dropout=0.3, rnn_sizes = [128, 128], fc_sizes=[64], batch_norm=True):
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

def build_bilstm(n_class=2, dropout=0.3, rnn_sizes = [128, 128], fc_sizes=[64], batch_norm=True):
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
            
    model.add(Dense(n_class, activation="sigmoid"))

    return model

if sys.argv[1] == 'gru':
	model = build_gru(n_class=n_class, 
					  dropout=0.0, 
					  rnn_sizes=[64], 
					  fc_sizes=[32], 
					  batch_norm=False)
elif sys.argv[1] == 'bilstm':
	model = build_bilstm(n_class=n_class, 
					  	 dropout=0.2, 
					  	 rnn_sizes=[256, 256], 
					  	 fc_sizes=[64], 
					  	 batch_norm=True)
else:
	print("Invalid argument")
	sys.exit()

opt = optimizers.Adam(1e-4)

model.compile(optimizer=opt, 
          loss='binary_crossentropy', 
	          metrics=['accuracy'])
print("Plotting model")
plot_model(model, to_file="bilstm_ptbdb.png", show_shapes=True)

logdir = os.path.join("logs", "scalars", str(datetime.now().strftime("%Y%m%d-%H%M%S")))
if not os.path.exists(logdir):
	os.makedirs(logdir)
#file_writer = tf.summary.create_file_writer(logdir + "/metrics")
#file_writer.set_as_default()
#tensorboard_callback = TensorBoard(log_dir=logdir)
early = EarlyStopping(monitor="loss", mode="min", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="loss", mode="min", patience=3, verbose=2)

class_weight = compute_class_weight('balanced', [0, 1], y_train)
print("Class weight: ", class_weight)

model.fit(X_train[:, :, :], y_train[:], 
		  validation_data=(X_eval, y_eval),
		  epochs=100, 
		  batch_size=32,
		  callbacks=[early,
                     redonplat],
         class_weight=class_weight)

softmax_prediction = model.predict(X_test)
predicted_y = np.array([softmax_prediction > 0.5]).astype(int).flatten()
print(predicted_y.shape)
print(Y_test.shape)
print("TEST EVALUATION")
print("F1-SCORE: ", f1_score(Y_test, predicted_y))
print("ACCURACY: ", accuracy_score(Y_test, predicted_y))
acc = roc_auc_score(Y_test, predicted_y)
print("AUROC score : %s "% acc)
precision, recall, _ = precision_recall_curve(Y_test, predicted_y)
auc_prc = auc(recall, precision)
print("AUPRC score : %s "% auc_prc)
print(confusion_matrix(Y_test, predicted_y))
model_dir = 'models'
if not os.path.exists(model_dir):
  os.makedirs(model_dir)
model.save(os.path.join(model_dir, str(datetime.now().strftime("%Y%m%d-%H%M%S"))))
