import pandas as pd
import numpy as np
import os
import sys
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
from tensorflow.keras.layers import LSTM, GRU, BatchNormalization, Bidirectional
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, TensorBoard

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

print("Distribution of classes in train set")
print(np.unique(y_train, return_counts=True))
print("Distribution of classes in evaluation set")
print(np.unique(y_eval, return_counts=True))

n_class = np.unique(Y).size

def build_gru(n_class=5, dropout=0.3, rnn_sizes = [128, 128], fc_sizes=[64], batch_norm=True):
    print("Building model...")
    print("Rnn sizes: ", rnn_sizes)
    print("Fc sizes: ", fc_sizes)
    
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


class F1_Metric(tf.keras.callbacks.Callback):

	 def on_epoch_end(self, epoch, logs={}):
	    predicted_y = np.argmax(self.model.predict(X_eval), axis=1)
	    score = f1_score(predicted_y, y_eval, average='macro')
	    tf.summary.scalar('f1_score', data=score, step=epoch)

	    # TODO: automatically find dominant class instead of hard-coding 0
	    print("\n", np.unique(predicted_y, return_counts=True))
	    num_predicted_dominant_class = np.sum(predicted_y == 0)
	    print("\nNum predicted dominant class: ", num_predicted_dominant_class)
	    tf.summary.scalar('Number of samples predicted as dominant class', 
	    				  data=num_predicted_dominant_class,
	    				  step=epoch)
	    print("\nf1_score: ", score)

if sys.argv[1] == 'gru':
	model = build_gru(n_class=5, 
					  dropout=0.2, 
					  rnn_sizes=[128, 128, 128], 
					  fc_sizes=[64], 
					  batch_norm=False)
elif sys.argv[1] == 'bilstm':
	model = build_bilstm(n_class=5, 
					  	 dropout=0.2, 
					  	 rnn_sizes=[256, 256], 
					  	 fc_sizes=[64, 32], 
					  	 batch_norm=True)
else:
	print("Invalid argument")
	sys.exit()

opt = optimizers.Adam(1e-4)

model.compile(optimizer=opt, 
		           loss="sparse_categorical_crossentropy", 
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
		  epochs=7, 
		  batch_size=32,
		  callbacks=[F1_Metric(), 
                     tensorboard_callback,
                     early,
                     redonplat])


print("TEST EVALUATION")
predicted_y = np.argmax(model.predict(X_test), axis=1)
print("F1-SCORE: ", f1_score(Y_test, predicted_y, average='macro'))
print(np.unique(predicted_y))
print(np.unique(Y_test))
print("ACCURACY: ", accuracy_score(Y_test, predicted_y))
print(confusion_matrix(Y_test, predicted_y))

model_dir = 'models'
if not os.path.exists(model_dir):
  os.makedirs(model_dir)
model.save(os.path.join(model_dir, str(datetime.now().strftime("%Y%m%d-%H%M%S"))))
