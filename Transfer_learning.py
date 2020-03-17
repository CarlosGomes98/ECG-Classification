#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np

import tensorflow
from tensorflow.keras import optimizers, losses, activations, models, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Input, Dropout, GRU,     concatenate, Add, Activation
from tensorflow.keras.models import load_model
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split


# In[3]:


df_1 = pd.read_csv("data/ptbdb_normal.csv", header=None)
df_2 = pd.read_csv("data/ptbdb_abnormal.csv", header=None)
df = pd.concat([df_1, df_2])

df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])

Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]


# In[92]:


base_model = load_model("rnn/rnn_mitbih.h5")


# ### Remove dense layers and freeze remaining layers

# In[95]:


cropped_model = Model(base_model.input, base_model.layers[2].output)
for i in range(1, 4):
    print(cropped_model.layers[i])
    cropped_model.layers[i].trainable = False


# In[96]:


cropped_model.summary()


# In[97]:


seq_len = 187

def get_model():
    n_class = 1
    
    x = cropped_model.output
    dense = Dense(64, activation='relu')(x)
    dense = Dropout(0.2)(dense)
    output = Dense(n_class, activation='sigmoid')(dense)
    
    opt = tensorflow.keras.optimizers.Adam(lr=0.001)
    
    full_model = models.Model(inputs=cropped_model.input, outputs=output)
    class_weights = compute_class_weight('balanced', [0, 1], Y)

    full_model.compile(
        loss='binary_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )
    full_model.summary()
    return full_model


# In[98]:


full_model = get_model()
early = EarlyStopping(monitor="val_accuracy", mode="max", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_accuracy", mode="max", patience=3, verbose=2)
callbacks_list = [early, redonplat]  # early

full_model.fit(X, Y, epochs=1000, verbose=1, callbacks=callbacks_list, validation_split=0.1)


# ### Unfreeze layers and continue training

# In[ ]:


for i in range(1, 4):
    print(full_model.layers[i])
    full_model.layers[i].trainable = True


# In[ ]:


file_path = "rnn_transfer.h5"
checkpoint = ModelCheckpoint(file_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_accuracy", mode="max", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_accuracy", mode="max", patience=3, verbose=2)
callbacks_list = [checkpoint, early, redonplat]  # early

full_model.fit(X, Y, epochs=1000, verbose=1, callbacks=callbacks_list, validation_split=0.1)
full_model.load_weights(file_path)


# In[32]:


pred_test = full_model.predict(X_test)
pred_test = (pred_test>0.5).astype(np.int8)

f1 = f1_score(Y_test, pred_test)

print("Test f1 score : %s "% f1)

acc = accuracy_score(Y_test, pred_test)

print("Test accuracy score : %s "% acc)

auc_roc = roc_auc_score(Y_test, pred_test)

print("AUROC score : %s "% auc_roc)

precision, recall, _ = precision_recall_curve(Y_test, pred_test)

auc_prc = auc(recall, precision)
print("AUPRC score : %s "% auc_prc)

print(confusion_matrix(Y_test, pred_test))

