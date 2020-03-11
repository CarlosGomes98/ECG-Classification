import pandas as pd
import numpy as np

import tensorflow
from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D,     concatenate, Add, Activation
from tensorflow.keras.models import load_model

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix


# ### MITBIH baseline
# #### Code mostly from https://github.com/CVxTz/ECG_Heartbeat_Classification/blob/master/code/baseline_mitbih.py

# In[4]:


df_1 = pd.read_csv("data/ptbdb_normal.csv", header=None)
df_2 = pd.read_csv("data/ptbdb_abnormal.csv", header=None)
df = pd.concat([df_1, df_2])

df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])

Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]



# ### Load model

# In[5]:


#model = load_model("baseline_cnn_res_mitbih.h5")


# ### Or ALTERNATIVELY train it

# In[16]:


def maxpool_block(X, filters, kernel_size=5, dropout=0.1, pool_size=2):
    img_1 = Convolution1D(filters, kernel_size=kernel_size, activation='relu', padding='same')(X)
    img_1 = Convolution1D(filters, kernel_size=kernel_size, activation='relu', padding='same')(img_1)
    img_1 = Convolution1D(filters, kernel_size=kernel_size, activation='relu', padding='same')(img_1)
    shortcut = Convolution1D(filters, kernel_size=kernel_size, activation='relu', padding='same')(X)
    img_1 = Add()([shortcut, img_1])
    img_1 = Activation('relu')(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    return img_1


# In[17]:


def identity_block(X, filters, kernel_size=5, dropout=0.1, pool_size=2):
    img_1 = Convolution1D(filters, kernel_size=kernel_size, activation='relu', padding='same')(X)
    img_1 = Convolution1D(filters, kernel_size=kernel_size, activation='relu', padding='same')(img_1)
    img_1 = Convolution1D(filters, kernel_size=kernel_size, activation='relu', padding='same')(img_1)
    img_1 = Add()([X, img_1])
    img_1 = Activation('relu')(img_1)
    return img_1


# In[18]:


def get_model():
    nclass = 1
    filters = [32, 32, 64, 128, 128]
    inp = Input(shape=(187, 1))
    img_1 = Convolution1D(32, kernel_size=5, activation=activations.relu, padding="valid")(inp)     
    for i in range(5):
        img_1 = maxpool_block(img_1, filters[i], dropout=0.2)
        img_1 = identity_block(img_1, filters[i], dropout=0.2) 
        img_1 = identity_block(img_1, filters[i], dropout=0.2) 
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="same")(img_1)  
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="same", name="final_conv")(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)

    dense_1 = Dense(32, activation=activations.relu, name="dense_1")(img_1)
    dense_1 = Dense(32, activation=activations.relu, name="dense_2")(dense_1)
    dense_1 = Dense(nclass, activation=activations.sigmoid, name="dense_3_mitbih")(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['acc'])
    model.summary()
    return model


# In[19]:


model = get_model()
file_path = "cnn_res_deep_mitbih.h5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
callbacks_list = [checkpoint, early, redonplat]  # early

model.fit(X, Y, epochs=1000, verbose=2, callbacks=callbacks_list, validation_split=0.1)
model.load_weights(file_path)


# ### Evaluation

# In[17]:


pred_test = model.predict(X_test)
pred_test = np.argmax(pred_test, axis=-1)

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

