import pandas as pd
import numpy as np

import tensorflow
from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D,     concatenate, Add, Activation
from tensorflow.keras.models import load_model

from sklearn.metrics import f1_score, accuracy_score


# ### MITBIH baseline
# #### Code mostly from https://github.com/CVxTz/ECG_Heartbeat_Classification/blob/master/code/baseline_mitbih.py

# In[4]:


df_train = pd.read_csv("data/mitbih_train.csv", header=None)
df_train = df_train.sample(frac=1)
df_test = pd.read_csv("data/mitbih_test.csv", header=None)

Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]


# ### Load model

# In[5]:


# model = load_model("baseline_cnn_res_mitbih.h5")


# ### Or ALTERNATIVELY train it

# In[2]:


def res_block(X, filters, kernel_size=5, dropout=0.1, pool_size=2):
    img_1 = Convolution1D(filters, kernel_size=kernel_size, activation='relu', padding='same')(X)
    img_1 = Convolution1D(filters, kernel_size=kernel_size, activation='relu', padding='same')(img_1)
    img_1 = Add()([X, img_1])
    img_1 = Activation('relu')(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    return img_1


# In[8]:


def get_model():
    nclass = 5
    inp = Input(shape=(187, 1))
    img_1 = Convolution1D(32, kernel_size=5, activation=activations.relu, padding="valid")(inp)
    for i in range(5):
        img_1 = res_block(img_1, 32, dropout=0.2)
    img_1 = GlobalMaxPool1D()(img_1)

    dense_1 = Dense(32, activation=activations.relu, name="dense_1")(img_1)
    dense_1 = Dense(32, activation=activations.relu, name="dense_2")(dense_1)
    dense_1 = Dense(nclass, activation=activations.softmax, name="dense_3_mitbih")(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model


# In[9]:


model = get_model()
file_path = "cnn_res_mitbih.h5"
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

print(tensorflow.math.confusion_matrix(
    Y_test, pred_test, num_classes=5))

