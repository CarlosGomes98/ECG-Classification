# Test f1 score : 0.8616716371097233 
# Test accuracy score : 0.9730038370180888
import pandas as pd
import numpy as np

from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate, LSTM
from sklearn.metrics import f1_score, accuracy_score


df_train = pd.read_csv("data/mitbih_train.csv", header=None)
df_train = df_train.sample(frac=1)
df_test = pd.read_csv("data/mitbih_test.csv", header=None)

Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

sample_weight = compute_sample_weight('balanced', np.unique(Y), Y)
sample_weights = np.array([sample_weight[i] for i in Y.tolist()])

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

print(X.shape, Y.shape)
def get_model():
    nclass = 5
    inp = Input(shape=(187, 1))
    img_1 = LSTM(128, dropout=0.2, return_sequences=False)(inp)
    # img_1 = LSTM(128, dropout=0.2)(img_1)

    dense_1 = Dense(nclass, activation=activations.sigmoid, name="dense_3_ptbdb")(img_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(lr=0.001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model

model = get_model()
file_path = "rnn_mitbih.h5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
callbacks_list = [checkpoint, early, redonplat]  # early

model.fit(X, Y, epochs=1000, verbose=2, callbacks=callbacks_list, validation_split=0.1, sample_weight=sample_weights)
model.load_weights(file_path)

pred_test = model.predict(X_test)
pred_test = np.argmax(pred_test, axis=-1)

f1 = f1_score(Y_test, pred_test, average="macro")

print("Test f1 score : %s "% f1)

acc = accuracy_score(Y_test, pred_test)

print("Test accuracy score : %s "% acc)