# -*- coding: utf-8 -*-
# This is the code for S. Wang, S. Bi and Y. A. Zhang, "Locational Detection of False Data Injection Attack in Smart Grid: a Multi-label Classification Approach," in IEEE Internet of Things Journal.
#Some scripts taken from : https://pythonprogramming.net/recurrent-neural-network-deep-learning-python-tensorflow-keras/
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, LSTM, Dropout, CuDNNLSTM
from keras import losses
from sklearn import preprocessing
from keras.optimizers import adam
import numpy as np

#from keras import backend as K
import keras


def test2train(save_name):
    y_pred = sio.loadmat(save_name)['output_mode_pred']
    return y_pred
    # If you want to save the result.
    # sio.savemat(save_name, {'input_h': X_test/10000000,'output_mode':Y_test,'output_mode_pred': y_pred})
        
           
def cal_acc(a,b):
    n=a.shape[0]
    m=a.shape[1]
    tterr=0
    r_err=0
    for i in range(n):
        cuerr=0
        for j in range(m):
            if a[i][j]!= b[i][j]:
               tterr+=1
               cuerr+=1
        if cuerr>0:
            r_err+=1
            
    return 1-r_err/n, 1-tterr/(n*m)

def weight_loss(a,b):#Self-defined loss function to handle the unbalance labels
    import tensorflow as tf
    mask_a=tf.greater_equal(a,0.5)
    mask_b=tf.less(a,0.5)
    return (5*losses.binary_crossentropy(tf.boolean_mask(a,mask_a),tf.boolean_mask(b,mask_a))+losses.binary_crossentropy(tf.boolean_mask(a,mask_b),tf.boolean_mask(b,mask_b)))/6


import scipy.io as sio 
# Load data
data_dir="/content/data118_traintest.mat"#"/media/rabi/Data/11111/openuae/datafromdrive/data118_1.mat"


x_train = sio.loadmat(data_dir)['x_train']
y_train= sio.loadmat(data_dir)['y_train']
x_test = sio.loadmat(data_dir)['x_test']
y_test = sio.loadmat(data_dir)['y_test']

x_train = preprocessing.scale(x_train)
x_test = preprocessing.scale(x_test)

# Define the network struture
# #In this example, the network is 4 layers 1DCNN + 1 Flatten Layer + 1 Fully Connected Layer
# model = Sequential()
# model.add(Conv1D(128, 5, activation='relu', input_shape=(x_train.shape[1], 1)))
# #model.add(Dropout(0.05))
# #model.add(Conv1D(256, 3, activation='relu'))
# #model.add(Conv1D(256, 3, activation='relu'))
# #model.add(MaxPooling1D(3))
# #model.add(Conv1D(128, 3, activation='relu'))
# #model.add(Conv1D(128, 3, activation='relu'))
# model.add(Conv1D(256, 3, activation='relu'))
# model.add(Conv1D(128, 3, activation='relu'))
# model.add(Conv1D(128, 3, activation='relu'))
# model.add(Flatten())
# model.add(Dense(180, activation='sigmoid'))

#LSTM model
model = Sequential()


model.add(CuDNNLSTM(128, input_shape=(180,1), return_sequences=False))
model.add(Dropout(0.2))

# model.add(CuDNNLSTM(256))
# model.add(Dropout(0.1))

# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))

model.add(Dense(180, activation='sigmoid'))



# import tensorflow as tf
# opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
opt = adam(lr=0.01, decay=1e-6)


# Compile model
model.compile(
    loss='binary_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)
model.fit(np.expand_dims(x_train,axis=2),
          y_train,
          epochs=3,
          validation_data=(np.expand_dims(x_test,axis=2), y_test))

import sys
sys.exit()
# Choose the loss function
# =============================================================================
# model.compile(loss=weight_loss,
#               optimizer='adam',
#               metrics=['accuracy'])
# =============================================================================
model.compile(loss='binary_crossentropy',
               optimizer='adam',
              metrics=['accuracy'])




# Train, evaluate, predict
import numpy as np
reduce_lr=keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
model.fit(np.expand_dims(x_train,axis=2), y_train, batch_size=100, epochs=2,callbacks=[reduce_lr]) #default epoch 200
score = model.evaluate(np.expand_dims(x_test,axis=2), y_test, batch_size=100)
pred_y=model.predict(np.expand_dims(x_test,axis=2), batch_size=100)

# The threshold can be changed to generate ROC curve, in this file, the threshold is set as 0.5
for i in range(10000): #(2000)
    for j in range (180):
        if pred_y[i][j]>0.5:
            pred_y[i][j]=1
        else:
            pred_y[i][j]=0
row,acca=cal_acc(pred_y,y_test)
print("Test Row Accuracy: ", row)
print("Test individual accuracy: ", acca)
#Save the result
sio.savemat('./118caseresult_weighted_test', {'output_mode':pred_y,'output_mode_pred': y_test})
