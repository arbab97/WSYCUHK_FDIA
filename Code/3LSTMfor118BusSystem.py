# -*- coding: utf-8 -*-
# This is the code for S. Wang, S. Bi and Y. A. Zhang, "Locational Detection of False Data Injection Attack in Smart Grid: a Multi-label Classification Approach," in IEEE Internet of Things Journal.
#Some scripts taken from : https://pythonprogramming.net/recurrent-neural-network-deep-learning-python-tensorflow-keras/
##Preprocessing: https://stats.stackexchange.com/questions/267012/difference-between-preprocessing-train-and-test-set-before-and-after-splitting
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, LSTM, Dropout, CuDNNLSTM
from keras import losses
from sklearn import preprocessing
from keras.optimizers import adam
import numpy as np
import time
#from keras import backend as K
import keras
from sklearn.metrics import f1_score
import pandas as pd
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
# data_dir="/content/data118_traintest.mat"#"/media/rabi/Data/11111/openuae/datafromdrive/data118_1.mat"
data_dir="/media/rabi/Data/11111/openuae/datafromdrive/data14_2.mat"
output_dir="/media/rabi/Data/11111/openuae/WSYCUHK_FDIA_results/"

x_train = sio.loadmat(data_dir)['x_train']
y_train= sio.loadmat(data_dir)['y_train']
x_test = sio.loadmat(data_dir)['x_test']
y_test = sio.loadmat(data_dir)['y_test']

norm = preprocessing.StandardScaler().fit(x_train)
x_train = norm.transform(x_train) 
x_test = norm.transform(x_test) #Preventing the x_test stats from leaking



all_results=pd.DataFrame(columns={
"Number of LSTM Units",
"Row Accuracy", 
"Test Accuracy",
"Training Accuracy",
"Validation Accuracy",
"Number of Parameters",
"Time Taken",
"F1 Score"}) 

Epochs=10
for units in [128, 64, 32, 16]:
    #LSTM model
    model = Sequential()
    shape=19 #180
    model.add(LSTM(units, input_shape=(shape,1), return_sequences=False))
    # model.add(CuDNNLSTM(128, input_shape=(180,1), return_sequences=False)) Colab Equivalent
    # model.add(LSTM(16, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(shape, activation='sigmoid'))
    # =============================================================================
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    # Train, evaluate, predict
    reduce_lr=keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    start_time=time.time()
    history=(model.fit(np.expand_dims(x_train,axis=2), y_train, batch_size=100, epochs=Epochs,callbacks=[reduce_lr],
    validation_split=0.3
    )) #default epoch 200
    end_time=time.time()
    score = model.evaluate(np.expand_dims(x_test,axis=2), y_test, batch_size=100)
    pred_y=model.predict(np.expand_dims(x_test,axis=2), batch_size=100)

    #Save the result
    sio.savemat(output_dir+"output_LSTM_"+str(units), {'output_mode':pred_y,'output_mode_pred': y_test})


    # The threshold can be changed to generate ROC curve, in this file, the threshold is set as 0.5
    for i in range(10000): #(2000)
        for j in range (shape):
            if pred_y[i][j]>0.5:
                pred_y[i][j]=1
            else:
                pred_y[i][j]=0
    row,acca=cal_acc(pred_y,y_test)
    print("Test Row Accuracy: ", row)
    print("Test individual accuracy: ", acca)


    model_stats=pd.DataFrame({
    "Training Loss":history.history['loss'],
    "Validation Loss":history.history['val_loss'],
    "Epoch":range(len(history.history['loss']))},
    index=range(len(history.history['loss'])))
    #Saving model stats (since they are being saved for individual model)
    model_stats.to_csv(output_dir+"LSTM_"+str(units)+".csv")


    single_result={
    "Number of LSTM Units": units,
    "Row Accuracy": row, 
    "Test Accuracy": acca,
    "Training Accuracy": history.history['accuracy'][-1],
    "Validation Accuracy": history.history['val_accuracy'][-1],
    "Number of Parameters": model.count_params(),
    "Time Taken": end_time-start_time,
    "F1 Score": f1_score(y_test, pred_y, average='micro')  #I think weighted has something to do with evaluation
    }
    all_results=all_results.append(single_result, ignore_index=True)

    keras.backend.clear_session() #destroying the old model

#saving results outside the loop since they are stored for all model

all_results.to_csv(output_dir+"All LSTM Models Commulative.csv")

