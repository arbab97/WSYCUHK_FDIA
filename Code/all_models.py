# -*- coding: utf-8 -*-
# This is the code for S. Wang, S. Bi and Y. A. Zhang, "Locational Detection of False Data Injection Attack in Smart Grid: a Multi-label Classification Approach," in IEEE Internet of Things Journal.
#Some scripts taken from : https://pythonprogramming.net/recurrent-neural-network-deep-learning-python-tensorflow-keras/
##Preprocessing: https://stats.stackexchange.com/questions/267012/difference-between-preprocessing-train-and-test-set-before-and-after-splitting
#https://pypi.org/project/keras-self-attention/
#https://github.com/tensorflow/tensorflow/issues/40911
#Example: https://stackoverflow.com/questions/56946995/how-to-build-a-attention-model-with-keras
#https://matthewmcateer.me/blog/getting-started-with-attention-for-classification/    #!!
#https://stackoverflow.com/questions/59811773/how-to-use-keras-attention-layer-on-top-of-lstm-gru   #example of encoder-decoder classifier #!!
#https://stackoverflow.com/questions/63060083/create-an-lstm-layer-with-attention-in-keras-for-multi-label-text-classification/64853996#64853996 #simple multi-label classificaiton with attention

#https://stackoverflow.com/questions/62948332/how-to-add-attention-layer-to-a-bi-lstm/62949137#62949137 #Better version. of the ABOVE #!!!!!
#https://machinelearningmastery.com/cnn-long-short-term-memory-networks/
#https://towardsdatascience.com/cnn-lstm-predicting-daily-hotel-cancellations-e1c75697f124
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, LSTM, Dropout, Bidirectional, TimeDistributed, CuDNNLSTM
# from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from keras import losses
from sklearn import preprocessing
from keras.optimizers import adam
import numpy as np
import time
from keras import backend as K
import keras
from sklearn.metrics import f1_score
import pandas as pd
from keras.layers import Layer
from keras import backend as K
import scipy.io as sio 
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--model',type=str, help='Model Type', required=True)
parser.add_argument('--data_dir', type=str, help='Input data Directory', required=True)
parser.add_argument('--output_dir', type=str, help='Where to store the results', required=True)
parser.add_argument('--n_epoch', type=int, help='number of epoches when maximizing', required=True)
parser.add_argument('--layers', type=int, help='number of Layers in network', required=True)
parser.add_argument('--neurons', type=int, help='number of units in each layer', required=True)
parser.add_argument('--shape', type=int, help='specify shape according to IEEE 14(19) OR IEEE 118(180)', required=True)

args = parser.parse_args()
print("You've selected: ", args.model)
print("With Number of Epochs: ", str(args.n_epoch))




# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import os
if 'COLAB_GPU' in os.environ:
  print("Running on Colab -------------")
  chosen_lstm=CuDNNLSTM
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
else:
  print("NOT RUNNING ON COLAB")
  chosen_lstm=LSTM
  import tensorflow as tf

class Attention(Layer):

    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(Attention,self).__init__()
        
    def build(self, input_shape):
        
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="zeros")
        
        super(Attention,self).build(input_shape)
        
    def call(self, x):
        
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        
        if self.return_sequences:
            return output
        
        return K.sum(output, axis=1)


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

    mask_a=tf.greater_equal(a,0.5)
    mask_b=tf.less(a,0.5)
    return (5*losses.binary_crossentropy(tf.boolean_mask(a,mask_a),tf.boolean_mask(b,mask_a))+losses.binary_crossentropy(tf.boolean_mask(a,mask_b),tf.boolean_mask(b,mask_b)))/6

def row_accuracy(y_true, y_pred):
    y_pred = K.round(y_pred)
    acc = K.all(K.equal(y_true, y_pred), axis=1)
    acc= K.cast(acc, 'float32')
    acc = K.sum(acc)
    acc = acc/K.cast(K.shape(y_true)[0], 'float32')
    return acc


# Load data
# data_dir="/content/data118_traintest.mat"#"/media/rabi/Data/11111/openuae/datafromdrive/data118_1.mat"
data_dir=args.data_dir#"/content/data118_traintest.mat"
output_dir=args.output_dir#"/content/"

x_train = sio.loadmat(data_dir)['x_train']
y_train= sio.loadmat(data_dir)['y_train']
x_test = sio.loadmat(data_dir)['x_test']
y_test = sio.loadmat(data_dir)['y_test']

norm = preprocessing.StandardScaler().fit(x_train)
x_train = norm.transform(x_train) 
x_test = norm.transform(x_test) #Preventing the x_test stats from leaking



all_results=pd.DataFrame(columns={
"Model",
"Number of Units in a layer",
"Row Accuracy", 
"Test Accuracy",
"Training Accuracy",
"Validation Accuracy",
"Number of Parameters",
"Time Taken",
"F1 Score"}) 



#attention based return_sequences=T/F
model = Sequential()

if args.model=="MLP":  #DNN
    for layer in range(args.layers):
        model.add(Dense(args.neurons,  activation='relu', input_shape=(args.shape,1) ))
        model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(args.shape, activation='sigmoid'))


elif args.model=="CNN":
    for layer in range(args.layers):
        model.add(Conv1D(args.neurons, 5, activation='relu', input_shape=(args.shape,1)))
        model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(args.shape, activation='sigmoid'))

elif args.model=="LSTM": 
    for layer in range(args.layers):
        model.add(chosen_lstm(args.neurons, input_shape=(args.shape,1), return_sequences=True)) #Colab Equivalent
        model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(args.shape, activation='sigmoid'))

elif args.model=="Attention":   
    for layer in range(args.layers):
        model.add(Bidirectional(chosen_lstm(args.neurons, return_sequences=True, input_shape=(args.shape,1))))
        model.add(Attention(return_sequences=True)) # receive 3D and output 2D
        model.add(Dropout(0.2))

    # model.add(Attention(return_sequences=True)) # receive 3D and output 2D
    model.add(Flatten())
    model.add(Dense(args.shape, activation='sigmoid'))

elif args.model=="cnn-lstm-paper-experiments":   

    model = Sequential()
    for layer in range(args.layers):
        # define CNN model
        model.add((Conv1D(args.neurons, 5, activation='relu', input_shape=(args.shape,1))))
        model.add(Dense(args.neurons, activation='relu'))
        # define LSTM model
        model.add(chosen_lstm(args.neurons, return_sequences=True)) 
        model.add(Dropout(0.2))

    model.add(Flatten())   
    model.add(Dense(args.shape, activation='sigmoid'))


elif args.model=="cnn-lstm-paper-original":   

    model = Sequential()
    # define CNN model
    model.add((Conv1D(args.neurons, 5, activation='relu', input_shape=(args.shape,1))))
    model.add((Conv1D(args.neurons, 5, activation='relu')))
    model.add((Conv1D(args.neurons, 5, activation='relu')))
    model.add((Conv1D(args.neurons, 5, activation='relu')))
    model.add(Dense(args.shape, activation='relu'))    #!!!!!!!!CHANGE .shape to neurons (DOUBLE CHECK)
    # define LSTM model
    model.add(chosen_lstm(args.neurons, return_sequences=True)) 
    model.add(chosen_lstm(args.neurons, return_sequences=True))
    model.add(Flatten())   
    # model.add(CuDNNLSTM(128, return_sequences=False))   #Colab Equivalent
    model.add(Dense(args.shape, activation='relu'))
    model.add(Dense(args.shape, activation='relu'))
    model.add(Dense(args.shape, activation='sigmoid'))


else:
    print("Please enter the correct mdoel")
    exit()

# =============================================================================
model.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy', row_accuracy])

# Train, evaluate, predict
reduce_lr=keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
start_time=time.time()
history=(model.fit(np.expand_dims(x_train,axis=2), y_train, batch_size=100, epochs=args.n_epoch,callbacks=[reduce_lr],
validation_split=0.3
)) #default epoch 200
end_time=time.time()
score = model.evaluate(np.expand_dims(x_test,axis=2), y_test, batch_size=100)
pred_y=model.predict(np.expand_dims(x_test,axis=2), batch_size=100)


# The threshold can be changed to generate ROC curve, in this file, the threshold is set as 0.5
for i in range(x_test.shape[0]): #(2000)
    for j in range (args.shape):
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
"Epoch":range(1,len(history.history['loss'])+1)},
index=range(len(history.history['loss'])))
#Saving model stats (since they are being saved for individual model)


single_result={
"Model": args.model,
"Number of Units in a layer": args.neurons,
"Row Accuracy": row, 
"Test Accuracy": acca,
"Training Accuracy": history.history['accuracy'][-1],
"Validation Accuracy": history.history['val_accuracy'][-1],
"Number of Parameters": model.count_params(),
"Time Taken": end_time-start_time,
"F1 Score": f1_score(y_test, pred_y, average='micro')  #I think weighted has something to do with evaluation
}
all_results=all_results.append(single_result, ignore_index=True)

# keras.backend.clear_session() #destroying the old model

save_id=args.model+"_"+str(args.layers)+"_"+str(args.neurons)
#Save the learning curve stats
model_stats.to_csv(output_dir+"stats_"+(save_id)+".csv")
#Save the results
all_results.to_csv(output_dir+"results_"+(save_id)+".csv")
#Save the result
sio.savemat(output_dir+"output_"+(save_id), {'output_mode':pred_y,'output_mode_pred': y_test})
#Save the trained model
model.save(output_dir+"model_"+(save_id))
