import scipy.io as sio 
from sklearn.metrics import f1_score
path="/media/rabi/Data/11111/openuae/WSYCUHK_FDIA_results2/output_LSTM_128"
pred_y= sio.loadmat(path)['output_mode']
y_test= sio.loadmat(path)['output_mode_pred'] #ignore the naming; its messed up
f1=f1_score(y_test, pred_y>0.5, average='micro')  #I think weighted has something to do with evaluation

print(f1)
