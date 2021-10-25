import scipy.io as sio 
# Load data
# data_dir="/content/data118_traintest.mat"#"/media/rabi/Data/11111/openuae/datafromdrive/data118_1.mat"
data_dir="/media/rabi/Data/11111/openuae/datafromdrive/data14_2.mat"
x_train = sio.loadmat(data_dir)['x_train']
y_train= sio.loadmat(data_dir)['y_train']
x_test = sio.loadmat(data_dir)['x_test']
y_test = sio.loadmat(data_dir)['y_test']
# sum(sum(y_test.T)>0)
