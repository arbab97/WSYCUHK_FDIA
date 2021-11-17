#Reference:
# 1) https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#plot-roc-curves-for-the-multilabel-problem
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import matplotlib as mlp

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
import scipy.io as sio 
font = {'family' : 'normal',
        'size'   : 14}
mlp.rc('font', **font)
# Import some data to play with
data_directory="/media/rabi/Data/11111/openuae/WSYCUHK_FDIA_results_3_Nov/"
path=data_directory+"output_LSTM"
pred_y= sio.loadmat(path)['output_mode']
y_test= sio.loadmat(path)['output_mode_pred'] #ignore the naming; its messed up
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
#!!!!!!!!!!!!!!!!!!!!!!!!
# for i in range(y_test.shape[-1]):
#     fpr[i], tpr[i], _ = roc_curve(y_test[ i,:], pred_y[i, :])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], thresh = roc_curve(y_test.ravel(), pred_y.ravel())
total_thresh=len(thresh)
percent=1 #100%
#sliciing
fpr_sliced, tpr_sliced=fpr["micro"][int(-total_thresh*percent):], tpr["micro"][int(-total_thresh*percent):]

roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])  #Here you can't pass sliced BECAUSE area is calculated for all values



plt.figure()
lw = 2
plt.plot(
    fpr_sliced,
    tpr_sliced,
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % roc_auc["micro"],
)
plt.plot([fpr_sliced.min(), fpr_sliced.max()], [tpr_sliced.min(), tpr_sliced.max()], color="navy", lw=lw, linestyle="--")
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Plot (After Micro-Averaging Multi-Class Labels) ")
plt.legend(loc="lower right")
plt.show()
print("Done")