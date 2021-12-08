#Reference:
# 1) https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0
import scipy.io as sio 
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
font = {'family' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)
import random


# Iterate through the five airlines
for i in range(1,6):
    data_dir="/media/rabi/Data/11111/openuae/data_ieee14_118_locational_detection/data118_"+ str(i)+".mat"
    y_train= sio.loadmat(data_dir)['y_train']
    # Subset to the airline
    attacks_per_node=sum(y_train.T)*(random.randrange(90,100)/100)
    
    # Draw the density plot
    sns.distplot(attacks_per_node, hist = False, kde = True,
                 kde_kws = {'linewidth': 2},
                 label ="$L_{2}-norm$="+  str(i))
    
# Plot formatting
plt.legend(prop={'size': 13}, title = 'Dataset Variant', loc ='upper right')
plt.title('Distribution of Compromised Nodes in IEEE-118')
plt.xlabel('Number of Compromised nodes (out of 180)')
plt.ylabel('Density')
plt.show()
print("Done")