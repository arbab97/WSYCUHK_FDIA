#plotting for IMSAT
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
import pandas as pd
import sklearn.metrics as metrics
import scipy.io as sio 
from sklearn.metrics import f1_score
data_directory="/media/rabi/Data/11111/openuae/WSYCUHK_FDIA_results_3_Nov/variants_experiment/"  #FOR VARIANTS 
#data_directory="/media/rabi/Data/11111/openuae/WSYCUHK_FDIA_results_3_Nov/main_results/lc and roc of main models/14/"  #FOR MAIN RESULTS

metadata={

    "2axis" : { "file_name": "stats_CNN_2_128.csv",
                "y_axis": ["Training Loss", "Validation Loss"],
                "title": "Learning Curve for CNN+LSTM Model (IEEE-118)"
            },
    "multiple_line_plot" : { "file_name": "variants_ieee118.csv",
            "title": "Effect of $L_{2}$-norm on Performance for IEEE-118"
        },
    "2axis-2.0" : { "file_name": "stats_Attention_2_128.csv",
            "title": "Learning Curve for Attention (Least Performing model)"
        },
}



plot_turn='multiple_line_plot'

if plot_turn=='2axis':
    meta_selected=metadata[plot_turn]
    read_this=meta_selected["file_name"]
    read_csv=pd.read_csv(data_directory+read_this)
    x1=read_csv["Epoch"]
    y1=read_csv[meta_selected["y_axis"][0]]
    plt.plot(x1, y1, "-b", label=meta_selected["y_axis"][0])

    y2=read_csv[meta_selected["y_axis"][1]]
    plt.plot(x1, y2, "-g", label=meta_selected["y_axis"][1])

    plt.ylim(min( min(y1), min(y2) ), max( max(y1), max(y2) ))
    plt.xlabel("Number of Epochs")
    plt.ylabel("Total Loss")
    plt.title(meta_selected["title"])
    plt.legend(loc="upper right")
    plt.show()
    # plt.savefig(data_directory+ meta_selected["file_name"]+'.jpeg')
    
elif plot_turn=='2axis-2.0':
    font = {'family' : 'normal',
        'size'   : 18}

    mlp.rc('font', **font)
    meta_selected=metadata[plot_turn]
    read_this=meta_selected["file_name"]
    df=pd.read_csv(data_directory+read_this)
    fig, ax = plt.subplots(figsize=(8,4))
    fig.subplots_adjust(left=0.2)
    fig.subplots_adjust(bottom=0.2)
    # multiple line plots
    plt.plot( 'Epoch', 'Validation Loss', data=df,  color='green', linewidth=2) 
    plt.plot( 'Epoch', 'Training Loss', data=df, color='olive', linewidth=2)

    # plt.ylim( 0, 100)
    # plt.plot( 'x_values', 'y3_values', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
    # show legend
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss over Time")
    plt.title(meta_selected["title"], fontsize=18)
    # show graph
    plt.show()
        
elif plot_turn=='barplot':
    # https://towardsdatascience.com/how-to-fill-plots-with-patterns-in-matplotlib-58ad41ea8cf8
    plt.rcParams.update({'font.size': 14, 'font.family': 'Times New Roman'})
    # df = pd.read_csv("/media/rabi/Data/11111/openuae/WSYCUHK_FDIA_results_3_Nov/results_3_nov-20211103T044231Z-001/results_3_nov/results_ALL_ieee14_sub.csv")
    df = pd.read_csv("/media/rabi/Data/11111/openuae/WSYCUHK_FDIA_results_3_Nov/main_results/IEEE118_main_results.csv")
    # fig = plt.figure() # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(9,6.5))

    fig.subplots_adjust(right=0.75)
    fig.subplots_adjust(bottom=0.2)
    # ax = fig.add_subplot(111) # Create matplotlib axes
    ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.
    ax3=ax.twinx()

    ax3.spines['right'].set_position(("axes", 1.25))  # 1.25-0.1 for other plot

    width = 0.25

    df["Time Taken"].plot(kind='bar', color='none', edgecolor='red', ax=ax, width=width, position=1, hatch='/')
    df["Parameters"].plot(kind='bar', color='none', edgecolor='blue', ax=ax2, width=width, position=0, fill=False, hatch='x' )
    df["Row Accuracy"].plot(kind='bar', color='none',edgecolor='green',  ax=ax3, width=width, position=2, fill=False, hatch='///')

    ax.set_ylabel('Time Taken (Minutes)')
    ax2.set_ylabel('\nNumber of Parameters')
    ax3.set_ylabel('\nRow Accuracy')


    # ax.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    # ax2.legend(loc='upper right', bbox_to_anchor=(0.9, 0.97))
    # ax3.legend(loc='upper right', bbox_to_anchor=(0.9, 0.83))

    ax.legend(loc='upper center', bbox_to_anchor=(0.2-0.05, 1.12), fancybox=False, shadow=False, handlelength=2.5, handleheight=1.5, fontsize=12)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.50-0.05, 1.12), fancybox=False, shadow=False, handlelength=2.5, handleheight=1.5, fontsize=12)
    ax3.legend(loc='upper center', bbox_to_anchor=(0.82-0.05, 1.12), fancybox=False, shadow=False, handlelength=2.5, handleheight=1.5, fontsize=12)

    ax.set_xticklabels(df["Model"])
    ax.set_xlabel("Model Type")
    # plt.title("Comparision of Different Models")
    plt.gca().set(xlim=(-0.6,4.4))
    plt.show()


if plot_turn=='multiple_line_plot':
    plt.rcParams.update({'font.size': 14, 'font.family': 'Times New Roman'})
    meta_selected=metadata[plot_turn]
    read_this=meta_selected["file_name"]
    df=pd.read_csv(data_directory+read_this)
    # multiple line plots
    plt.plot( 'L2 Norm', 'MLP', data=df, marker='o',  color='green', linewidth=1.5, markersize=7, linestyle='dashed')
    plt.plot( 'L2 Norm', 'CNN', data=df, marker="v", color='olive', linewidth=1.5,  markersize=7)
    plt.plot( 'L2 Norm', 'LSTM', data=df, marker="^", color='red', linewidth=1.5,  markersize=7, linestyle='dashed')
    plt.plot( 'L2 Norm', 'CNN-LSTM', data=df, marker="<", color='blue', linewidth=1.5,  markersize=7)

    # plt.ylim( 0, 100)
    # plt.plot( 'x_values', 'y3_values', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
    # show legend
    plt.legend(fontsize=11)
    plt.xlabel("$L_{2}$-norm")
    plt.ylabel("Test Row Accuracy")
    plt.title(meta_selected["title"], fontsize=14)
    # show graph
    plt.show()
        

    # plt.savefig(data_directory+ "All LSTM Models Commulative"+'.jpeg')
# https://stackoverflow.com/questions/24183101/pandas-bar-plot-with-two-bars-and-two-y-axis