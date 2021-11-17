#plotting for IMSAT
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as metrics
import scipy.io as sio 
from sklearn.metrics import f1_score
data_directory="/media/rabi/Data/11111/openuae/WSYCUHK_FDIA_results_3_Nov/variants_experiment/"
metadata={

    "2axis" : { "file_name": "stats_cnn-lstm_118.csv",
                "y_axis": ["Training Loss", "Validation Loss"],
                "title": "Learning Curve for CNN+LSTM Model (IEEE-118)"
            },
    "multiple_line_plot" : { "file_name": "variants_ieee118_old.csv",
            "title": "Effect of L2 Norm on the Performance of Deep Learning Models"
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
    plt.savefig(data_directory+ meta_selected["file_name"]+'.jpeg')
    
elif plot_turn=='barplot':
    plt.rcParams.update({'font.size': 14})
    df = pd.read_csv("/media/rabi/Data/11111/openuae/WSYCUHK_FDIA_results_3_Nov/results_3_nov-20211103T044231Z-001/results_3_nov/results_ALL_ieee14_sub.csv")

    fig = plt.figure() # Create matplotlib figure
    fig.subplots_adjust(right=0.75)
    fig.subplots_adjust(bottom=0.2)
    ax = fig.add_subplot(111) # Create matplotlib axes
    ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.
    ax3=ax.twinx()
    ax3.spines['right'].set_position(("axes", 1.16))

    width = 0.11

    df["Time Taken"].plot(kind='bar', color='black', ax=ax, width=width, position=1)
    df["Parameters"].plot(kind='bar', color='grey', ax=ax2, width=width, position=0)
    df["Row Accuracy"].plot(kind='bar', color='red', ax=ax3, width=width, position=2)

    ax.set_ylabel('Time Taken (Seconds)')
    ax2.set_ylabel('\n Number of Parameters')
    ax3.set_ylabel('\n Row Accuracy')
    ax.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    ax2.legend(loc='upper right', bbox_to_anchor=(0.9, 0.97))
    ax3.legend(loc='upper right', bbox_to_anchor=(0.9, 0.83))

    ax.set_xticklabels(df["Model"])
    ax.set_xlabel("Model Type")
    plt.title("Comparision of Different Models")
    plt.show()


if plot_turn=='multiple_line_plot':
    meta_selected=metadata[plot_turn]
    read_this=meta_selected["file_name"]
    df=pd.read_csv(data_directory+read_this)
    # multiple line plots
    plt.plot( 'L2 Norm', 'MLP', data=df, marker='o',  color='green', linewidth=2, markersize=7, linestyle='dashed')
    plt.plot( 'L2 Norm', 'CNN', data=df, marker="v", color='olive', linewidth=2,  markersize=7)
    plt.plot( 'L2 Norm', 'LSTM', data=df, marker="^", color='red', linewidth=2,  markersize=7, linestyle='dashed')
    plt.plot( 'L2 Norm', 'Attention', data=df, marker="<", color='blue', linewidth=2,  markersize=7)

    # plt.ylim( 0, 100)
    # plt.plot( 'x_values', 'y3_values', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
    # show legend
    plt.legend()
    plt.xlabel("L2 Norm")
    plt.ylabel("Test Row Accuracy")
    plt.title(meta_selected["title"])
    # show graph
    plt.show()
        

    # plt.savefig(data_directory+ "All LSTM Models Commulative"+'.jpeg')
# https://stackoverflow.com/questions/24183101/pandas-bar-plot-with-two-bars-and-two-y-axis