#plotting for IMSAT
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as metrics
import scipy.io as sio 
from sklearn.metrics import f1_score
data_directory="/media/rabi/Data/11111/openuae/WSYCUHK_FDIA_results_3_Nov/results_3_nov-20211103T044231Z-001/results_3_nov/"
metadata={

    "2axis" : { "file_name": "stats_MLP.csv",
                "y_axis": ["Training Loss", "Validation Loss"],
                "title": "Learning Curve for FDIA locational detection (IEEE-118)"
            },
}



plot_turn='barplot'

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
    plt.rcParams.update({'font.size': 9})
    df = pd.read_csv("/media/rabi/Data/11111/openuae/WSYCUHK_FDIA_results_3_Nov/results_3_nov-20211103T044231Z-001/results_3_nov/results_ALL (copy).csv")

    fig = plt.figure() # Create matplotlib figure
    fig.subplots_adjust(right=0.75)
    fig.subplots_adjust(bottom=0.2)
    ax = fig.add_subplot(111) # Create matplotlib axes
    ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.
    ax3=ax.twinx()
    ax3.spines['right'].set_position(("axes", 1.16))

    width = 0.11

    df["Time Taken"].plot(kind='bar', color='black', ax=ax, width=width, position=1)
    df["Number of Parameters"].plot(kind='bar', color='grey', ax=ax2, width=width, position=0)
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



    # plt.savefig(data_directory+ "All LSTM Models Commulative"+'.jpeg')
# https://stackoverflow.com/questions/24183101/pandas-bar-plot-with-two-bars-and-two-y-axis