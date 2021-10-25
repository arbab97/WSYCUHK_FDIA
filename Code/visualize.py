#plotting for IMSAT
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data_directory="/media/rabi/Data/11111/openuae/WSYCUHK_FDIA_results/"
metadata={

    "2axis" : { "file_name": "LSTM_16.csv",
                "y_axis": ["Training Loss", "Validation Loss"],
                "title": "Learning Curve of LSTM for FDIA locational detection 16 Units"
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

    df = pd.read_csv("/media/rabi/Data/11111/openuae/WSYCUHK_FDIA_results/All LSTM Models Commulative.csv")

    fig = plt.figure() # Create matplotlib figure

    ax = fig.add_subplot(111) # Create matplotlib axes
    ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

    width = 0.2

    df["Time Taken"].plot(kind='bar', color='red', ax=ax, width=width, position=1)
    df["Number of Parameters"].plot(kind='bar', color='blue', ax=ax2, width=width, position=0)

    ax.set_ylabel('Time Taken (Seconds)')
    ax2.set_ylabel('\n Number of Parameters')
    ax.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    ax2.legend(loc='upper right', bbox_to_anchor=(0.9, 0.97))
    ax.set_xticklabels(df["Number of LSTM Units"])
    ax.set_xlabel("Number of LSTM Units")
    plt.title("Comparision of Computational parameters for different LSTM Architectures")
    plt.show()

    # plt.savefig(data_directory+ "All LSTM Models Commulative"+'.jpeg')
# https://stackoverflow.com/questions/24183101/pandas-bar-plot-with-two-bars-and-two-y-axis