
variant="IEEE14"
import pandas as pd
pd.set_option("display.precision", 2) 
import numpy as np
models=["MLP", "CNN", "LSTM", "Attention" ,"cnn-lstm-paper-experiments" ]
layers=[2]
neurons=[128]
indiv_results_directory="/content/drive/MyDrive/openUAE/locational detection/tradeoff_analysis/"+variant+"/"

all_results=pd.DataFrame(columns={
"Model",
"RACC", 
"Time Taken",
"Number of Parameters"}) 


## now loop through the above array
for model in models:
  for layer in layers:
    for neuron in neurons:
        # print (" Currently Trying ---------------------->",  model, layer, neuron)
        try:
            file_name="results_"+model+"_"+str(layer)+"_"+str(neuron)+".csv"
            indiv_data=pd.read_csv(indiv_results_directory+file_name)
            single_result={
            "Model":model,
            "RACC":float(indiv_data["Row Accuracy"])*100  ,
            "Time Taken":float(indiv_data["Time Taken"])/60,
            "Number of Parameters": float(indiv_data["Number of Parameters"])
            }
            all_results=all_results.append(single_result, ignore_index=True)
            print(" Saved  ---------------------->",  model, layer, neuron)
        except:
            print (" Not Found ---------------------->",  model, layer, neuron)

#Sort and secondary sort the results
# all_results.sort_values(['RACC' ], ascending=[ True])
#Save all_results
all_results[["Model",
"RACC", 
"Number of Parameters",
"Time Taken"]].to_csv(indiv_results_directory+ variant+"_main_results.csv", float_format='%.2f')
print("Done")
