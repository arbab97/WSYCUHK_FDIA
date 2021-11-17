
import pandas as pd
pd.set_option("display.precision", 2) 
import numpy as np
models=["MLP", "CNN", "LSTM", "Attention" ,"cnn-lstm-paper-experiments" ]
layers=[1 ,2, 4]
neurons=[64 ,128 ,256]
indiv_results_directory="/media/rabi/Data/11111/openuae/WSYCUHK_FDIA_results_3_Nov/test/"



all_results=pd.DataFrame(columns={
"Model",
"Layers",
"Neurons",
"RACC", 
"Time Taken",
"F1 Score"}) 


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
            "Layers":layer,
            "Neurons":neuron,
            "RACC":float(indiv_data["Row Accuracy"])*100  ,
            "Time Taken":float(indiv_data["Time Taken"])/60,
            "F1 Score": float(indiv_data["F1 Score"]*100)
            }
            all_results=all_results.append(single_result, ignore_index=True)
            print(" Saved  ---------------------->",  model, layer, neuron)
        except:
            print (" Not Found ---------------------->",  model, layer, neuron)

#Sort and secondary sort the results
all_results.sort_values(['Model', 'Layers', 'Neurons'], ascending=[True, True, True])
#Save all_results
all_results[["Model",
"Layers",
"Neurons",
"RACC", 
"F1 Score",
"Time Taken"]].to_csv(indiv_results_directory+"tradeoff.csv")
print("Done")
