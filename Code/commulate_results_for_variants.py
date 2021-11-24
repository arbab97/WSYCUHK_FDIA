
import pandas as pd
pd.set_option("display.precision", 2) 
import numpy as np
# models=["MLP", "CNN", "LSTM", "Attention" ,"cnn-lstm-paper-experiments" ]
models=["MLP", "CNN", "LSTM","cnn-lstm-paper-experiments" ] #Change this 
variants=[1,2,3,4,5]
layers=[2]
neurons=[128]
indiv_results_directory="/content/drive/MyDrive/openUAE/locational detection/variant_analysis/IEEE118/"



all_results=pd.DataFrame(columns={
"Variant","MLP", "CNN", "LSTM","cnn-lstm-paper-experiments"}) 


## now loop through the above array
for variant in variants:
    single_result_i={"Variant":variant}
    for model in models:
      for layer in layers:
        for neuron in neurons:
            # print (" Currently Trying ---------------------->",  model, layer, neuron)
            try:
                file_name="results_"+model+"_"+str(layer)+"_"+str(neuron)+".csv"
                indiv_data=pd.read_csv(indiv_results_directory+'/'+str(variant)+'/'+file_name)
                single_result_i[model]=float(indiv_data["Row Accuracy"])*100 
                print(" Saved  ---------------------->",  variant, model, layer, neuron)
            except:
                print (" Not Found ---------------------->", variant, model, layer, neuron)
    all_results=all_results.append(single_result_i, ignore_index=True)
#Sort and secondary sort the results
all_results.sort_values(['Variant'], ascending=[True])
#Save all_results
all_results[["Variant"]+models].to_csv(indiv_results_directory+"tradeoff.csv", float_format='%.2f')
print("Done")
