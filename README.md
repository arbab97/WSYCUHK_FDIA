
# Comparison  of  Deep  Learning  Algorithms  for  Site  Detection  of  False  Data Injection Attacks in Smart Grids

## Data
Download the data from below and copy it inside your google drive. 
[Click Here](https://drive.google.com/drive/folders/1aryy2jZXwGlRQS-Hh8_CzjO2Mqq_Pg8k?usp=sharing)

For two Bus Systems (IEEE-14 and IEEE-118), Five different variants of dataset are present , which differ in their L-2 Norm.

The Dataset is adopted from  [Here](https://github.com/arbab97/WSYCUHK_FDIA)


## Requirements and Installation
All the requirements are given in `gpu.yml`. However the environment will be set automatically when using the colab/notebook (as explained later). For manual installation use the following commands: 

`conda config --set restore_free_channel true`

`conda env create -f gpu.yml` 

Few important dependencies include:
* TensorFlow(version  1.15.2) 
* Keras(version 2.3.1) 
* Pandas  (version  1.1.3)  
* NumPy (version    1.18.1) 

## Training Data
### The training data file contains：
For training, the input data (x_train) has diemensions 110000×B and output values (y_train) also has dimensions of for all variants of dataset. 
For testing, the input data(x_test) and output data (y_test) have dimensions 10000×B

where B represent the nunmber of readings of Bus size, i.e., 19 for IEEE-14 and 180 for IEEE-118 Bus System



## Running the Models
#### `Locational Detection main script.ipynb`
This python notebook sets the environment for training the models and fetches code from github. It also connects the google drive to the notebook so that the data could be fetched directly from there. This code includes the training for tradeoff analysis, variants analysis and results from individual models. 
For convenience, this could be run on google colab.  


#### `run_all_models.sh`
This shell script is a driver code for `all_models.py` and the following could be selected from here, for running a combination of traiings (as required for tradeoff and variant analysis:
* Epochs
* Model
* Number of layers
* Number of Neurons


#### `all_models.py`
This contains the code for loading, preprocessing and setting the architecture of models.  The following is the format to use this script. 

      `export CUDA_VISIBLE_DEVICES=0 && source activate gpu && python ...Code/all_models.py  --model "$model" --n_epoch "$epochs" \
      --data_dir "/select/the/path/to/dataset/file" \
      --output_dir "store/results/here" \
      --layers "$layer"\
      --neurons "$neuron"\
      --shape B`

## Commulating results from saved models
### `Locational Detection Supplementry Script.ipynb`

## Visualisation  #upload all figures && source CSV files   THEN EXPLAIN:
#### Figure: Comparison between Conventional and Smart Grid
Source file is provided in `figures` folder

#### Figure: Single LSTM Cell
Source file is provided in `figures` folder


#### Figure: Distribution of Compromised Nodes in IEEE-18 Bus System
Use `visualize_distribution`


### Fig.1 
### Fig.1 
### Fig.1 
### Fig.1 
### Fig.1 


### -------------------------------------------------------
