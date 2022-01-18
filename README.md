
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

## Training Data
### The training data file contains：
For training, the input data (x_train) has diemensions 110000×B and output values (y_train) also has dimensions of for all variants of dataset. 
For testing, the input data(x_test) and output data (y_test) have dimensions 10000×B

where B represent the nunmber of readings of Bus size, i.e., 19 for IEEE-14 and 180 for IEEE-118 Bus System



## Running the Models
### `Locational Detection main script.ipynb`
### `run_all_models.sh`
### `all_models.py`



## Commulating results from saved models
### `Colab Driver Scripts`

## Visualisation  #upload all figures && source CSV files   THEN EXPLAIN:
### Fig.1 
### Fig.1 
### Fig.1 
### Fig.1 
### Fig.1 
### Fig.1 


### -------------------------------------------------------
