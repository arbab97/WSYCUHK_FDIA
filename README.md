
# Comparison  of  Deep  Learning  Algorithms  for  Site  Detection  of  False  Data Injection Attacks in Smart Grids

## Data
Download the data from below and copy it inside your google drive. 
[Click Here](https://drive.google.com/drive/folders/1aryy2jZXwGlRQS-Hh8_CzjO2Mqq_Pg8k?usp=sharing)

Download the results and output summary files 
[Here](https://drive.google.com/drive/folders/1a-kdXNuLUFz2sxxxgfgn7O0Jy7bAJG9-?usp=sharing)

Download the Trained models For Tradeoff Analysis [Here](https://drive.google.com/drive/folders/1sjSekF13ypKEPIrtjM2eW8rqV10UIyld?usp=sharing)

Download the Trained models For Variant Analysis [Here](https://drive.google.com/drive/folders/1ev97Eef62JAzIoIC1nknl4_W6tTnY7RK?usp=sharing)


For two Bus Systems (IEEE-14 and IEEE-118), Five different variants of dataset are present , which differ in their L-2 Norm.

The Dataset is adapted from  [Here](https://github.com/arbab97/WSYCUHK_FDIA)


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

### Generation of dataset

The following exceprt from paper explains the generation of the dataset. 

•	The load was simulated on each bus. In terms of distribution of readings, these were normally distributed with their mean equal to the input load, and variance was set to be one-sixth of that value.

•	Nodes were selected randomly to generate the FDIA.

•	A set of target state variables (distributed from 2 to 5 for IEEE-14 and from 2 to 10 for IEEE-118 bus system) was chosen to attack at random to generate compromised readings.

•	Now, to generate the compromised readings, it was required that they pass the BDD process, which is employed as a standard defense mechanism technique in smart meters. To achieve this purpose, injection attack data based on the work done by Suzhi Bi et al. [22], which required only incomplete information of the network topology via min-cut strategy, was used.

•	Finally, to take into consideration the noise in measurement encountered practical conditions, a random gaussian noise (with a standard deviation of 0.2) was added in both the attacked and original readings.




## Running the Models
#### `Locational Detection main script.ipynb`
This python notebook sets the environment for training the models and fetches code from github. It also connects the google drive to the notebook so that the data could be fetched directly from there. This code includes the training for tradeoff analysis, variants analysis and results from individual models. 
For convenience, this could be run on google colab.  


#### `code/run_all_models.sh`
This shell script is a driver code for `all_models.py` and the following could be selected from here, for running a combination of traiings (as required for tradeoff and variant analysis:
* Epochs
* Model
* Number of layers
* Number of Neurons


#### `code/all_models.py`
This contains the code for loading, preprocessing and setting the architecture of models.  The following is the format to use this script. 

      `export CUDA_VISIBLE_DEVICES=0 && source activate gpu && python ...Code/all_models.py  --model "$model" --n_epoch "$epochs" \
      --data_dir "/select/the/path/to/dataset/file" \
      --output_dir "store/results/here" \
      --layers "$layer"\
      --neurons "$neuron"\
      --shape B`

## Commulating results from saved models using `Locational Detection Supplementry Script.ipynb`
This script commulate results for three analysis:
#### 1) `commulate_results.py`:
This is used to commulate results for tradeoff analysis
#### 1) `commulate_results_for_variants.py`:
This is used to for variant analysis results 
#### 1) `commulate_results_main_models.py`: 
This is used for individual model. Please note that this script uses the models already trained during tradeoff analysis. The default model selected here had 2 layers and 128 neurons/layer. 

## Visualisation  
#### Figure: Comparison between Conventional and Smart Grid
Source file is provided in `figures` folder

#### Figure: Single LSTM Cell
Source file is provided in `figures` folder


#### Figure: Distribution of Compromised Nodes in IEEE-18 Bus System
Use `code/visualize_distribution.py`. Edit the `data_dir` variable to the path of IEEE-118 dataset. 

#### Figure: Learning Curve of Best and Worst Performing Models for IEEE-14 Bus System
Use `Code/Visualise.py` and make following changes:
* Set `plot_turn` variable to `2axis-2.0`
* Edit the `data_directory` to `Results/main_results/lc and roc of main models/14/"`

#### Figure: Learning Curve of Best and Worst Performing Models for IEEE-118 Bus System
Use `Code/Visualise.py` and make following changes:
* Set `plot_turn` variable to `2axis-2.0`
* Edit the `data_directory` to `{Results Directory}/main_results/lc and roc of main models/118/"`

#### Comparison of Different Models for IEEE-14 Bus System
Use `Code/Visualise.py` and make following changes:
* Set `plot_turn` variable to `bar_plot`
* Edit the path in line `78` to `{Results Directory}/main_results/IEEE14_main_results.csv"`

#### Comparison of Different Models for IEEE-118 Bus System
Use `Code/Visualise.py` and make following changes:
* Set `plot_turn` variable to `bar_plot`
* Edit the path in line `78` to `{Results Directory}/main_results/IEEE118_main_results.csv"`

#### Effect of 𝑙2-normon performance(for IEEE-14 and 118 Bus System)
Use `Code/Visualise.py` and make following changes:
* Set `plot_turn` variable to `multiple_line_plot`
* Edit the `data_directory` to `{Results Directory}/variants_experiment/"`
* Adjust the field `file_name` at line `18` accordintly 


### ---------------------------------------------------------------------------------------------------
