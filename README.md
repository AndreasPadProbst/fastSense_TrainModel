# Train a Word Sense Classifier for the German Language

## Table of contents
* [General info](#general-info)
* [Setup](#setup)
* [How to Use](#how-to-use)
* [Code Example](#code-example)


## General Info
This package trains a word sense classifier from the data that is generate with the algorithm from [https://github.com/AndreasPadProbst/fastSense_CreateData](https://github.com/AndreasPadProbst/fastSense_CreateData).
The main structure of this code is taken from [https://github.com/texttechnologylab/fastSense](https://github.com/texttechnologylab/fastSense) (accessed: 20.02.2021) and modified to be able to process
the German version of the Wikipedia. [https://github.com/texttechnologylab/fastSense](https://github.com/texttechnologylab/fastSense) is a model based on fastSense (see [paper](https://www.aclweb.org/anthology/L18-1168/)).

## Setup
This package was used with **Python 3.9** with the following package specifications:
* **tensorflow** = 2.5

## How to Use
To train the model, the cli_train.py script needs to me executed with the following command line specifications:

* **--data**: Path to the folder containing the training, validation, and test data.
* **--models_dir**: Path to the folder where the trained model should be saved. If the folder already contains an existing model, the script will continue training the model.
* **--final_models_dir**: Path to the folder where the trained models will be saved after each epoch. Each model is placed in a timestamped subfolder, together with information on the used parameters to train that model.
* **--jobs**: Path to a JSON file containing the parameters that should be used to train the model.

## Code Example
```
python cli_train.py --data ./training_data --models_dir ./saved_model --final_ models_dir ./final_model_output --data ./train_jobs.json
```

