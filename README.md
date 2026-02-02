🇺🇸 US 
---

# Research Project Biomass Prediction 

This repo contains a research project performed by Matheus Sousa and oriented by Antonio Goulart about predicting Above Ground Biomass from time series images from Finland Forests. It contains two projects that runs separately: an exploratory data analysis focused on understand the time series images and their relation with the predictions made by the models available in the [biomassters competition](https://github.com/drivendataorg/the-biomassters). The second project is an experimentation where we change the backbone of the first winner model in the competition to see if there's some form to improve the model and discuss results of multiple backbones including the default backbone chosen by the model creators.

## Exploratory Data Analysis

All commands will be given here with the context as inside the directory `exploratory_data_analysis_test_dataset`.

To run the EDA, first, build the environment using anaconda with `conda env create -f environment.yml` or if you're using pyenv or simply pip, use the `requirements.txt` file to install the dependencies.

Once installed, select the environment you just created ("EDA_biomassters" in case you're using anaconda).

As a second requirement, you need to put the data in place to run the notebook. Download the `features_metadata.csv` and the `test_features` zip files from the [HuggingFace page](https://huggingface.co/datasets/nascetti-a/BioMassters/tree/main), uncompress it and replace the directory and the csv file in the symlink left in the project.

You'll also need the predictions of each model to run the notebook. Download the [1st place predictions](https://drive.google.com/file/d/1n_NPszN-HAirMdNePUt6RuWxi_0_D-HR/view?usp=sharing) and [2nd place predictions](https://drive.google.com/file/d/1fh-s-Juxet3qB4T9rA6Yi5Q336iG3vEp/view?usp=sharing), uncompress and replace it in place of the symlink in the project.

Now, you're able to run the project. 

## First winner Biomassters model experimentation

All commands will be given here with the context as inside the directory `1st-place-biomassters`.

This experiment is an adaptation from the first winner model available in the [biomassters competition github](https://github.com/drivendataorg/the-biomassters). The adaptation was made mainly to cover ViTs (Vision Transformers) and to run in the [Euler cluster](https://euler.icmc.usp.br/).

To run this model, you must create an environment with `conda env create -f environment.yml` and later access it with `conda activate .1st-place-biomassters`.
You can create this environment with pyenv or pip. Just use the requirements with a python version 3.8 for the project. 

With environment installed, you must download the datasets and the metadata. You can download in the [HuggingFace page](https://huggingface.co/datasets/nascetti-a/BioMassters/tree/main), uncompress it and put it in the data directory (you must create this directory replacing the symlink). The data directory must contain: a `train_features`, `train_agbm` and `test_features` directories and the `features_metadata.csv` file.

Additionally, you must replace the models symlink with an empty directory of the same name. The model will save all weights and logs in that directory. 

If you wanna test the best model given by the creator, just download the weights model and put it in `models` directory. You can download the weights [here](https://disk.yandex.ru/d/01YXhPyiKZifYw)


Now, you can run the project with the `run.sh` file to train the model or with `submit.sh` to only predict future AGBM's. The `run_manager.sh` file was made only to automate job creations in the cluster to train progressively each backbone. 