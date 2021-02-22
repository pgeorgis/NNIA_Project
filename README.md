# CONLL POS Preprocessing
> This project reads a concatenated .conll file and writes a .tsv file containing tokens and relevant POS annotations, along with a .info summary file containing general information about the dataset.

## Table of contents
* [General Info](#general-info)
* [Setup](#setup)
* [Data Preprocessing](#data-preprocessing)

## General Info
The goal of this project is ultimately to classify English word tokens according to their part of speech (POS). At this stage, the project simply preprocesses POS-annotated data from a .conll input file and summarizes the dataset.  

## Setup
The code runs on a concatenated .conll file, which is yielded from a collection of .gold_conll files by first running the following command in the bash command:
`cat *.gold_conll >> sample.conll`

This command concatenates the contents of all files with the extension .gold_conll in the relevant directory into a single file called "sample.conll".

## Data Preprocessing
Once the setup is complete, we can run the preprocessing step using the following syntax from the command line:
`python3 data_preprocess.py sample.conll sample.tsv sample.info`

* "sample.conll": the input .conll file to be preprocessed
* "sample.tsv": the output .tsv file containing the preprocessed data
* "sample.info": the output summary file