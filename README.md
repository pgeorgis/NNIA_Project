# CONLL POS Preprocessing
> This project reads a concatenated .conll file and writes a .tsv file containing tokens and relevant POS annotations, along with a .info summary file containing general information about the dataset.

## Table of contents
* [General Info](#general-info)
* [Setup](#setup)
* [Data Preprocessing](#data-preprocessing)
* [Data Loading](#data-loading)

## General Info
The goal of this project is ultimately to classify English word tokens according to their part of speech (POS). At this stage, the project preprocesses POS-annotated data from a .conll input file and summarizes the dataset. It additionally loads in the .tsv file generated from the preprocessing step and uses a BERT model to tokenize and create embeddings for the sentences.  

## Setup
The code runs on a concatenated .conll file, which is yielded from a collection of .gold_conll files by first running the following command in the bash command:
`cat *.gold_conll >> dataset.conll`

This command concatenates the contents of all files with the extension .gold_conll in the relevant directory into a single file called "dataset.conll".

## Data Preprocessing
Once the setup is complete, we can run the preprocessing step using the following syntax from the command line:
`python3 data_preprocess.py dataset.conll dataset 0.7`

* "dataset.conll": the input .conll file to be preprocessed
* "dataset": the name for the output .tsv files containing the preprocessed data
* 0.7: the proportion of the data to be used as the training set; the validation and test tests will be constructed from equal portions of the remaining data (e.g. 70% train, 15% validation, 15% test)

This step will output three .tsv files:
* dataset_train.tsv: the train set
* dataset_validation.tsv: the validation set
* dataset_test.tsv: the test set

## Data Loading
In this step, we load the sentences and POS tags from the .tsv files yielded in the previous step. I had difficulty following/understanding the template of the dataset loader provided, so I have written my own version. 

After running "load_dataset.py", we can load in a dataset by the command:
`dataset = Dataset(filepath)`

* "dataset": the name to be assigned to the new Dataset class object
* "filepath": the path to the stored .tsv file

For example:
`train_set = Dataset('dataset_train.tsv')`

This command will load and tokenize the dataset using the BERT base cased tokenizer. The result is stored in a pandas DataFrame object, accessible via the Dataset's "data" attribute, which contains columns of 'Sentences', 'Tags', 'Tokenized', 'Indices', and 'Length'.

* 'Sentences': the sentences of the dataset [list]
* 'Tags': the POS tags for each word of the (untokenized) sentence [list]
* 'Tokenized': the BERT-tokenized sentence [list]
* 'Indices': the BERT vocabulary indices of the tokenized sentence [list]

The sentences are not embedded automatically upon running the program. To achieve this, we must run the command:
`dataset.embed_sentences(batch_size)`

* batch_size: the number of sentences per batch, by default set at 100

After the embedding completes (which takes some time), the results are stored in a column labeled 'Embeddings' of dataset.data [list of tensors].