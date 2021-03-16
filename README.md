# CONLL POS Classification
> This project reads a concatenated .conll file and writes a .tsv file containing tokens and relevant part of speech (POS) annotations, along with a .info summary file containing general information about the dataset. Next, it uses a BERT model to create word embeddings for all sentences in the corpus, to be used as input for POS classification by an artificial neural network.

## Table of contents
* [General Info](#general-info)
* [Setup](#setup)
* [Data Preprocessing](#data-preprocessing)
* [Data Loading](#data-loading)
* [Model Training](#model-training)
* [Model Testing](#model-testing)

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

This command will load and tokenize the dataset using the BERT base cased tokenizer. 

Optionally, one can include a maximum size of the dataset, e.g. 10000 sentences:
`small_train = Dataset('dataset_train.tsv', max_size=10000)`

In this case, only the first 10000 entries (sentences) of the input .tsv will be stored. 

The result is stored in a pandas DataFrame object, accessible via the Dataset's "data" attribute, which contains columns of 'Sentences', 'Tags', 'Tokenized', 'Indices', and 'Length'.

* 'Sentences': the sentences of the dataset [list]
* 'Tags': the POS tags for each word of the (untokenized) sentence [list]
* 'Tokenized': the BERT-tokenized sentence [list]
* 'Indices': the BERT vocabulary indices of the tokenized sentence [list]

The sentences are not embedded automatically upon running the program. To achieve this, we must run the command:
`dataset.embed_sentences(batch_size)`

* batch_size: the number of sentences per batch, by default set at 100

After the embedding completes (which takes some time), the full embedding results are stored in a column labeled 'Embeddings' of dataset.data [list of tensors]. However, this also includes embeddings for the [CLS] and [SEP] tags at the beginning and end, respectively, of tokenized sentences, and individual embeddings for each part of words which have been tokenized into multiple parts. 

Instead, the column labeled 'Tagged Embeddings' of dataset.data [list of tuples (tag, embeddings)] contains the final embeddings matched to each POS tag for each sentence -- this is the input to the neural network model. To free up memory, the 'Embeddings' column is removed after the 'Tagged Embeddings' column has been generated.

## Model Training
Two model architectures, a fully-connected feed-forward network and an LSTM network, are implemented. Thus far I have slightly modified the implementation of an online example of an LSTM, though I don't yet quite understand it. This is something I will work on in the next week.

The `train()` function allows us to train a specified model given its optimizer, input features `X`, and output labels `y`. By default, the data are processed in batches of 100 sentences and trained over 20 epochs, though these settings are configurable when running the command. The function additionally records the training loss at each epoch, and stores these data in a model attribute. Optionally, the loss may be plotted by setting the parameter `plot_loss=True`. It is also possible to test the model's accuracy during the course of the training epochs by setting the parameter `test=True` and supplying test input features `X_test` and test output labels `y_test` (e.g. from the validation set). An example of the full syntax for running this command is shown below:

`train(NN_model=lstm, optimizer=lstm_optimizer, X=train_X, y=train_y, epochs=20, batch_size=100, plot_loss=True, test=True, X_test=val_X, y_test=val_y)`

## Model Testing
The `test_model()` function allows us to test a particular model on specified data. By default, the accuracy is printed, but the function can be configured to return the accuracy value additionally or instead. An example function call is shown below:

`test_model(NN_model=lstm, X_test=test_X, y_test=test_y, batch_size=500, print_results=True, return_results=False)`

If during training the parameter `train=True` was selected, the train and test accuracies of a model can be easily plotted using the network's plot_accuracy() method, e.g.:

`lstm.plot_accuracy()`

Remaining to be done/implemented:
* Commenting/understanding/tweaking the LSTM model
* Training both models on training sets of varying sizes
* Implementation for use from the command line via main() function, etc.