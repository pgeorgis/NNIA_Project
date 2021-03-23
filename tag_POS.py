#Load load_dataset.py script, including all classes, models, and functions
from load_dataset import *

#Load other modules
import torch, math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


#Set random seed for replicable results
random_seed = 1
torch.manual_seed(random_seed)


#Use .info file created in preprocessing step to identify full set of POS tags for encoder
tags = []
with open('dataset.info', 'r') as f:
    f = f.readlines()
    f = [item.split('\t')[0] for item in f[6:]]
    tags.extend(f)
#len(tags) = 49, or 50 with 'NONE'

tags.append('NONE')
n_labels = len(tags)


#Fit label encoder to full set of POS tags
le = LabelEncoder()    
le.fit(tags)

#Create dictionary of POS tags and their encodings
encoding_dict = {le.transform([tag])[0]:tag for tag in tags}



def pad_sequence(sequence, pad_item, min_length):
    seq_len = len(sequence)
    if seq_len < min_length:
        return sequence + [pad_item for i in range(min_length-seq_len)]
    else:
        return sequence

def prepare_data(filepath, max_size=None, batch_size=20, encoder=le):
    """Reads in dataset from file to Dataset class, embeds sentences in set,
    transforms embeddings and labels to proper format for input to NN, 
    returns tuple of input and output features"""
    
    #Load dataset of specified maximum size (maximum number of sentences) from file
    dataset = Dataset(filepath, max_size)
    
    #Embed the sentences of the dataset in batches of specified size
    dataset.embed_sentences(batch_size=batch_size)
    
    #Input data: the embeddings matched with their corresponding POS tags
    data = dataset.data['Tagged Embeddings']
    
    n_features = len(data[0][0][1]) #768 features from BERT embeddings
    
    #Get length of longest sequence
    longest_sequence = len(max(data, key=len))
    
    #Create tensor of zeroes with the length of the number of features
    z = torch.zeros(n_features)
    
    #Pad all sentences with pad until they are all the same length as the longest sentence
    pad = ('NONE', z)
    data = data.apply(lambda x: pad_sequence(x, pad, longest_sequence))    
    
    #Extract input features grouped by sentences
    X = [[token[1] for token in sentence] for sentence in data]
    
    #Stack input features into tensor of dimensions: sentences x tokens x features
    X = torch.stack(tuple([torch.stack(tuple([X[j][i] for i in range(len(X[j]))])) 
                           for j in range(len(X))]))    
    
    #Extract output features grouped by sentences
    y = [[token[0] for token in sentence] for sentence in data]
    
    #Encode y as features using specified encoder and stack into tensor
    y = [encoder.transform(y[i]) for i in range(len(y))]
    y = torch.stack(tuple([torch.tensor(y[i]) for i in range(len(y))]))
    
    return X, y

            

#Prepare train, validation, and test datasets
#TRAINING SET 
train_X, train_y = prepare_data('dataset_train.tsv', max_size=4900)


#VALIDATION SET
val_X, val_y = prepare_data('dataset_validation.tsv', max_size=1050)


#TEST SET
test_X, test_y = prepare_data('dataset_test.tsv', max_size=1050)


#GENERAL NETWORK CLASS TEMPLATE
#(containing attributes and methods for storing and plotting performance)
class Net(nn.Module):
    def __init__(self):
        #Inherit methods and attributes of parent class
        super(Net, self).__init__() 
        
        #Attributes to track losses and train/test accuracy
        self.losses = []
        self.train_accuracy = []
        self.test_accuracy = []
        
        #Attributes to track losses and train/test accuracy
        self.losses = []
        self.train_accuracy = []
        self.test_accuracy = []
        
        #Confusion matrix of classification results
        self.confusion = defaultdict(lambda:defaultdict(lambda:0))
        
    def plot_accuracy(self, start_epoch=0, end_epoch=-1, validation=True,
                      title=None, directory=''):
        """Generates a plot of the model's training and test accuracy"""
        if end_epoch == -1:
            indices = list(range(start_epoch, len(self.train_accuracy)))
        else:
            indices = list(range(start_epoch, end_epoch+1))
        accuracies = [self.train_accuracy[i] for i in indices]
        plt.plot(indices, accuracies, label='Train')
        if validation == True:
            test_label = 'Validation'
        else:
            test_label = 'Test'
        plt.plot(indices, self.test_accuracy, label=test_label)
        plt.xlabel('Training Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        if title != None:
            plt.title(title)
            plt.savefig(f'{directory}{title}.png', dpi=100)
        plt.show()
        plt.close()
            

#FULLY CONNECTED FEED-FORWARD NEURAL NETWORK MODEL
class FCNet(Net):
    def __init__(self, name, input_size, hidden_size, output_size):
        #Inherit methods and attributes of parent Net class
        super(FCNet, self).__init__() 
        
        self.name = name
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        #fc1 = 1st Fully Connected layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        #Additional fully connected layers
        #Input is has dimensions = 64 (64x1) now because it takes input from fc1
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        
        #Output n_labels dimensions after final hidden layer --> number of POS classes
        self.fc4 = nn.Linear(hidden_size, output_size)
        
        
    def forward(self, x):
        """Feed forward  step of NN"""
        #Pass x through all hidden layers, with ReLU activation function
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        #No activation function on the last layer
        x = self.fc4(x) 
        
        #Instead apply log softmax function to get (log) probability-like 
        #distribution over multiple output class labels
        return F.log_softmax(x, dim=1)
    


class LSTMNet(Net):
    def __init__(self, name, input_size, hidden_size):
        super().__init__()
        self.name = name
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.Tensor(input_size, hidden_size * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4))
        self.init_weights()
                
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
         
    def forward(self, x, 
                init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device), 
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
         
        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            
            #Batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        
        #Reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        
        return hidden_seq, (h_t, c_t)



#MODEL TRAINING AND TESTING
def train(NN_model, optimizer, X, y, 
          epochs=20, batch_size=32,
          plot_loss=True, 
          test=False, X_test=None, y_test=None):
    
    #Raise error if no test data has been specified, if test == True
    if test == True:
        if (X_test == None) or (y_test == None):
            print('Input required for X_test and y_test in order to test model!')
            raise ValueError
        else:
            #Calculate baseline accuracy prior to training
            train_acc = test_model(NN_model, X, y, return_results=True, print_results=False)
            test_acc = test_model(NN_model, X_test, y_test, return_results=True, print_results=False)
            
            #Save these accuracy values
            NN_model.train_accuracy.append(train_acc)
            NN_model.test_accuracy.append(test_acc)
            print(f'Initial accuracy: {round(train_acc*100, 2)}% (train), {round(test_acc*100, 2)}% (test)')
            
    #Split data into batches
    n_batches = max(round(len(X)/batch_size), 1)
    X_batches = torch.chunk(X, n_batches, dim=0)
    y_batches = torch.chunk(y, n_batches, dim=0)
    
    #Train NN model over specified number of epochs
    for epoch in range(epochs):
        
        for i in range(len(X_batches)):
            batch_input = X_batches[i]
            batch_labels = torch.flatten(y_batches[i])
            
            #Feed data through model
            NN_model.zero_grad()
            
            #Reshape the input to fit the FCNet's specifications
            if type(NN_model) == FCNet:
                output = NN_model(batch_input.view(-1,768))
            else: #LSTM doesn't need input to be reshaped
                output = NN_model(batch_input)
                #but LSTM output returns a tuple, the actual output is the first element
                #and needs to be reshaped to fit the size of the labels
                output = output[0].view(batch_input.shape[0] * batch_input.shape[1], n_labels)
            
            #Calculate and backpropagate loss
            loss = F.nll_loss(output, batch_labels)
            loss.backward()
            
            #Adjust the weights
            optimizer.step()
        
        #Save and print the final loss at the current epoch
        NN_model.losses.append(loss)
        print(f'Epoch #{epoch+1}: {loss} loss')
        
        #If test == True, also test the model at the current epoch on both train and test data
        if test == True:
            
            #Create confusion matrix in final training epoch
            confusion = False
            if epoch == epochs-1:
                confusion = True
            
            #Get training and test accuracy at current epoch
            train_acc = test_model(NN_model, X, y, return_results=True, print_results=False, confusion=confusion)
            test_acc = test_model(NN_model, X_test, y_test, return_results=True, print_results=False, confusion=confusion)
            
            #Save these accuracy values
            NN_model.train_accuracy.append(train_acc)
            NN_model.test_accuracy.append(test_acc)
    
    print(f'Final accuracy: {round(NN_model.train_accuracy[-1]*100, 2)}% (train), {round(NN_model.test_accuracy[-1]*100, 2)}% (test)')
    
    #If plot_loss == True, then also display a plot of loss over training epochs        
    if plot_loss == True:
        plt.plot(list(range(len(NN_model.losses)+1)), [None] + NN_model.losses, label='Training Loss')
        
        #If test == True, also plot the training and test accuracy on the same plot
        if ((test == True) and (X_test != None) and (y_test != None)):
            plt.plot(list(range(len(NN_model.train_accuracy))), NN_model.train_accuracy, label='Training Accuracy')
            plt.plot(list(range(len(NN_model.test_accuracy))), NN_model.test_accuracy, label='Test Accuracy')
        
        plt.xlabel('Epochs')
        plt.legend(loc='best')
        plt.show()
        plt.close()



def test_model(NN_model, X_test, y_test,
               batch_size=500, confusion=True,
               print_results=True, return_results=False):
    
    correct, total = 0, 0
    
    #Split data into batches
    n_batches = max(round(len(X_test)/batch_size), 1)
    X_batches = torch.chunk(X_test, n_batches, dim=0)
    y_batches = torch.chunk(y_test, n_batches, dim=0)
    
    with torch.no_grad():
        
        for i in range(n_batches):
            batch_input = X_batches[i]
            batch_labels = torch.flatten(y_batches[i])

    
            #Feed data through model
            if type(NN_model) == FCNet:
                output = NN_model(batch_input.view(-1,768))
            else:
                output = NN_model(batch_input)
                #but LSTM output returns a tuple, the actual output is the first element
                #and needs to be reshaped to fit the size of the labels
                output = output[0].view(batch_labels.shape[0], n_labels)
                
            
            #Check whether model prediction is correct
            for index, j in enumerate(output):
                prediction = int(torch.argmax(j))
                gold = int(batch_labels[index])
                if prediction == gold:
                    correct += 1
                total += 1
                if confusion == True:
                    NN_model.confusion[encoding_dict[gold]][encoding_dict[prediction]] += 1
    
    if print_results == True:
        print(f'Accuracy: {correct} correct of {total} ({round((correct/total)*100, 2)}%)')
    
    if return_results == True:
        return correct/total



#MODEL INITIALIZATION AND EVALUATION

def evaluate_models(train_sizes, 
                    X_train=train_X, X_val=val_X, X_test=test_X,
                    y_train=train_y, y_val=val_y, y_test=test_y,
                    epochs=10, batch_size=32,
                    return_performance=True):
    
    #Structures for storing trained models and evaluation results
    trained_models = {}
    train_accuracies = {}
    validation_accuracies = {}
    test_accuracies = {}
    
    #Get performance of both models before any training
    fc_net = FCNet(name='Feed-Forward Network', input_size=768, hidden_size=64, output_size=n_labels)
    lstm = LSTMNet('LSTM Network', 768, 50)
    pretrain_fcnet_train = test_model(fc_net, X_test=X_train, y_test=y_train, print_results=False, return_results=True)
    pretrain_fcnet_test = test_model(fc_net, X_test=X_test, y_test=y_test, print_results=False, return_results=True)
    pretrain_lstm_train = test_model(lstm, X_test=X_train, y_test=y_train, print_results=False, return_results=True)
    pretrain_lstm_test = test_model(lstm, X_test=X_test, y_test=y_test, print_results=False, return_results=True)
    train_accuracies[0] = pretrain_fcnet_train, pretrain_lstm_train
    test_accuracies[0] = pretrain_fcnet_test, pretrain_lstm_test
    
    #Iterate through specified training sizes
    for train_size in train_sizes:
        
        #Create subset of train set of specified size
        sub_trainX = X_train[:train_size]
        sub_trainY = y_train[:train_size]
        
        #Initialize the feed-forward model and its optimizer
        fc_net = FCNet(name='Feed-Forward Network', input_size=768, hidden_size=64, output_size=n_labels)
        fc_net_optimizer = optim.Adam(fc_net.parameters(), lr=0.001)
        
        #Train and evaluate the feed-forward model
        print(f'\nTraining Feed-Forward Network model with {train_size} sentences...')
        train(fc_net, fc_net_optimizer, X=sub_trainX, y=sub_trainY, 
              test=True, X_test=X_val, y_test=y_val, 
              epochs=epochs, batch_size=batch_size)
        fc_net.plot_accuracy(title=f'Feed-Forward Network Performance (Train Size = {train_size})')
        
        print(f'Testing Feed-Forward Network model with {train_size} sentences...')
        fc_test_acc = test_model(fc_net, X_test=X_test, y_test=y_test, return_results=True, print_results=True)
        
        #Initialize the LSTM model and its optimizer
        lstm = LSTMNet('LSTM Network', 768, 50)
        lstm_optimizer = optim.Adam(lstm.parameters(), lr=0.001)
        
        #Train and evaluate the LSTM model
        print(f'\nTraining LSTM model with {train_size} sentences...')
        train(lstm, lstm_optimizer, X=sub_trainX, y=sub_trainY, 
              test=True, X_test=X_val, y_test=y_val, 
              epochs=epochs, batch_size=batch_size)
        lstm.plot_accuracy(title=f'LSTM Model Performance (Train Size = {train_size})')
        
        print(f'Testing LSTM model with {train_size} sentences...')
        lstm_test_acc = test_model(lstm, X_test=test_X, y_test=test_y, return_results=True, print_results=True)
        
        #Save the accuracy scores
        train_accuracies[train_size] = fc_net.train_accuracy[-1], lstm.train_accuracy[-1]
        validation_accuracies[train_size] = fc_net.test_accuracy[-1], lstm.test_accuracy[-1]
        test_accuracies[train_size] = fc_test_acc, lstm_test_acc
        
        #Save the trained models
        trained_models[train_size] = fc_net, lstm
    
    #Plot model comparison results
    x = list(train_accuracies.keys())
    y_train_fc = [train_accuracies[size][0] for size in train_accuracies]
    y_test_fc = [test_accuracies[size][0] for size in test_accuracies]
    y_train_lstm = [train_accuracies[size][1] for size in train_accuracies]
    y_test_lstm = [test_accuracies[size][1] for size in test_accuracies]
    plt.plot(x, y_train_fc, label='Feed Forward Model (Train)')
    plt.plot(x, y_test_fc, label='Feed Forward Model (Test)')
    plt.plot(x, y_train_lstm, label='LSTM Model (Train)')
    plt.plot(x, y_test_lstm, label='LSTM Model (Test)')
    plt.xlabel('Training Set Size (# Sentences)')
    plt.ylabel('Accuracy')
    plt.ylim((0,1))
    plt.legend(loc='best')
    plt.title('Model Accuracy by Training Set Size')
    plt.savefig('Model Accuracy by Training Set Size', dpi=200)
    plt.show()
    plt.close()
    
    #Return the evaluation scores and trained models
    if return_performance == True:
        return trained_models, train_accuracies, validation_accuracies, test_accuracies

#Evaluate models on training subsets of varying sizes; save results and trained models
results = evaluate_models(train_sizes=[100, 200, 500, 1000, 2000, 3000, 4000, 4900])
trained_models, train_accuracies, validation_accuracies, test_accuracies = results

