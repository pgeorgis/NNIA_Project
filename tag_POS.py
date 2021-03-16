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
        
    def plot_accuracy(self, adjust=0):
        plt.plot([i+adjust for i in range(len(self.train_accuracy))], self.train_accuracy, label='Train')
        plt.plot([i+adjust for i in range(len(self.test_accuracy))], self.test_accuracy, label='Test')
        plt.xlabel('Training Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')

#FULLY CONNECTED NEURAL NETWORK MODEL
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
     

#768 input dimensions from BERT embeddings; output to next layer is 64 dimension
fc_net = FCNet(name='Fully Connected NN', input_size=768, hidden_size=64, output_size=n_labels)
fc_net_optimizer = optim.Adam(fc_net.parameters(), lr=0.001)


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
            # batch the computations into a single matrix multiplication
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
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)

lstm = LSTMNet('LSTM', 768, 50)
lstm_optimizer = optim.Adam(lstm.parameters(), lr=0.001)

def train(NN_model, optimizer, X, y, 
          epochs=20, batch_size=100,
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
            
    #Split data into batches
    n_batches = max(round(len(X)/batch_size), 1)
    X_batches = torch.chunk(X, n_batches, dim=0)
    y_batches = torch.chunk(y, n_batches, dim=0)
    
    #Train NN model over specified number of epochs
    for epoch in range(epochs):
        
        for i in range(n_batches):
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
            #Get training and test accuracy at current epoch
            train_acc = test_model(NN_model, X, y, return_results=True, print_results=False)
            test_acc = test_model(NN_model, X_test, y_test, return_results=True, print_results=False)
            
            #Save these accuracy values
            NN_model.train_accuracy.append(train_acc)
            NN_model.test_accuracy.append(test_acc)
            
    #If plot_loss == True, then also display a plot of loss over training epochs        
    if plot_loss == True:
        plt.plot(list(range(len(NN_model.losses))), NN_model.losses, label='Training Loss')
        
        #If test == True, also plot the training and test accuracy on the same plot
        if ((test == True) and (X_test != None) and (y_test != None)):
            plt.plot(list(range(len(NN_model.train_accuracy))), NN_model.train_accuracy, label='Training Accuracy')
            plt.plot(list(range(len(NN_model.test_accuracy))), NN_model.test_accuracy, label='Test Accuracy')
        
        plt.xlabel('Epochs')
        plt.legend(loc='best')


def test_model(NN_model, X_test, y_test,
               batch_size=500, 
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
                if torch.argmax(j) == batch_labels[index]:
                    correct += 1
                total += 1
    
    if print_results == True:
        print(f'Accuracy: {correct} correct of {total} ({round((correct/total)*100, 2)}%)')
    
    if return_results == True:
        return correct/total


        
def main():
    pass


