#Load load_dataset.py script, including all classes, models, and functions
from load_dataset import *

#Load other modules
import torch
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
#len(tags) = 49


#Fit label encoder to full set of POS tags
le = LabelEncoder()    
le.fit(tags)


def chunk_list(lis, n):
    """Splits a list into sublists of length n; if not evenly divisible by n,
    the final sublist contains the remainder"""
    return [lis[i * n:(i + 1) * n] for i in range((len(lis) + n - 1) // n)]


def prepare_data(filepath, encoder=le, max_size=None, batch_size=20):
    """Reads in dataset from file to Dataset class, embeds sentences in set,
    transforms embeddings and labels to proper format for input to NN, 
    returns tuple of input and output features"""
    
    #Load dataset of specified maximum size (maximum number of sentences) from file
    dataset = Dataset(filepath, max_size)
    
    #Embed the sentences of the dataset in batches of specified size
    dataset.embed_sentences(batch_size=batch_size)
    
    #Input data: the embeddings matched with their corresponding POS tags
    data = dataset.data['Tagged Embeddings']
    
    #Reshape/transform input data:
    #Currently it is organized into sentences, but we want it organized as 
    #a flattened list of individual words (POS tags) with their embeddings
    X = [word_data[1] for sentence in data for word_data in sentence]
    X = torch.stack(tuple([X[i] for i in range(len(X))]))
    
    #Likewise transform output data into a flattened list of POS tags
    y = [word_data[0] for sentence in data for word_data in sentence]
    
    #Encode the output labels using specified encoder
    y_enc = torch.tensor(encoder.transform(y))
    
    #Return only the transformed input and output features
    return X, y, y_enc
    

#Prepare train, validation, and test datasets
#TRAINING SET 
train_set = prepare_data('/Users/phgeorgis/Documents/School/MSc/Saarland_University/Courses/Semester_5/Neural_Networks/Assignments/Project/dataset_train.tsv',
                         max_size=300)

train_X, train_y, train_y_enc = train_set


#VALIDATION SET
validation_set = prepare_data('/Users/phgeorgis/Documents/School/MSc/Saarland_University/Courses/Semester_5/Neural_Networks/Assignments/Project/dataset_validation.tsv',
                              max_size=100)

val_X, val_y, val_y_enc = validation_set


#TEST SET
test_set = prepare_data('/Users/phgeorgis/Documents/School/MSc/Saarland_University/Courses/Semester_5/Neural_Networks/Assignments/Project/dataset_test.tsv', 
                        max_size=100)

test_X, test_y, test_y_enc = test_set




#BUILD NEURAL NET MODEL
class Net(nn.Module):
    def __init__(self):
        #Inherit methods and attributes of parent class
        super(Net, self).__init__() 
        
        #fc1 = 1st Fully Connected layer
        #768 input dimensions from BERT embeddings; output to next layer is 64 dimension
        self.fc1 = nn.Linear(768, 64)
        
        #Additional fully connected layers
        #Input is has dimensions = 64 (64x1) now because it takes input from fc1
        self.fc2 = nn.Linear(64, 64) 
        self.fc3 = nn.Linear(64, 64)
        
        #Output 49 dimensions after final hidden layer --> 49 POS classes
        self.fc4 = nn.Linear(64, 49)
        
        #Attributes to track losses and train/test accuracy
        self.losses = []
        self.train_accuracy = []
        self.test_accuracy = []
        
        
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
    
    def plot_accuracy(self):
        plt.plot(list(range(len(self.train_accuracy))), self.train_accuracy, label='Train')
        plt.plot(list(range(len(self.test_accuracy))), self.test_accuracy, label='Test')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        
    
net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.001) 



def train(NN_model, X, y, encoder=le,
          epochs=20, batch_size=100,
          plot_loss=False, 
          test=False, X_test=None, y_test=None):
    
    #Raise error if no test data has been specified, if test == True
    if test == True:
        if (X_test == None) or (y_test == None):
            print('Input required for X_test and y_test in order to test model!')
            raise ValueError
    
    #Train NN model over specified number of epochs
    for epoch in range(epochs):
        index = 0
        batch_n = 0
        
        #Process input data in batches of specified size
        while index < len(X):
            #Prepare batch input and labels
            batch_input = X[(batch_n)*batch_size:(batch_n+1)*batch_size]
            batch_labels = torch.tensor(encoder.transform(y[(batch_n)*batch_size:(batch_n+1)*batch_size]))
            batch_n += 1
            index += len(batch_input)
            
            #Feed data through model
            NN_model.zero_grad()
            output = NN_model(batch_input.view(-1,768))
            
            #Calculate and backpropagate loss
            loss = F.nll_loss(output, batch_labels)
            loss.backward()
            optimizer.step()
        
        #Save and print the loss at the current epoch
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
        
    
    





def test_model(NN_model, X_test, y_test, encoder=le,
               batch_size=500, 
               print_results=True, return_results=False):
    
    correct, total = 0, 0
    with torch.no_grad():
        
        batch_indices = list(range(len(X_test)))
        batches = chunk_list(batch_indices, batch_size)
        
        #Process test data in batches of specified size
        for i in range(len(batches)):
            
            #Prepare batch input features and output labels
            batch = batches[i]
            batch_input = X_test[batch[0]:batch[-1]+1]
            batch_labels = torch.tensor(encoder.transform(y_test[batch[0]:batch[-1]+1]))
    
            #Feed data through model
            output = NN_model(batch_input.view(-1,768))
            
            #Check whether model prediction is correct
            for index, j in enumerate(output):
                if torch.argmax(j) == batch_labels[index]:
                    correct += 1
                total += 1
   
            #print(f'Accuracy at Batch {i}: {correct} correct of {total} ({round((correct/total)*100, 2)}%)')
    
    if print_results == True:
        print(f'Accuracy: {correct} correct of {total} ({round((correct/total)*100, 2)}%)')
    
    if return_results == True:
        return correct/total


        
def main():
    pass


