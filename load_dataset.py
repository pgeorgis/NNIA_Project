import torch, sys
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

#Load bert-base-cased model
model = BertModel.from_pretrained('bert-base-cased', output_hidden_states = True)


def BERT_tokenize(sentence):
    """Adds [CLS] and [SEP] tokens and then tokenizes the sentence using 
    the pre-trained BERT model tokenizer."""
    tokenized = sentence[:]
    tokenized.insert(0, "[CLS]")
    tokenized.append("[SEP]")
    tokenized = tokenizer.tokenize(' '.join(tokenized))
    return tokenized

def pad_sentence(sentence, length, pad=0):
    """Pads a list (sentence) with the specified pad character (default 0)
    until it reaches the specified length"""
    return sentence[:] + [pad]*(length-len(sentence))
    

class Dataset:
    def __init__(self, file):
        self.filepath = file
        self.data = pd.DataFrame()
        
        print('Loading dataset...')
        self.load_dataset()
        self.n_sentences = len(self.data['Sentences'])
        
        
        print('Tokenizing sentences...')
        self.tokenize_sentences()
        self.pad_sentences()
        
        
    def load_dataset(self):
        sentences = [[]]
        POS_tags = [[]]
        with open(self.filepath, 'r') as f:
            f = f.readlines()
            lines = [line.split('\t') for line in f]            
            for line in lines:
                line = [item.strip() for item in line]
                dataset_index, sentence_index = line[0], line[1]
                if sentence_index == '*':
                    sentences.append([])
                    POS_tags.append([])
                else:
                    word, tag = line[2], line[3]
                    sentences[-1].append(word)
                    POS_tags[-1].append(tag)
            
            #Remove final empty list
            del sentences[-1]
            del POS_tags[-1]
        
        #Add sentences and POS tags to self.data Dataframe
        self.data['Sentences'] = sentences
        self.data['Tags'] = POS_tags
            
            
    def tokenize_sentences(self):
        #Tokenize the sentences in the dataset using BERT Tokenizer
        self.data['Tokenized'] = self.data['Sentences'].apply(lambda x: BERT_tokenize(x))
        
        #Convert tokenized sentences into BERT vocabulary indices
        self.data['Indices'] = self.data['Tokenized'].apply(lambda x: tokenizer.convert_tokens_to_ids(x))    
    
    def pad_sentences(self):
        #Get the length for each sentence, and the legnth of the longest tokenized sentence
        self.data['Length'] = self.data['Indices'].apply(lambda x: len(x))
        max_length = max(self.data['Length'])
        
        #Pad all tokenized sentences to arrays of this length
        self.padded = torch.tensor(np.array([pad_sentence(sentence, max_length) for sentence in self.data['Indices']]))
        
        #Mask padded values
        self.masked = torch.tensor(np.where(self.padded != 0, 1, 0))
    
   
    def embed_sentences(self, batch_size=100):
        #Split into batches of specified size
        input_batches = torch.split(self.padded, batch_size)
        mask_batches = torch.split(self.masked, batch_size)
        n_batches = len(input_batches)
        
        #Create dictionary to store embeddings
        sentence_embeddings = {}
        
        for i in range(n_batches):
            print(f'Embedding batch #{i+1} of {n_batches}...')
            batch_input, batch_mask = input_batches[i], mask_batches[i]
            
            with torch.no_grad(): 
                outputs = model(batch_input, attention_mask=batch_mask)
                hidden_states = outputs[2]
            
            #Reshape the hidden state outputs such that we can see the vector 
            #embeddings for each token
            #current dimensions: layers, batches, tokens, features
            #desired dimensions: batches, tokens, layers, features
            token_embeddings = torch.stack(hidden_states, dim=0)
            token_embeddings = token_embeddings.permute(1,2,0,3)

            
            #Iterate through sentences in each batch
            for j in range(len(batch_input)):
                sentence_ID = (i*batch_size)+j
                
                #Get the length of the unpadded sentence
                n_tokens = self.data['Length'][sentence_ID]
                
                #Only save embeddings for original tokens (without padding)
                sentence_embedding = token_embeddings[j][:n_tokens]
                
                #Save only the final layer embeddings for each token
                final_layer = [token[-1] for token in sentence_embedding]
                sentence_embeddings[sentence_ID] = final_layer
        
        #Save final embeddings to self.data DataFrame
        self.data['Embeddings'] = list(sentence_embeddings.values())
  

    