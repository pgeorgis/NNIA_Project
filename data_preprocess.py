#NEURAL NETWORKS: IMPLEMENTATION AND APPLICATION
#PROJECT, PART 1: PREPROCESSING DATA
#Philip Georgis (s8phgeor)

from collections import defaultdict
import operator, sys, random

def parse_conll(conll_file):
    """Loads data from a concatenated .conll file, and returns relevant information
    in a dictionary of parsed sequences and a dictionary of POS tag counts."""
    data = defaultdict(lambda:[])
    tag_counts = defaultdict(lambda:0)
    
    with open(conll_file, 'r') as f:
        f = f.read()
        
        #Separate the sequences by the blank line delimiter
        sequences = f.split('\n\n')
        
        #Iterate through sequences
        for i in range(len(sequences)):
            sequence = sequences[i]
            
            #Separate the annotated lines within the sequence
            lines = sequence.split('\n')
            
            #Iterate through annotated lines
            for line in lines:
                
                #Ignore any remaining blank lines
                if len(line) > 0:
                    
                    #Ignore any lines beginning with '#'
                    if line[0] != '#':
                        
                        #Separate annotations
                        line = line.split()
                        
                        #Parse the annotations
                        position, word, POS = line[2], line[3], line[4]
                        
                        #Add annotations to the dictionary of parse sequences
                        data[i].append((position, word, POS))
                        
                        #Update count of POS tags
                        tag_counts[POS] += 1
    
    return data, tag_counts
                


def create_tsv(data, tsv_output):  
    """Creates a .tsv file containing the relevant POS annotations"""            
    index = 1
    data_len = len(data)
    with open(tsv_output, 'w') as w:

        #Iterate through entries of parsed sequences
        for i in data:
            tokens = data[i]
            
            #Iterate through tokens within sequences
            for token in tokens:
                
                #Parse the token and annotations
                position, word, POS = token
                
                #Write data to file
                w.write(f'{index}\t{position}\t{word}\t{POS}\n')
                index += 1
            
            #Separate sequences with '*' (don't add \n if it is the last entry)
            if i == data_len:
                w.write(f'{index}\t*\t\t')
            else:    
                w.write(f'{index}\t*\t\t\n')
            index += 1
    
        
    
def summarize_sequences(data, tag_counts): 
    """Given output from parse_conll() function, returns a summary of sequences
    and POS tag frequencies."""
    
    #Get list of sequence lengths
    sequence_lengths = [len(data[i]) for i in data]
    
    #Maximum sequence length
    max_length = max(sequence_lengths)
    
    #Minimum sequence length
    min_length = min(sequence_lengths)
    
    #Average sequence length
    avg_length = sum(sequence_lengths) / len(sequence_lengths)
    
    #Total number of sequences
    n_sequences = len(data)
    
    #List of POS tags with their normalized frequencies
    tag_frequency = [(tag, tag_counts[tag]/sum(tag_counts.values())) 
                     for tag in tag_counts]
    
    #Sort the POS tags by frequency in descending order
    tag_frequency.sort(key=operator.itemgetter(1), reverse=True)
    
    return max_length, min_length, avg_length, n_sequences, tag_frequency



def write_summary(data, tag_counts, summary_output):
    """Writes a summary of the annotated data to file."""
    
    #Get the summary of the data from the output of parse_conll() function
    summary = summarize_sequences(data, tag_counts)
    max_length, min_length, avg_length, n_sequences, tag_frequency = summary
    
    #Write the data to the designated output file
    with open(summary_output, 'w') as w:
        w.write(f'Maximum sequence length: {max_length}\n')
        w.write(f'Minimum sequence length: {min_length}\n')
        w.write(f'Mean sequence length: {avg_length}\n')
        w.write(f'Number of sequences: {n_sequences}\n')
        w.write('\nTags:\n')
        for tag in tag_frequency:
            POS, freq = tag
            percent = round(freq*100, 2)
            w.write(f'{POS}\t{percent}%\n')
    
    
    
def main():
    """Parses the input .conll file and generates .tsv and summary files."""
    
    #Get the command line arguments, e.g.:
    #python3 data_preprocess.py sample.conll sample.tsv sample.info
    args = sys.argv
    conll_file, output = args[1], args[2]
    train_proportion = float(args[3])
    test_proportion = (1 - train_proportion) / 2
    validation_proportion = test_proportion

    #Parse the .conll file
    print(f'Parsing {conll_file}...')
    data, tag_counts = parse_conll(conll_file)
    
    #Randomly split dataset into train, validation, and test sets
    print(f'Splitting into train ({round(train_proportion*100)}%), validation ({round(validation_proportion*100)}%), and test ({round(test_proportion*100)}%) sets...')
    random.seed(1)
    data_keys = list(data.keys())
    dataset_size = len(data_keys)
    random.shuffle(data_keys)
    train_end = round(dataset_size*train_proportion)
    validation_start = train_end + 1
    validation_end = validation_start + round(dataset_size*validation_proportion)
    test_start = validation_end + 1
    training_items = data_keys[:train_end+1]
    validation_items = data_keys[validation_start:validation_end+1]
    test_items = data_keys[test_start:]
    train_set = {index:data[index] for index in training_items}
    validation_set = {index:data[index] for index in validation_items}
    test_set = {index:data[index] for index in test_items}
    
    #Create the .tsv  file for each train/validation/test
    create_tsv(train_set, f'{output}_train.tsv')
    create_tsv(validation_set, f'{output}_validation.tsv')
    create_tsv(test_set, f'{output}_test.tsv')
    print(f'Wrote data from {conll_file} to {output}_train.tsv, {output}_validation.tsv, and {output}_test.tsv.')
    
    #Write the summary file
    write_summary(data, tag_counts, f'{output}.info')
    print(f'Wrote summary of data to {output}.info.')
    
    #Print a message to indicate completion
    print(f'Done.')



if __name__ == "__main__":
    main()



