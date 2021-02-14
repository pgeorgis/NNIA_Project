#NEURAL NETWORKS: IMPLEMENTATION AND APPLICATION
#PROJECT, PART 1: PREPROCESSING DATA
#Philip Georgis (s8phgeor)

from collections import defaultdict
import operator

#Load data from concatenated .conll file
#Save relevant information to a dictionary of parsed sequences
concatenated_file = 'sample.conll'

data = defaultdict(lambda:[])
tag_counts = defaultdict(lambda:0)

with open(concatenated_file, 'r') as f:
    f = f.read()
    sequences = f.split('\n\n') #sequences separated by blank line
    for i in range(len(sequences)):
        sequence = sequences[i]
        lines = sequence.split('\n')
        for line in lines:
            if len(line) > 0: #ignore any remaining blank lines
                if line[0] != '#': #ignore lines beginning with '#'
                    line = line.split()
                    position, word, POS = line[2], line[3], line[4]
                    data[i].append((position, word, POS))
                    tag_counts[POS] += 1
                
                
#Create .tsv file containing relevant data
index = 1
with open('sample.tsv', 'w') as w:    
    for i in data:
        tokens = data[i]
        for token in tokens:
            position, word, POS = token
            w.write(f'{index}\t{position}\t{word}\t{POS}\n')
            index += 1
        w.write(f'{index}\t*\n') #separate sequences with '*'
        index += 1
        
        
#Get some summary information about the data
sequence_lengths = [len(data[i]) for i in data]
max_length = max(sequence_lengths)
min_length = min(sequence_lengths)
avg_length = sum(sequence_lengths) / len(sequence_lengths)
n_sequences = len(data)
tag_frequency = [(tag, tag_counts[tag]/sum(tag_counts.values())) for tag in tag_counts]
tag_frequency.sort(key=operator.itemgetter(1), reverse=True)

#Create summary file with information about data
with open('sample.info', 'w') as w:
    w.write(f'Maximum sequence length: {max_length}\n')
    w.write(f'Minimum sequence length: {min_length}\n')
    w.write(f'Mean sequence length: {avg_length}\n')
    w.write(f'Number of sequences: {n_sequences}\n')
    w.write('\nTags:\n')
    for tag in tag_frequency:
        POS, freq = tag
        percent = round(freq*100, 2)
        w.write(f'{POS}\t{percent}%\n')
    
    