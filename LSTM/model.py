import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import keras
from keras.utils import to_categorical


# replace with any text file containing full set of data
mozart_data = './txt-files/notewise/custom/mozart.txt'

with open(mozart_data, 'r') as file:
    text = file.read()
    file.close()
    
    
# get vocabulary set
words = sorted(tuple(set(text.split())))
n = len(words)

# create word-integer encoder/decoder
word2int = dict(zip(words, list(range(n))))
int2word = dict(zip(list(range(n)), words))

# encode all words in dataset into integers
encoded = np.array([word2int[word] for word in text.split()])


# define model using the pytorch nn module
class WordLSTM(nn.ModuleList):
    
    def __init__(self, sequence_len, vocab_size, hidden_dim, batch_size):
        super(WordLSTM, self).__init__()
        
        # init the hyperparameters
        self.vocab_size = vocab_size
        self.sequence_len = sequence_len
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        
        # first layer lstm cell
        self.lstm_1 = nn.LSTMCell(input_size=vocab_size, hidden_size=hidden_dim)
        
        # second layer lstm cell
        self.lstm_2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim)
        
        # dropout layer
        self.dropout = nn.Dropout(p=0.5)
        
        # fully connected layer
        self.fc = nn.Linear(in_features=hidden_dim, out_features=vocab_size)
        
    # forward pass in training   
    def forward(self, x, hc):
        """
            accepts 2 arguments: 
            1. x: input of each batch 
                - shape 128*149 (batch_size*vocab_size)
            2. hc: tuple of init hidden, cell states 
                - each of shape 128*512 (batch_size*hidden_dim)
        """
        
        # create empty output seq
        output_seq = torch.empty((self.sequence_len,
                                  self.batch_size,
                                  self.vocab_size))
        
        # init hidden, cell states for lstm layers
        hc_1, hc_2 = hc, hc
        
        # for t-th word in every sequence 
        for t in range(self.sequence_len):
            
            # layer 1 lstm
            hc_1 = self.lstm_1(x[t], hc_1)
            h_1, c_1 = hc_1
            
            # layer 2 lstm
            hc_2 = self.lstm_2(h_1, hc_2)
            h_2, c_2 = hc_2
            
            # dropout and fully connected layer
            output_seq[t] = self.fc(self.dropout(h_2))
            
        return output_seq.view((self.sequence_len * self.batch_size, -1))
          
    def init_hidden(self):
        
        # initialize hidden, cell states for training
        return (torch.zeros(self.batch_size, self.hidden_dim),
                torch.zeros(self.batch_size, self.hidden_dim))
    
    def init_hidden_generator(self):
        
        # initialize hidden, cell states for prediction of 1 sequence
        return (torch.zeros(1, self.hidden_dim),
                torch.zeros(1, self.hidden_dim))
    
    def predict(self, word, top_k=5, seq_len=128):
        """
            accepts 3 arguments: 
            1. word: starting word for prediction (prompt)
                - shape 1*149 (1*vocab_size)
            2. top_k: top k words to sample prediction from
            3. seq_len: how many words to generate in the sequence
        """
        
        # set evaluation mode
        self.eval()
        
        # init output sequence vector with pre-defined starting word
        seq = np.empty(seq_len+1)
        seq[0] = word2int[word]
        
        # init hidden, cell states for generation
        hc = self.init_hidden_generator()
        
        # encode starting word to one-hot encoding
        word = to_categorical(word2int[word], num_classes=self.vocab_size)
        
        # add batch dimension
        word = torch.from_numpy(word).unsqueeze(0)
        
        hc_1, hc_2 = hc, hc
        
        # forward pass
        for t in range(seq_len):
            
            # layer 1 lstm
            hc_1 = self.lstm_1(word, hc_1)
            h_1, c_1 = hc_1
            
            # layer 2 lstm
            hc_2 = self.lstm_2(h_1, hc_2)
            h_2, c_2 = hc_2
            
            # fully connected layer without dropout (no need)
            output = self.fc(h_2)
            
            # software to get probabilities of output options
            output = F.softmax(output, dim=1)
            
            # get top k words and corresponding probabilities
            p, top_word = output.topk(top_k)
            
            # sample from top k words to get next word
            p = p.detach().squeeze().numpy()
            top_word = torch.squeeze(top_word)
            
            word = np.random.choice(top_word, p = p/p.sum())
            
            # add word to sequence
            seq[t+1] = word
            
            # encode predicted word to one-hot encoding for next step
            word = to_categorical(word, num_classes=self.vocab_size)
            word = torch.from_numpy(word).unsqueeze(0)
            
        return seq

    
    
def get_batches(arr, n_seqs, n_words):
    """
        create generator object that returns batches of input (x) and target (y).
        x of each batch has shape 128*128*149 (batch_size*seq_len*vocab_size).
        
        accepts 3 arguments:
        1. arr: array of words from text data
        2. n_seq: number of sequence in each batch (aka batch_size)
        3. n_word: number of words in each sequence
    """
    
    # compute total elements / dimension of each batch
    batch_total = n_seqs * n_words
    
    # compute total number of complete batches
    n_batches = arr.size//batch_total
    
    # chop array at the last full batch
    arr = arr[: n_batches* batch_total]
    
    # reshape array to matrix with rows = no. of seq in one batch
    arr = arr.reshape((n_seqs, -1))
    
    # for each n_words in every row of the dataset
    for n in range(0, arr.shape[1], n_words):
        
        # chop it vertically, to get the input sequences
        x = arr[:, n:n+n_words]
        
        # init y - target with shape same as x
        y = np.zeros_like(x)
        
        # targets obtained by shifting by one
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, n+n_words]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        
        # yield function is like return, but creates a generator object
        yield x, y   

        
# compile the network - sequence_len, vocab_size, hidden_dim, batch_size
net = WordLSTM(sequence_len=128, vocab_size=len(word2int), hidden_dim=512, batch_size=128)

# define the loss and the optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# split dataset into 90% train and 10% using index
val_idx = int(len(encoded) * (1 - 0.1))
train_data, val_data = encoded[:val_idx], encoded[val_idx:]

# empty list for the validation losses
val_losses = list()

# empty list for the samples
samples = list()



# finally train the model
for epoch in range(100):
    
    # init the hidden and cell states to zero
    hc = net.init_hidden()
    
    # (x, y) refers to one batch with index i, where x is input, y is target
    for i, (x, y) in enumerate(get_batches(train_data, 128, 128)):
        
        # get the torch tensors from the one-hot of training data
        # also transpose the axis for the training set and the targets
        x_train = torch.from_numpy(to_categorical(x, num_classes=net.vocab_size).transpose([1, 0, 2]))
        targets = torch.from_numpy(y.T).type(torch.LongTensor)  # tensor of the target
        
        # zero out the gradients
        optimizer.zero_grad()
        
        # get the output sequence from the input and the initial hidden and cell states
        # calls forward function
        output = net(x_train, hc)
    
        # calculate the loss
        # we need to calculate the loss across all batches, so we have to flat the targets tensor
        loss = criterion(output, targets.contiguous().view(128*128))
        
        # calculate the gradients
        loss.backward()
        
        # update the parameters of the model
        optimizer.step()
    
        # feedback every 10 batches
        if i % 100 == 0: 
            
            # initialize the validation hidden state and cell state
            val_h, val_c = net.init_hidden()
            
            for val_x, val_y in get_batches(val_data, 128, 128):
        
                # prepare the validation inputs and targets
                val_x = torch.from_numpy(to_categorical(val_x).transpose([1, 0, 2]))
                val_y = torch.from_numpy(val_y.T).type(torch.LongTensor).contiguous().view(128*128)
            
                # get the validation output
                val_output = net(val_x, (val_h, val_c))
                
                # get the validation loss
                val_loss = criterion(val_output, val_y)
                
                # append the validation loss
                val_losses.append(val_loss.item())
                 
                # samples.append(''.join([int2char[int_] for int_ in net.predict("p33", seq_len=1024)]))
                
            with open("./training_output/loss/loss_epoch" + str(epoch) + "_batch" + str(i) + ".txt", "w") as loss_file:
                loss_file.write("Epoch: {}, Batch: {}, Train Loss: {:.6f}, Validation Loss: {:.6f}".format(epoch, i, loss.item(), val_loss.item()))

            with open("./training_output/samples/result_epoch" + str(epoch) + "_batch" + str(i) + ".txt", "w") as outfile:
                outfile.write(' '.join([int2word[int_] for int_ in net.predict("p33", seq_len=512)]))