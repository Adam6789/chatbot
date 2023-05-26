import torch
import torch.nn as nn
import numpy as np
import random

class Encoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, vocab, dropout=0):
        
        super(Encoder, self).__init__()
        
        # self.embedding provides a vector representation of the inputs to our model
        self.embedding = nn.Embedding(input_size, hidden_size)
        
#         weights_matrix = np.zeros((input_size, hidden_size))
#         words_found = 0
#         for i, word in enumerate(vocab.words):
#             try: 
#                 weights_matrix[i] = glove[word]
#                 words_found += 1
#             except KeyError:
#                 weights_matrix[i] = np.random.normal(scale=0.6, size=(hidden_size, ))
#         print(f"For {words_found} of {len(vocab.words)} words an entry has been found in glove.")
        
#         weights_matrix = torch.from_numpy(weights_matrix)
#         self.embedding.load_state_dict({'weight':weights_matrix})
        
        # self.lstm, accepts the vectorized input and passes a hidden state
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        
    
    def forward(self, i, h):
        
        '''
        Inputs: i, the src vector
        Outputs: o, the encoder outputs
                h, the hidden state (actually a tuple of hidden state and cell state)
        '''
        embedding = self.embedding(i)
        x,y = h
        o, h= self.lstm(embedding, h)
        o = self.dropout(o)
        
        return o, h
    

class Decoder(nn.Module):
      
    def __init__(self, hidden_size, output_size, vocab, dropout):
        
        super(Decoder, self).__init__()
        
        # self.embedding provides a vector representation of the target to our model
        self.embedding = nn.Embedding(output_size, hidden_size)
        
#         weights_matrix = np.zeros((output_size, hidden_size))
#         words_found = 0
#         for i, word in enumerate(vocab.words):
#             try: 
#                 weights_matrix[i] = glove[word]
#                 words_found += 1
#             except KeyError:
#                 weights_matrix[i] = np.random.normal(scale=0.6, size=(hidden_size, ))
        
#         weights_matrix = torch.from_numpy(weights_matrix)
#         self.embedding.load_state_dict({'weight':weights_matrix})
        
        # self.lstm, accepts the embeddings and outputs a hidden state
        self.lstm = nn.LSTM(hidden_size, hidden_size)

        # self.ouput, predicts on the hidden state via a linear output layer  
        self.linear = nn.Linear(hidden_size, output_size)
        
        self.dropout = nn.Dropout(p=dropout)
        
        #self.softmax = nn.Softmax(dim=1)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, i, h):
        
        '''
        Inputs: i, the target vector
        Outputs: o, the decoder output
                h, the hidden state (actually a tuple of hidden state and cell state)
        '''

        embedding = self.embedding(i)

        o, h = self.lstm(embedding, h)

        o = self.linear(o)
        
        o = self.dropout(o)

        o = self.softmax(o)

        
        return o, h
        
        

class Seq2Seq(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, vocab_source, vocab_target, dropout_E=0, dropout_D=0, teacher_forcing_ratio=1,certain=True):
        
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, vocab_source, dropout_E)
        self.decoder = Decoder(hidden_size, output_size, vocab_target, dropout_D)
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.certain = certain
        self.hidden_size = hidden_size
        #self.softmax = nn.LogSoftmax(dim=1)
                
        
    
    
    def forward(self, src, trg, start): 
        '''
        Inputs: src, the source vector
                trg, the target vector
        Outputs: o, the prediction
                
        '''
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        src.to(device)
        trg.to(device)
        start.to(device)
        
            
        
        # encoder
        hidden = (torch.zeros(1,self.hidden_size).to(device), torch.zeros(1,self.hidden_size).to(device))
        for word in src:
            #print("word",word.shape,word)
            o, hidden = self.encoder(word.view(-1).to(device), hidden)
            x,y = hidden

        
        # decoder
        o = start
        #rprint(trg)
        
        prediction = []
        for word in trg:

            o, hidden = self.decoder(o.view(-1).to(device), hidden)
            x,y = hidden


            
            if self.training == False:
                if self.certain:             
                    #print(o)
                    o = torch.argmax(o,dim=1)
                    #print(o)
                #else:
                #    o = torch.LongTensor(random.choices(list(range(len(o[0]))), weights=(o[0].detach().cpu().apply_(lambda x: np.exp(x))), k=1))[0]                                                                              
                
            prediction.append(o)
            
            if self.training:
                o = word if random.random() < self.teacher_forcing_ratio else torch.argmax(o,dim=1)
                #o = word if random.random() < self.teacher_forcing_ratio else torch.LongTensor(random.choices(list(range(len(o))), weights=(o.detach().cpu().apply_(lambda x: np.exp(x))), k=1))[0]
                                                    
                



        prediction = torch.stack(prediction)

        prediction = prediction.squeeze()

        return prediction

    

