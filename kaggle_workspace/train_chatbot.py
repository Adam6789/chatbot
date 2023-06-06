import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
def pretrain(model, vQ, vA, w2v):
    
    hidden_size = len(list(model.encoder.parameters())[0][1])
    
    weights_matrix = list(model.encoder.parameters())[0].detach().numpy()
    words_found = 0
    # known_words = []
    # unknown_words = []
    for i, word in enumerate(vQ.words):
        try: 
            weights_matrix[i] = w2v.wv[word]
            words_found += 1
    #         known_words.append(word)
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(hidden_size, ))
    #         unknown_words.append((word,i))
    print(f"For {words_found} of {len(vQ.words)} words an entry has been found in the brown corpus.")
    weights_matrix = torch.from_numpy(weights_matrix)
    model.encoder.embedding.load_state_dict({'weight':weights_matrix})
    
    # DECODER
    
    weights_matrix = list(model.decoder.parameters())[0].detach().numpy()
    words_found = 0
    # known_words = []
    # unknown_words = []
    for i, word in enumerate(vA.words):
        try: 
            weights_matrix[i] = w2v.wv[word]
            words_found += 1
    #         known_words.append(word)
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(hidden_size, ))
    #         unknown_words.append((word,i))
    print(f"For {words_found} of {len(vA.words)} words an entry has been found in the brown corpus.")
    weights_matrix = torch.from_numpy(weights_matrix)
    model.decoder.embedding.load_state_dict({'weight':weights_matrix})
    return model


def train(epochs, batch_size, print_each, lr, weight_decay, model, version, questions_train, answers_train, questions_valid, answers_valid, vQ, vA):  


    batches = len(questions_train) // batch_size
    
    
    if Path(f"model_{version}.pt").is_file():
        model.load_state_dict(torch.load(f"model_{version}.pt", map_location=torch.device('cpu')))
        print(f"Loading from checkpoint: 'model_{version}.pt'")
    else:
        print(f"Nothing to load at checkpoint: 'model_{version}.pt'")
        
    model.to(device) 
    print(f"Computing on {device}.")
    
    optim = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.NLLLoss()
    
    epoch = 0
    for epoch in range(epochs):
        train_loss = 0
        for i, (batch_q, batch_a) in enumerate(zip(heteroDataLoader(questions_train,batch_size), heteroDataLoader(answers_train,batch_size))):   
            model.train()
            batch_loss = 0
    
            for m, (q, a) in enumerate(zip(batch_q, batch_a)):  
                output = model(q,a)
                output.to(device)
                model.to(device)
                loss = loss_fn(output,a)

                
            batch_loss += loss
    
            batch_loss = batch_loss/batch_size
            batch_loss.backward()
            optim.step()
            optim.zero_grad()
            train_loss+=batch_loss
       

    
        
                
            if i % (print_each) == 0:
                valid_loss = 0
                for n, (batch_q, batch_a) in enumerate(zip(heteroDataLoader(questions_valid,batch_size), heteroDataLoader(answers_valid,batch_size))):     
                    # evaluation loop
                    model.eval()
                    batch_loss = 0
                    for q, a in zip(batch_q, batch_a):      
                        assert len(q.shape) ==1, f"Answer must be 1-dimensional. But {q.shape}"
                        assert len(a.shape) ==1, f"Answer must be 1-dimensional. But {a.shape}"
                        output = model(q,a)
                        try:
                            loss = loss_fn(output,a)
                        except:
                            print("could not be computed for:", q, a, output)
                            loss = loss
                        batch_loss += loss
     
                valid_loss += batch_loss / batch_size
                valid_loss = round(valid_loss.item() / (len(questions_valid)//batch_size),3)
                train_loss = round(train_loss.item() / ((i+1)*5),3)    
                print("epoch:", epoch,"batch:", f"{i}/{batches}","train_loss:",train_loss, "valid_loss", valid_loss)
                
                randint = random.randint(0, len(questions_valid)-1)
                question = questions_valid[randint]
                answer = answers_valid[randint]
                prediction = model(question, answer)
                text = ""
                for x in question:
                    text += vQ.index[str(x.item())] + " "
                print("question:",text)
                text = ""
                for x in answer:
                    text += vA.index[str(x.item())] + " "
                print("answer:", text)
                text = ""
                for x in prediction:
                    text += vA.index[str(torch.argmax(x,dim=0).item())] + " "
                print("prediction:", text)

                
                torch.save(model.state_dict(),f"model_{version}.pt")
                print("Saved model.")
                print("")
                train_loss = 0
                
def heteroDataLoader(single_samples, batch_size, shuffle = True):

    """
    Inputs:
    -------
    dataset: list
        A list of single samples.
    Outputs:
    --------
    batches: list
        A list of lists with each having multiple samples.
    """
    len_batches = len(single_samples) // batch_size
    random.shuffle(single_samples)
    batches = []
    for i in range(len_batches):
        batches.append(single_samples[i*batch_size:(i+1)*batch_size])
    return batches
                
                
                      
    
    
                