import numpy as np
import torch
  
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


def train():
    questions = src.utils.tokenize_questions(raw_questions, vocab_source)
    answers = src.utils.tokenize_answers(raw_answers, vocab_target)
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = torch.cuda.is_available()
    #is_cuda = False
    print(f"Computing on {device}.")
    
    
    # hyperparams
    epochs = 5
    batch_size = 64
    print_each = 30
    
    lr = 0.1
    weight_decay = 0.0
    penalize_early_eos = 1 # the smaller the higher the penalty, i.e. the less the weight
    
    #hidden_size is defined at the very top
    teacher_forcing_ratio = 0.5 # the higher the teacher_forcing_ratio, the easier it is to learn
    dropout_E=0.5
    dropout_D=0.0
    
    
    
    
    
    
    split = int(0.98*len(questions))
    batches = len(questions[:split]) // batch_size
    
    
    
    # model
    input_size = len(vocab_source.words)
    hidden_size = hidden_size
    output_size = len(vocab_target.words)
    model = Seq2Seq(input_size, hidden_size, output_size, vocab_source, vocab_target, glove, dropout_E, dropout_D, teacher_forcing_ratio=teacher_forcing_ratio)
    
    
    v = "submission"
    if Path(f"checkpoints/model_{v}.pt").is_file():
        model.load_state_dict(torch.load(f"checkpoints/model_{v}.pt", map_location=torch.device('cpu')))
        print(f"loading from checkpoint: 'checkpoints/model_{v}.pt'")
    else:
        print(f"nothing to load at checkpoint: 'checkpoints/model_{v}.pt'")
    v="submission"
    
    
    
    
    model.to(device)
    # training
    
    
    optim = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    weight = torch.Tensor([1]*output_size).to(device)
    weight[2]=penalize_early_eos
    loss_fn = nn.NLLLoss(weight=weight)
    
    epoch = 0
    missed = 0
    for epoch in range(epochs):
        train_loss = 0
        for i, (batch_q, batch_a) in enumerate(zip(src.utils.heteroDataLoader(questions[:split],batch_size), src.utils.heteroDataLoader(answers[:split],batch_size))):   
            #print(i,"yes")
            # training loop
            model.train()
            batch_loss = 0
    
            for m, (q, a) in enumerate(zip(batch_q, batch_a)):  
                start = next(iter(torch.LongTensor([vocab_target.words["<SOS>"]])))
                start.to(device)
                a=a.to(device)
                q=q.to(device)
                output = model(q,a,start)
                output.to(device)
                model.to(device)
                try:
                    loss = loss_fn(output,a)
                except:
                    print(f"Loss could not be computed for: {output} with shape {output.shape}.")
                    print(f"Question: {q}")
                    print(f"Answer: {a}")
                    print(f"Old loss will be used with the value of {loss}.")
                    loss = loss
                    
                batch_loss += loss
    
            batch_loss = batch_loss/batch_size
            batch_loss.backward()
            optim.step()
            optim.zero_grad()
            train_loss+=batch_loss
       
            #print(i)
            if i % print_each== 0:
                print("batch:", f"{i}/{batches}")
    
        
                
            if i % (print_each * 2) == 0:
                valid_loss = 0
                for n, (batch_q, batch_a) in enumerate(zip(src.utils.heteroDataLoader(questions[split:],batch_size), src.utils.heteroDataLoader(answers[split:],batch_size))):     
                    # evaluation loop
                    model.eval()
                    batch_loss = 0
                    for q, a in zip(batch_q, batch_a):      
                        start = next(iter(torch.LongTensor([vocab_target.words["<SOS>"]])))
                        q,a,start = q.to(device),a.to(device),start.to(device)
                        assert len(q.shape) ==1, f"Answer must be 1-dimensional. But {q.shape}"
                        assert len(a.shape) ==1, f"Answer must be 1-dimensional. But {a.shape}"
                        output = model(q,a,start)
                        try:
                            loss = loss_fn(output,a)
                        except:
                            missed = missed +1
    #                         print(f"Loss could not be computed for: {output} with shape {output.shape}.")
    #                         print(f"Question: {q}")
    #                         print(f"Answer: {a}")
    #                         print(f"Old loss will be used with the value of {loss}.")
                            loss = loss
                        batch_loss += loss
     
                    valid_loss += batch_loss / batch_size
                valid_loss = round(valid_loss.item() / (len(questions[split:])//batch_size),3)
                train_loss = round(train_loss.item() / ((i+1)*5),3)
    
                print(" ")
                
                print("epoch:", epoch,"batch:", f"{i}/{batches}","train_loss:",train_loss, "valid_loss", valid_loss)
                text = ""
                for x in q:
                    text += vocab_source.index[str(x.item())] + " "
                print("question:",text)
                text = ""
                for x in a:
                    text += vocab_target.index[str(x.item())] + " "
                print("answer:", text)
                text = ""
                for x in output:
                    text += vocab_target.index[str(torch.argmax(x,dim=0).item())] + " "
                print("prediction:", text)
                torch.save(model.state_dict(),f"checkpoints/model_{v}.pt")
                print(f"{missed}/{print_each * 2 * batch_size} samples could not be considered.")
                print("Saved model.")
                print("")
                missed = 0
                train_loss = 0
                
                
                      
    
    
                