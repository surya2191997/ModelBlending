import torch 
from torchtext import data
# torch.__version__

import nltk
nltk.download('punkt')

def tokenizer(text): # create a tokenizer function
    return nltk.word_tokenize(text)

import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [sent len, batch size]
        
        embedded = self.dropout(self.embedding(x))
        
        #embedded = [sent len, batch size, emb dim]
        
        output, (hidden, cell) = self.rnn(embedded)
        
        #output = [sent len, batch size, hid dim * num directions]
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]
        
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout
        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
                
        #hidden = [batch size, hid dim * num directions]
            
        return self.fc(hidden.squeeze(0))

import numpy as np
import torch.nn.functional as nf
def multiclass_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    rounded_preds = torch.argmax(nf.softmax(preds),1)
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum()/len(correct)
    return acc

from torchtext import datasets
TREC_TEXT = data.Field(tokenize=tokenizer)
TREC_LABEL = data.LabelField()

TREC_train_data, TREC_test_data = datasets.TREC.splits(TREC_TEXT, TREC_LABEL)

print(len(TREC_train_data))
# print(len(TREC_valid_data))
print(len(TREC_test_data))
TREC_TEXT.build_vocab(TREC_train_data, max_size=25000, vectors="glove.6B.300d")
TREC_LABEL.build_vocab(TREC_train_data)

BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TREC_train_iterator, TREC_test_iterator = data.BucketIterator.splits(
    (TREC_train_data, TREC_test_data), 
    batch_size=BATCH_SIZE, 
    device=device)

INPUT_DIM = len(TREC_TEXT.vocab)
print(INPUT_DIM)
EMBEDDING_DIM = 300
HIDDEN_DIM = 256
OUTPUT_DIM = 6
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

lstmmodel2 = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)

pretrained_embeddings = TREC_TEXT.vocab.vectors

print(pretrained_embeddings.shape)
print('ok1')
lstmmodel2.embedding.weight.data.copy_(pretrained_embeddings)

import torch.optim as optim
print('ok2')
optimizer = optim.Adam(lstmmodel2.parameters())
# optimizer = optim.SGD
criterion2 = nn.NLLLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lstmmodel2 = lstmmodel2.to(device)
criterion2 = criterion2.to(device)
print('ok3')
log = nn.LogSoftmax()
def train_model(model, train_iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in train_iterator:
        
        optimizer.zero_grad()
        
        predictions = model(batch.text).squeeze(1)
        
        loss = criterion(log(predictions), batch.label)
        
        acc = multiclass_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(train_iterator), epoch_acc / len(train_iterator)

def evaluate_model(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    pred_lstm = []
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            pred_lstm.append(predictions)
            loss = criterion(log(predictions), batch.label)
            
            acc = multiclass_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator),pred_lstm
print('ok4')
N_EPOCHS = 5
for epoch in range(N_EPOCHS):
    train_loss, train_acc = train_model(lstmmodel2, TREC_train_iterator, optimizer, criterion2)
    valid_loss, valid_acc,temp_pred = evaluate_model(lstmmodel2, TREC_test_iterator, criterion2)
    
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc*100:.2f}%, Val. Loss: {valid_loss:.3f}, Val. Acc: {valid_acc*100:.2f}%')
print('ok5')
import time
stime = time.time()
test_loss_rnn, test_acc_rnn,pred_lstm = evaluate_model(lstmmodel2, TREC_test_iterator, criterion2)
print(time.time()-stime)

print(f'| Test Loss: {test_loss_rnn:.3f} | Test Acc: {test_acc_rnn*100:.2f}% |')

rounded_pred_lstm =[]
for p in pred_lstm:
  rounded_preds = torch.argmax(nf.softmax(p),1)
  rounded_pred_lstm.append(rounded_preds)

print(rounded_pred_lstm)

torch.save(lstmmodel2.state_dict(),'model_lstm_trec_nltk.pth')