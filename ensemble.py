# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 54}, "colab_type": "code", "id": "ExJDVB0xRFIf", "outputId": "4c64a568-af5f-4689-e5ee-da6c4dc90642"}
# http://pytorch.org/
try:
    import torch
except ImportError:
    from os.path import exists
    from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
    platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
    cuda_output = !ldconfig -p|grep cudart.so|sed -e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/'
    accelerator = cuda_output[0] if exists('/dev/nvidia0') else 'cpu'
    
    !pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.4.1-{platform}-linux_x86_64.whl
    # !pip install torchvision
    # subprocess.check_call(['pip', 'install', '-q', f'http://download.pytorch.org/whl/{accelerator}/torch-0.4.1-{platform}-linux_x86_64.whl'])
    import torch

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 258}, "colab_type": "code", "id": "j0O5sERiSrwH", "outputId": "11aae7e2-6064-427c-f8aa-5b2930e41b3d"}
try:
    import torchtext
except ImportError:
    # subprocess.check_call(['pip', 'install', 'torchtext'])
    !pip install torchtext
try:
    import spacy
except ImportError:
    # subprocess.check_call(['pip', 'install', 'spacy'])
    !pip install spacy

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 34}, "colab_type": "code", "id": "W4CLUYXKStun", "outputId": "7ac3ae35-a5aa-46f1-94b8-9e4333ec4939"}
import torch 
from torchtext import data
torch.__version__

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 136}, "colab_type": "code", "id": "MoibxMzLSvx1", "outputId": "b034c6c7-ae82-4b49-ff08-1a52f4950a62"}
import spacy.cli
spacy.cli.download('en')

# + {"colab": {}, "colab_type": "code", "id": "vbYvfvakSylP"}
import torch
from torchtext import data
from torchtext import datasets
import random

SEED = 1234

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

train_data, valid_data = train_data.split(random_state=random.seed(SEED))

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 34}, "colab_type": "code", "id": "mb-4pC26TUhC", "outputId": "5486f806-712d-495f-801b-4d49f8d8d705"}
print(len(train_data))
TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d")
LABEL.build_vocab(train_data)

# + {"colab": {}, "colab_type": "code", "id": "_7teQfPeTYjo"}
BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size=BATCH_SIZE, 
    device=device)

# + {"colab": {}, "colab_type": "code", "id": "N6fLjyVeTb4E"}
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs,embedding_dim)) for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes)*n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [sent len, batch size]
        
        x = x.permute(1, 0)
                
        #x = [batch size, sent len]
        
        embedded = self.embedding(x)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
            
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim=1))

        #cat = [batch size, n_filters * len(filter_sizes)]
            
        return self.fc(cat)



# + {"colab": {}, "colab_type": "code", "id": "Hv2jqY5VTlI1"}
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [3,4,5]
OUTPUT_DIM = 1
DROPOUT = 0.5

cnnmodel = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 136}, "colab_type": "code", "id": "nSoHO4X1TsvU", "outputId": "a90143d0-5067-4212-cf07-a84ebea91492"}
pretrained_embeddings = TEXT.vocab.vectors

cnnmodel.embedding.weight.data.copy_(pretrained_embeddings)

# + {"colab": {}, "colab_type": "code", "id": "or3pLTxKTtXE"}
import torch.optim as optim

optimizer = optim.Adam(cnnmodel.parameters())

criterion = nn.BCEWithLogitsLoss()

cnnmodel = cnnmodel.to(device)
criterion = criterion.to(device)


# + {"colab": {}, "colab_type": "code", "id": "p8yIybJYTyKN"}
def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum()/len(correct)
    return acc


# + {"colab": {}, "colab_type": "code", "id": "iQ2088bwUn8A"}
def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
       
        predictions = model(batch.text).squeeze(1)
#         print(predictions.shape)
#         print(batch.label.shape)
        loss = criterion(predictions, batch.label)
        
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# + {"colab": {}, "colab_type": "code", "id": "qv3U8qovUsJh"}
def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            print(type(predictions))
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# + {"colab": {"base_uri": "https://localhost:8080/", "height": 102}, "colab_type": "code", "id": "oeN7xmKfUwLz", "outputId": "b4683ddf-0179-4190-945b-6378040c3afb"}
N_EPOCHS = 5

for epoch in range(N_EPOCHS):

    train_loss, train_acc = train(cnnmodel, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(cnnmodel, valid_iterator, criterion)
    
    print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')

# + {"colab": {}, "colab_type": "code", "id": "jSGb9jEGU57i"}


# + {"colab": {"base_uri": "https://localhost:8080/", "height": 6879}, "colab_type": "code", "id": "pGRTQ93AWj7E", "outputId": "4fdcb8c1-0a48-46c3-946a-70d7ef5c2d8a"}
import time 
start_time = time.time()
test_loss, test_acc = evaluate(cnnmodel, test_iterator, criterion)
print(time.time()-start_time)

print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')

# + {"colab": {}, "colab_type": "code", "id": "toLRlbmnWkp3"}
torch.save(cnnmodel.state_dict(),'model_cnn_new.pth')

# + {"colab": {}, "colab_type": "code", "id": "YkWUMR6ZWmuS"}
# ensemble
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


# + {"colab": {}, "colab_type": "code", "id": "AWVM2KaFk89R"}
INPUT_DIM_RNN = len(TEXT.vocab)
EMBEDDING_DIM_RNN = 100
HIDDEN_DIM_RNN= 256
OUTPUT_DIM_RNN = 1
N_LAYERS_RNN = 2
BIDIRECTIONAL_RNN = True
DROPOUT = 0.5

lstmmodel = RNN(INPUT_DIM_RNN, EMBEDDING_DIM_RNN, HIDDEN_DIM_RNN, OUTPUT_DIM_RNN, N_LAYERS_RNN, BIDIRECTIONAL_RNN, DROPOUT)
lstmmodel.load_state_dict(torch.load('model_BiDirectionalLSTM.pth'))
lstmmodel = lstmmodel.to(device)

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 34}, "colab_type": "code", "id": "LM4GzueOlv5x", "outputId": "4c7aaa05-644b-4d92-f1e5-56348369ceae"}
test_loss_rnn, test_acc_rnn = evaluate(lstmmodel, test_iterator, criterion)

print(f'| Test Loss: {test_loss_rnn:.3f} | Test Acc: {test_acc_rnn*100:.2f}% |')


# + {"colab": {}, "colab_type": "code", "id": "NwvyMbf9mL55"}
# ensemble gamma = 0.5
def evaluate_ensemble(cnnmodel,lstmmodel, gamma, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions_cnn = cnnmodel(batch.text).squeeze(1)
            predictions_lstm = lstmmodel(batch.label).squeeze(1)
            
            predictions = (1-gamma)*predictions_cnn + gamma*predictions_lstm
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# + {"colab": {"base_uri": "https://localhost:8080/", "height": 929}, "colab_type": "code", "id": "O01HMKxG4c0q", "outputId": "1e863573-7efc-4cac-fced-2b3046969904"}
GAMMA = torch.tensor(0.5).cuda()
test_loss_en, test_acc_en = evaluate_ensemble(cnnmodel, lstmmodel, GAMMA, test_iterator, criterion)

print(f'| Test Loss: {test_loss_en:.3f} | Test Acc: {test_acc_en*100:.2f}% |')

# + {"colab": {}, "colab_type": "code", "id": "8TNQQj_o4qOI"}

