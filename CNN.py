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

# + {"colab": {}, "colab_type": "code", "id": "ys8jEZ3CjJ3W"}
# import subprocess

# + {"colab": {}, "colab_type": "code", "id": "g1kxMLYnjNOE"}
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

# + {"colab": {}, "colab_type": "code", "id": "Duhc8JCbjNjZ"}
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

# + {"colab": {}, "colab_type": "code", "id": "iTFpKAz0jiyR"}
import torch 
from torchtext import data
torch.__version__

# + {"colab": {}, "colab_type": "code", "id": "fjGCfy6MjsIu"}
import spacy.cli
spacy.cli.download('en')

# + {"colab": {}, "colab_type": "code", "id": "aoEZ5plDjtiP"}
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

# + {"colab": {}, "colab_type": "code", "id": "gy2E3dQ2kAai"}
print(len(train_data))
TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d")
LABEL.build_vocab(train_data)

# + {"colab": {}, "colab_type": "code", "id": "_IfFWiF1klVP"}
print(len(test_data))

# + {"colab": {}, "colab_type": "code", "id": "j_Hct1tmko27"}
BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size=BATCH_SIZE, 
    device=device)

# + {"colab": {}, "colab_type": "code", "id": "Ms9onvLpMGVl"}


# + {"colab": {}, "colab_type": "code", "id": "duyFqN7FksNz"}
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



# + {"colab": {}, "colab_type": "code", "id": "OGkUIv3dk1hX"}
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [3,4,5]
OUTPUT_DIM = 1
DROPOUT = 0.5

model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)

# + {"colab": {}, "colab_type": "code", "id": "PCm6bd0ck6pI"}
pretrained_embeddings = TEXT.vocab.vectors

model.embedding.weight.data.copy_(pretrained_embeddings)

# + {"colab": {}, "colab_type": "code", "id": "nmswp64ClAvL"}
import torch.optim as optim

optimizer = optim.Adam(model.parameters())

criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)


# + {"colab": {}, "colab_type": "code", "id": "C39oE3dslGtx"}
def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum()/len(correct)
    return acc


# + {"colab": {}, "colab_type": "code", "id": "CubSSGdAlJhf"}

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
       
        predictions = model(batch.text).squeeze(1)

        loss = criterion(predictions, batch.label)
        
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# + {"colab": {}, "colab_type": "code", "id": "FIi7lWuglMP1"}
def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# + {"colab": {}, "colab_type": "code", "id": "eFewK5FslPGn"}
N_EPOCHS = 5

for epoch in range(N_EPOCHS):

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')

# + {"colab": {}, "colab_type": "code", "id": "Rw2g28iKlRp1"}
test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')

# + {"colab": {}, "colab_type": "code", "id": "z2v1hE98mmoH"}
torch.save(model.state_dict(),'model_cnn.pth')

# + {"colab": {}, "colab_type": "code", "id": "B4xYjyfAnOMn"}
dick = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
dick.load_state_dict(torch.load('model_cnn.pth'))
dick = dick.to(device)

# + {"colab": {}, "colab_type": "code", "id": "wDeYp5VCnhGd"}
import time
start_time = time.time()
test_loss, test_acc = evaluate(dick, test_iterator, criterion)
end_time = time.time()

print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')
print(end_time-start_time)

# + {"colab": {}, "colab_type": "code", "id": "qscMotpPnkh4"}
# TREC
TREC_TEXT = data.Field(tokenize='spacy')
TREC_LABEL = data.LabelField(dtype=torch.float)

TREC_train_data, TREC_test_data = datasets.TREC.splits(TREC_TEXT, TREC_LABEL)

TREC_train_data, TREC_valid_data = TREC_train_data.split(random_state=random.seed(SEED))

# + {"colab": {}, "colab_type": "code", "id": "7H9EK3nKoSLz"}
print(len(TREC_train_data))
print(len(TREC_valid_data))
print(len(TREC_test_data))
TREC_TEXT.build_vocab(TREC_train_data, max_size=25000, vectors="glove.6B.100d")
TREC_LABEL.build_vocab(TREC_train_data)

# + {"colab": {}, "colab_type": "code", "id": "FWZazoS4oymd"}
BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TREC_train_iterator, TREC_valid_iterator, TREC_test_iterator = data.BucketIterator.splits(
    (TREC_train_data, TREC_valid_data, TREC_test_data), 
    batch_size=BATCH_SIZE, 
    device=device)

# + {"colab": {}, "colab_type": "code", "id": "87ga0DTvpe81"}


# + {"colab": {}, "colab_type": "code", "id": "JAgYqjrepOpQ"}
INPUT_DIM = len(TREC_TEXT.vocab)
EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [3,4,5]
OUTPUT_DIM = 6
DROPOUT = 0.5

TREC_model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)

# + {"colab": {}, "colab_type": "code", "id": "7RNGlKkH262E"}
import torch.optim as optim

optimizer = optim.Adam(model.parameters())

criterion = nn.NLLLoss()

TREC_model = TREC_model.to(device)
criterion = criterion.to(device)

# + {"colab": {}, "colab_type": "code", "id": "gDfKt3Tp2G2D"}


# + {"colab": {}, "colab_type": "code", "id": "tfVIIcP-zfxp"}
import numpy as np
def multiclass_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = np.argmax(nn.Softmax(preds),axis = 0)
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum()/len(correct)
    return acc


# + {"colab": {}, "colab_type": "code", "id": "YQmWWjoT4cGV"}
def make_one_hot(labels):
    '''
    Converts an integer label Tensor/Variable to a one-hot variable.
    
    Parameters
    ----------
    labels : torch.Tensor or torch.Variable
        N x H x W, where N is batch size. 
        Each value is an integer representing correct classification.
    
    Returns
    -------
    target : torch.LongTensor or torch.Variable
        N x C x H x W, where C is class number. One-hot encoded.
        Returns `Variable` if `Variable` is given as input, otherwise
        returns `torch.LongTensor`.
    '''
    v = False
    if isinstance(labels, torch.autograd.variable.Variable):
        labels = labels.data
        v = True
        
    C = labels.max() + 1
    labels_ = labels.unsqueeze(1)
    one_hot = torch.LongTensor(labels_.size(0), C, labels_.size(2), labels_.size(3)).zero_()
    target = one_hot.scatter_(1, labels_, 1)
    
    if v and torch.cuda.is_available():
        target = Variable(target)
        
    return target


# + {"colab": {}, "colab_type": "code", "id": "k4h524R4_JYM"}
for batch in TREC_train_iterator:
        batch.label = batch.label.reshape(1,1,batch.label.shape[0])
        batch.label= batch.label.permute(0,2,1) 
        batch_size, k, _ = batch.label.size()
        labels_one_hot = torch.cuda.FloatTensor(batch_size, k, 6).zero_()
        
        labels_one_hot.scatter_(2, batch.label.long(), 1)
        print(labels_one_hot)
  

# + {"colab": {}, "colab_type": "code", "id": "2q7ehAxY60dZ"}
def one_hot(label,num_classes):
#         print(batch.label.shape[0])
        label = label.reshape(1,1,label.shape[0])
        label=label.permute(0,2,1) 
        batch_size, k, _ = label.size()
        labels_one_hot = torch.cuda.LongTensor(batch_size, k, num_classes).zero_()
        
        labels_one_hot.scatter_(2, label.long(), 1)
        return labels_one_hot.squeeze(0)
#         print(labels_one_hot)



# + {"colab": {}, "colab_type": "code", "id": "7kTnYJQWwkSN"}
def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
       
        predictions = model(batch.text).squeeze(1)
        print(type(predictions))
        print(predictions.shape)
        print(type(one_hot(batch.label,6)))
        print(one_hot(batch.label,6).shape)
        loss = criterion(predictions, batch.label.long())
        
        acc = multiclass_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# + {"colab": {}, "colab_type": "code", "id": "s3_0_-Jx0wML"}
def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions,batch.label.long())
            
            acc = multiclass_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# + {"colab": {}, "colab_type": "code", "id": "4i7nUpwg02Pv"}
N_EPOCHS = 5

for epoch in range(N_EPOCHS):

    TREC_train_loss, TREC_train_acc = train(TREC_model, TREC_train_iterator, optimizer, criterion)
    TREC_valid_loss, TREC_valid_acc = evaluate(TREC_model, TREC_valid_iterator, criterion)
    
    print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')

# + {"colab": {}, "colab_type": "code", "id": "K7Rq-gJm07Qm"}

