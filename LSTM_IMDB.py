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

# + {"colab": {}, "colab_type": "code", "id": "5NYYy5xIpjAp"}
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

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 170}, "colab_type": "code", "id": "HVW0DR2upsdF", "outputId": "85a24ca2-9cc1-40d3-822b-58a36d075932"}
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

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 34}, "colab_type": "code", "id": "AaQ6DEHkpxLS", "outputId": "95483725-b423-403f-c4d0-841fed5d46ad"}
import torch 
from torchtext import data
torch.__version__

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 136}, "colab_type": "code", "id": "_RKw5sgVp0Tn", "outputId": "6d9907d2-d293-41de-beb7-0872204a3f05"}
import spacy.cli
spacy.cli.download('en')

# + {"colab": {}, "colab_type": "code", "id": "1PJs6Bllp2fS"}
SEED = 1234

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(dtype=torch.float)

# + {"colab": {}, "colab_type": "code", "id": "eZzjsYYrp5nV"}
from torchtext import datasets
train, test = datasets.IMDB.splits(TEXT, LABEL)

# + {"colab": {}, "colab_type": "code", "id": "qYLk1WXqqCQs"}
import random
train, valid = train.split(split_ratio = 0.8, random_state=random.seed(SEED))

# + {"colab": {}, "colab_type": "code", "id": "Cy1G4eBPqGC1"}
TEXT.build_vocab(train, max_size=25000, vectors="glove.6B.100d")
LABEL.build_vocab(train)

# + {"colab": {}, "colab_type": "code", "id": "dtFRnaJaqJan"}
BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train, valid, test), 
    batch_size=BATCH_SIZE,
    device=device)

# + {"colab": {}, "colab_type": "code", "id": "V0HueLN0qRIo"}
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


# + {"colab": {}, "colab_type": "code", "id": "Yyun-1XHqVG_"}
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

lstmmodel = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 34}, "colab_type": "code", "id": "l4hMdH62qYRE", "outputId": "3227b997-622d-4d30-a5d0-c8e8f3adc8fd"}
pretrained_embeddings = TEXT.vocab.vectors

print(pretrained_embeddings.shape)

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 136}, "colab_type": "code", "id": "x-wIZDq9qatW", "outputId": "f5932953-633e-44ed-e438-32d53ae8800b"}
lstmmodel.embedding.weight.data.copy_(pretrained_embeddings)

# + {"colab": {}, "colab_type": "code", "id": "ZEjB1hFUqeU-"}
import torch.optim as optim

optimizer = optim.Adam(lstmmodel.parameters())
# optimizer = optim.SGD
criterion = nn.BCEWithLogitsLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lstmmodel = lstmmodel.to(device)
criterion = criterion.to(device)

# + {"colab": {}, "colab_type": "code", "id": "ZwE7tyNgqhvw"}
import torch.nn.functional as F

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(F.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum()/len(correct)
    return acc


# + {"colab": {}, "colab_type": "code", "id": "-UbCUFcvqknx"}
def train_model(model, train_iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in train_iterator:
        
        optimizer.zero_grad()
        
        predictions = model(batch.text).squeeze(1)
        
        loss = criterion(predictions, batch.label)
        
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(train_iterator), epoch_acc / len(train_iterator)


# + {"colab": {}, "colab_type": "code", "id": "8LDuwT5sqnW6"}
def evaluate_model(model, iterator, criterion):
    
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


# + {"colab": {"base_uri": "https://localhost:8080/", "height": 1133}, "colab_type": "code", "id": "p7maMAxXqp8V", "outputId": "17ea1ad3-8049-474c-a11d-db8b0dcc41fc"}
N_EPOCHS = 5

for epoch in range(N_EPOCHS):

    train_loss, train_acc = train_model(lstmmodel, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate_model(lstmmodel, valid_iterator, criterion)
    
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc*100:.2f}%, Val. Loss: {valid_loss:.3f}, Val. Acc: {valid_acc*100:.2f}%')

# + {"colab": {}, "colab_type": "code", "id": "eZ92JkDqquF-"}
test_loss_rnn, test_acc_rnn = evaluate_model(lstmmodel, test_iterator, criterion)

print(f'| Test Loss: {test_loss_rnn:.3f} | Test Acc: {test_acc_rnn*100:.2f}% |')

# + {"colab": {}, "colab_type": "code", "id": "aUSJjsn2q3-z"}
torch.save(lstmmodel.state_dict(),'model_lstm_new.pth')
