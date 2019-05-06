from fastai.text import *
import numpy as np 
import html 
import torch
import pandas as pd
import re

BOS = 'xbos'
FLD = 'xfld'
# add comand line argument to read in the data
PATH=Path('data/aclImdb/')

CLASSES = ['neg', 'pos', 'unsup']

def get_texts(path):
    texts,labels = [],[]
    for idx,label in enumerate(CLASSES):
        for fname in (path/label).glob('*.*'):
            texts.append(fname.open('r', encoding='utf-8').read())
            labels.append(idx)
    return np.array(texts),np.array(labels)

trn_texts,trn_labels = get_texts(PATH/'train')
val_texts,val_labels = get_texts(PATH/'test')
col_names = ['labels','text']

np.random.seed(42)
trn_idx = np.random.permutation(len(trn_texts))
val_idx = np.random.permutation(len(val_texts))

trn_texts = trn_texts[trn_idx]
val_texts = val_texts[val_idx]

trn_labels = trn_labels[trn_idx]
val_labels = val_labels[val_idx]

df_trn = pd.DataFrame({'text':trn_texts, 'labels':trn_labels}, columns=col_names)
df_val = pd.DataFrame({'text':val_texts, 'labels':val_labels}, columns=col_names)

df_trn[df_trn['labels']!=2].to_csv('cf_train.csv', header=False, index=False)
df_val.to_csv('cf_test.csv', header=False, index=False)

('classes.txt').open('w', encoding='utf-8').writelines(f'{o}\n' for o in CLASSES)

# split the train and validation data

trn_texts,val_texts = sklearn.model_selection.train_test_split(np.concatenate([trn_texts,val_texts]), test_size=0.1)
print("len(trn_texts), len(val_texts)")
print(len(trn_texts), len(val_texts))


df_trn = pd.DataFrame({'text':trn_texts, 'labels':[0]*len(trn_texts)}, columns=col_names)
df_val = pd.DataFrame({'text':val_texts, 'labels':[0]*len(val_texts)}, columns=col_names)

df_trn.to_csv('lm_train.csv', header=False, index=False)
df_val.to_csv('lm_test.csv', header=False, index=False)

# Tokenization 

chunksize = 24000

re1 = re.compile(r'  +')

def fixup(x):
    '''
    Clean Text of unidentified symbols
    '''
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace('nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace('<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))

def get_texts(df, n_lbls=1):
    labels = df.iloc[:,range(n_lbls)].values.astype(np.int64)
    texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
    for i in range(n_lbls+1, len(df.columns)): texts += f' {FLD} {i-n_lbls} ' + df[i].astype(str)
    texts = list(texts.apply(fixup).values)

    tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
    return tok, list(labels)

def get_all(df, n_lbls):
    tok, labels = [], []
    for i, r in enumerate(df):
        print(i)
        tok_, labels_ = get_texts(r, n_lbls)
        tok += tok_;
        labels += labels_
    return tok, labels

df_trn = pd.read_csv('lm_train.csv', header=None, chunksize=chunksize)
df_val = pd.read_csv('lm_test.csv', header=None, chunksize=chunksize)

tok_trn, trn_labels = get_all(df_trn, 1)
tok_val, val_labels = get_all(df_val, 1)

np.save('tok_trn.npy', tok_trn)
np.save('tok_val.npy', tok_val)
freq = Counter(p for o in tok_trn for p in o)

# Change Vocablary size to extremes and check the impact 
max_vocab = 60000
min_freq = 2

itos = [o for o,c in freq.most_common(max_vocab) if c>min_freq]
itos.insert(0, '_pad_')
itos.insert(0, '_unk_')

stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
print(len(itos))

trn_lm = np.array([[stoi[o] for o in p] for p in tok_trn])
val_lm = np.array([[stoi[o] for o in p] for p in tok_val])

np.save('trn_ids.npy', trn_lm)
np.save('val_ids.npy', val_lm)
pickle.dump(itos, open('itos.pkl', 'wb'))

print('testing')
trn_lm = np.load('trn_ids.npy')
val_lm = np.load('val_ids.npy')
itos = pickle.load(open('itos.pkl', 'rb'))
print('done testing')
print(len(trn_lm))