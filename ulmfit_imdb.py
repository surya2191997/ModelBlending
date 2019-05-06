import numpy as np
import torch 

print("Torch version ")
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.backends.cudnn.enabled)
import argparse
import fastai
from fastai import *
from fastai.text import *
import html
# %TODO MOVE TO README
# ! wget -nH -r -np -P {PATH} http://files.fast.ai/models/wt103/

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default=".")
    parser.add_argument("--valid", type=str, default=".")
    parser.add_argument("--itos", type=str, default=".")
    args = parser.parse_args()
    return args


# Start of the program 
args = get_args()


trn_lm = np.load(args.train)
val_lm = np.load(args.valid)

em_sz,nh,nl = 400,1150,3
PATH=Path('data/aclImdb/')
PRE_PATH = 'models'/'wt103'
PRE_LM_PATH = PRE_PATH/'fwd_wt103.h5'
itos = pickle.load(open(args.itos, 'rb'))
vs = len(itos)
wgts = torch.load(PRE_LM_PATH, map_location=lambda storage, loc: storage)
enc_wgts = to_np(wgts['0.encoder.weight'])
row_m = enc_wgts.mean(0)
itos2 = pickle.load((PRE_PATH/'itos_wt103.pkl').open('rb'))
stoi2 = collections.defaultdict(lambda:-1, {v:k for k,v in enumerate(itos2)})
new_w = np.zeros((vs, em_sz), dtype=np.float32)
for i,w in enumerate(itos):
    r = stoi2[w]
    new_w[i] = enc_wgts[r] if r>=0 else row_m

wgts['0.encoder.weight'] = T(new_w)
wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(new_w))
wgts['1.decoder.weight'] = T(np.copy(new_w))

wd=1e-7
bptt=70
bs=52
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

trn_dl = LanguageModelLoader(np.concatenate(trn_lm), bs, bptt)
val_dl = LanguageModelLoader(np.concatenate(val_lm), bs, bptt)
md = LanguageModelData(PATH, 1, vs, trn_dl, val_dl)
drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*0.7

learner= md.get_model(opt_fn, em_sz, nh, nl, 
    dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])

learner.metrics = [accuracy]
learner.freeze_to(-1)

learner.model.load_state_dict(wgts)
lr=1e-3
lrs = lr
learner.fit(lrs/2, 1, wds=wd, use_clr=(32,2), cycle_len=1)

learner.save('lm_last_ft')
learner.unfreeze()
learner.lr_find(start_lr=lrs/10, end_lr=lrs*10, linear=True)
learner.fit(lrs, 1, wds=wd, use_clr=(20,10), cycle_len=15)

learner.save('lm1')
learner.save_encoder('lm1_enc')