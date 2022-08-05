import torch
from collections import OrderedDict

model = OrderedDict()
old = torch.load("pytorch_model_old.bin", map_location=torch.device('cpu'))
for k, v in old.items():
    if k[:12] == 'decoder.bert':
        nk = 'bert' + k[12:]
        nk = nk.replace('gamma', 'weight')
        nk = nk.replace('beta', 'bias')
        model[nk] = v
    elif k[:11] == 'decoder.cls':
        nk = 'cls' + k[11:]
        model[nk] = v

torch.save(model, "pytorch_model.bin")