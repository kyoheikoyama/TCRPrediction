print('in check.py')

import torch, os

print('torch.version.cuda', torch.version.cuda)

print('torch.cuda.is_available()',  torch.cuda.is_available())

print(os.listdir('../../../'))
print(os.listdir('../../user_work'))
print(os.listdir('../../user_work/tcrpred'))
