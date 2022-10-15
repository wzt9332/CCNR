import torch
from torch.nn.functional import interpolate
import random
import numpy as np
a = torch.zeros((4,1,64,64))
# for i in range(4):
#     a[:,:,i*16:i*16+16, 0:16] = i + 1
# for i in range(4):
#     a[:,:,i*16+7:i*16+10, 37:42] = i + 1

for b in range(4):
    for i in range(64):
        for j in range(64):
            a[b,:,i,j] = random.randint(0,6)




b = interpolate(a, (4,4), mode='bicubic')

aa = a[0,0].numpy()
bb = b[0,0].numpy()

out = np.zeros((4,4))
for i in range(4):
    for j in range(4):
        crop = aa[i*16:i*16+16, j*16:j*16+16]
        sort = sorted([(np.sum(crop == w), w) for w in set(crop.flat)])
        out[i,j] = sort[0][1]

print()
