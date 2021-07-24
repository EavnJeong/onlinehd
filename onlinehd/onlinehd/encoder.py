import math

import torch
import numpy as np

class Encoder(object):
    def __init__(self, features : int, dim : int = 4000):
        self.dim = dim
        self.features = features
        
    def __call__(self, x : torch.Tensor, basis : torch.Tensor, base : torch.Tensor, alpha):
        n = x.size(0)
        bsize = math.ceil(0.01*n)
        h = torch.empty(n, self.dim, device=x.device, dtype=x.dtype)
        self.temp = torch.empty(bsize, self.dim, device=x.device, dtype=x.dtype)
        
        # we need batches to remove memory usage
        '''
            x[i:i+bsize].shape        : 525, 784
            basis.T.shape             : 784, 10000
            base.shape                : 10000
            self.temp.shape           : 525, 10000
            self.h[i:i+bsize].shape   : 525, 10000
        '''
        for i in range(0, n, bsize):
            torch.matmul(x[i:i+bsize], basis.T, out=self.temp)
            torch.add(self.temp, base, out=h[i:i+bsize])
            h[i:i+bsize].div_(alpha) # it must be defined before traning.
            h[i:i+bsize].cos_()
        
        return h
    
    
    
#     def to(self, *args):
#         self.basis = self.basis.to(*args)
#         self.base = self.base.to(*args)
#         return self
