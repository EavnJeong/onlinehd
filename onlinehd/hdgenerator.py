import math

import torch
import numpy as np

class HDGenerator(object):
    def __init__(self, features : int, dim : int = 4000):
        self.dim = dim
        self.features = features
        self.scope = {
                       "0" : torch.empty(2, self.dim), 
                       "1" : torch.empty(2, self.dim),
                       "2" : torch.empty(2, self.dim),
                       "3" : torch.empty(2, self.dim),
                       "4" : torch.empty(2, self.dim),
                       "5" : torch.empty(2, self.dim),
                       "6" : torch.empty(2, self.dim),
                       "7" : torch.empty(2, self.dim),
                       "8" : torch.empty(2, self.dim),
                       "9" : torch.empty(2, self.dim)  }
    
    def __call__(self, size : int, target : str, basis : torch.Tensor, base : torch.Tensor, alpha):
        ratio = torch.rand((size, self.dim))
        
        return self.decode((self.scope[target][1] - self.scope[target][0]) * ratio + self.scope[target][0], basis, base, alpha)
    
    def decode(self, h : torch.Tensor, basis : torch.Tensor, base : torch.Tensor, alpha):
        '''
        h = [52500, 10000]
        basis.T = [10000, 784]
        '''
        n = h.size(0)
        bsize = math.ceil(0.01*n)
        pred_x = torch.empty(n, self.features)
        self.temp = torch.empty(bsize, self.dim)
        
        for i in range(0, n, bsize):
            h[i:i+bsize].acos_()
            h[i:i+bsize].mul_(alpha)
            torch.sub(h[i:i+bsize], base, out=self.temp)
            basis_pinv = torch.linalg.pinv(basis.T)
            torch.matmul(self.temp, basis_pinv, out=pred_x[i:i+bsize])
            
        return pred_x
    
    def fit(self,
            h : torch.Tensor,
            y : torch.Tensor):

        for data, lbl in zip(h, y):
            lbl = str(lbl.item())
            dic_min = self.scope[lbl][0] 
            dic_max = self.scope[lbl][1]

            min_scope = torch.tensor([data[i] if data[i] < dic_min[i] else dic_min[i] for i in range(self.dim)])
            max_scope = torch.tensor([data[i] if data[i] > dic_min[i] else dic_min[i] for i in range(self.dim)])
            self.scope[lbl] = torch.stack((min_scope, max_scope), 0)
        
        return self
    
    def to(self, *args):
        self.basis = self.basis.to(*args)
        self.scope = self.scope.to(*args)
        return self
    