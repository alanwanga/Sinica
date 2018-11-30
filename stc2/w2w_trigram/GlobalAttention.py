"""
Global attention takes a matrix and a query vector. It
then computes a parameterized convex combination of the matrix
based on the input query.


        H_1 H_2 H_3 ... H_n
          q   q   q       q
            |  |   |       |
              \ |   |      /
                      .....
                  \   |  /
                          a

Constructs a unit mapping.
    $$(H_1 + H_n, q) => (a)$$
    Where H is of `batch x n x dim` and q is of `batch x dim`.

    The full def is  $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$.:

"""

import torch
import torch.nn as nn
import math
from IPython import embed

class GlobalAttention(nn.Module):
    def __init__(self, dim, method='concat'):
        super(GlobalAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_concatWa = nn.Linear(dim*2, dim, bias=False)
        self.linear_concatVa = nn.Linear(dim, 1, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None
        self.method = method
        print('use %s attention method'%(method))

    def applyMask(self, mask):
        self.mask = mask

    def general(self, input, context):
        targetT = self.linear_in(input).unsqueeze(2) # batch x dim x 1
        return torch.bmm(context, targetT).squeeze(2) # batch x 1

    def dot(self, input, context):
        targetT = input.unsqueeze(2) # batch x dim x 1
        return torch.bmm(context, targetT).squeeze(2) # batch x 1

    def concat(self, input, context):
        targetT = input.unsqueeze(1).expand(input.size(0),context.size(1),input.size(1)) # batch x sourceL x dim
        ht_hs = torch.cat((targetT,context),2) # batch x sourceL x dim*2
        size = ht_hs.size()[:2]
        ht_hs = ht_hs.view(size[0]*size[1], -1) # batch*sourceL x dim*2
        Wa_ht_hs = self.linear_concatWa(ht_hs) # batch*sourceL x dim
        tanh_Wa_ht_hs = self.tanh(Wa_ht_hs) 
        return self.linear_concatVa(tanh_Wa_ht_hs).view(size[0],1,-1).squeeze(1) # batch x 1

    def forward(self, input, context):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        if self.method == 'general':
            attn = self.general(input,context)
        if self.method == 'dot':
            attn = self.dot(input,context)
        if self.method == 'concat':
            attn = self.concat(input,context)
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL
        weightedContext = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        return weightedContext, attn
