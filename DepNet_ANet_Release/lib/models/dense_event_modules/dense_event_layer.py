import math
import numpy as np
import torch
import torch.nn as nn

def get_positional_encoding(d_model, idx):
    positional_encoding = torch.zeros((d_model,))  # (max_length, d_model)
    i = idx
    for j in range(d_model):
        if j % 2 == 0:
            positional_encoding[j] = math.sin(i / math.pow(10000, j / d_model))
        else:
            positional_encoding[j] = math.cos(i / math.pow(10000, (j - 1) / d_model))

#     positional_encoding = positional_encoding.unsqueeze(0)  # (1, max_length, d_model)

    return positional_encoding

class DoubleAttentionLayer(nn.Module):
    def __init__(self, in_channels, c_m, c_n,k =1 ):
        super(DoubleAttentionLayer, self).__init__()

        self.K           = k
        self.c_m = c_m
        self.c_n = c_n
        self.softmax     = nn.Softmax()
        self.in_channels = in_channels

        # self.convA = nn.Conv2d(in_channels, c_m, 1)
        # self.convB = nn.Conv2d(in_channels, c_n, 1)
        # self.convV = nn.Conv2d(in_channels, c_n, 1)

        self.convA = nn.Conv3d(in_channels, c_m, 1)
        self.convB = nn.Conv3d(in_channels, c_n, 1)
        self.convV = nn.Conv3d(in_channels, c_n, 1)

    def forward(self, x):

        b, c, d, h, w = x.size()

        assert c == self.in_channels,'input channel not equal!'
        #assert b//self.K == self.in_channels, 'input channel not equal!'

        A = self.convA(x)
        B = self.convB(x)
        V = self.convV(x)

        batch = int(b/self.K)

        tmpA = A.view( batch, self.K, self.c_m, d*h*w ).permute(0,2,1,3).view( batch, self.c_m, self.K*d*h*w )
        tmpB = B.view( batch, self.K, self.c_n, d*h*w ).permute(0,2,1,3).view( batch*self.c_n, self.K*d*h*w )
        tmpV = V.view( batch, self.K, self.c_n, d*h*w ).permute(0,1,3,2).contiguous().view( int(b*d*h*w), self.c_n )

        softmaxB = self.softmax(tmpB).view( batch, self.c_n, self.K*d*h*w ).permute( 0, 2, 1)  #batch, self.K*h*w, self.c_n
        softmaxV = self.softmax(tmpV).view( batch, self.K*d*h*w, self.c_n ).permute( 0, 2, 1)  #batch, self.c_n  , self.K*h*w

        tmpG     = tmpA.matmul( softmaxB )      #batch, self.c_m, self.c_n
        tmpZ     = tmpG.matmul( softmaxV ) #batch, self.c_m, self.K*h*w
        tmpZ     = tmpZ.view(batch, self.c_m, self.K,d*h*w).permute( 0, 2, 1,3).view( int(b), self.c_m, d, h, w )
        return tmpZ