from torch import nn
from core.config import config
import models.frame_modules as frame_modules
import models.prop_modules as prop_modules
import models.map_modules as map_modules
import models.fusion_modules as fusion_modules
import models.bmn_modules as bmn_layer
from IPython import embed

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



class TAN(nn.Module):
    def __init__(self):
        super(TAN, self).__init__()

        self.frame_layer = getattr(frame_modules, config.TAN.FRAME_MODULE.NAME)(config.TAN.FRAME_MODULE.PARAMS)
        self.prop_layer = getattr(prop_modules, config.TAN.PROP_MODULE.NAME)(config.TAN.PROP_MODULE.PARAMS)
        self.fusion_layer = getattr(fusion_modules, config.TAN.FUSION_MODULE.NAME)(config.TAN.FUSION_MODULE.PARAMS)
        self.map_layer = getattr(map_modules, config.TAN.MAP_MODULE.NAME)(config.TAN.MAP_MODULE.PARAMS)
        self.pred_layer = nn.Conv2d(config.TAN.PRED_INPUT_SIZE, 1, 1, 1)
        self.bmn_layer = bmn_layer.BMN()
        self.nlblock = DoubleAttentionLayer(512, 512, 512)

        d_model = 128
        self.pos_feat = torch.zeros((d_model*3, 8, 32, 32))
        for k in range(8):
            for i in range(32):
                for j in range(i, 32):
                    self.pos_feat[0:d_model, k, i, j] = get_positional_encoding(d_model, k+1)
                    self.pos_feat[d_model:(d_model*2), k, i, j] = get_positional_encoding(d_model, i+1)
                    self.pos_feat[(d_model*2):(d_model*3), k, i, j] = get_positional_encoding(d_model, j+1)
        self.pos_feat = self.pos_feat.cuda()

    def forward(self, textual_input, textual_mask, sentence_mask, visual_input):
        # visual_input: (b,256, input_size) i.e.(32,256,500)
        # textual_input: (b,K,seq,300) tensor
        # textual_mask: (b,K,seq,1) tensor
        # sentence_mask: (b,K,1) tensor
        batch_size = textual_input.size(0)
        seq = textual_input.size(2)

        # identical as single
        vis_h = self.frame_layer(visual_input.transpose(1, 2)) #vis_h (b,512,64)
        map_h, map_mask = self.prop_layer(vis_h) #map_h (b,512,64,64) map_mask (b,1,64,64)
        map_h = self.bmn_layer(vis_h)
        map_size = map_h.size(3)
        # different due to dense
        fused_h, map_mask = self.fusion_layer(textual_input, textual_mask, sentence_mask, map_h, map_mask)
        # fused_h (b, K,512,64,64)
        fused_h = fused_h.view(batch_size * 8, 512, map_size, map_size)  # fused_h (b*8,512,64,64)

        map_mask = map_mask.view(batch_size * 8, 1, map_size, map_size)
        sentence_mask = sentence_mask.view(batch_size * 8, 1)
        sentence_mask = sentence_mask[:, :, None, None]  # sentence_mask (b*8,1, 1, 1)
        map_mask = map_mask * sentence_mask

        # different due to conv3d
        # map_mask (b*8,1,64,64) -> (b,1,8,64,64)
        # fused_h  (b*8,512,64,64) -> (b,512,8,64,64)
        map_mask = map_mask.view(batch_size, 8, 1, map_size, map_size)
        map_mask = map_mask.permute(0, 2, 1, 3, 4)
        fused_h = fused_h.view(batch_size, 8, 512, map_size, map_size)
        fused_h = fused_h.permute(0, 2, 1, 3, 4)
        fused_h = torch.cat((self.pos_feat.repeat(fused_h.size(0), 1, 1, 1, 1), fused_h), dim=1)
        # embed()
        fused_h = self.map_layer(fused_h, map_mask) #fused_h (b,512,8,64,64)
        fused_h = self.nlblock(fused_h) * map_mask
        # different due to conv3d
        # map_mask (b,1,8,64,64)  -> (b*8,1,64,64)
        # fused_h  (b,512,8,64,64) -> (b*8,512,64,64)
        fused_h = fused_h.permute(0, 2, 1, 3, 4)
        map_mask = map_mask.permute(0, 2, 1, 3, 4)

        fused_h = fused_h.contiguous().view(batch_size*8, 512, map_size, map_size)
        map_mask = map_mask.contiguous().view(batch_size*8, 1, map_size, map_size)
        prediction = self.pred_layer(fused_h)  #prediction (b*K,1,64,64)

        prediction = prediction * map_mask #prediction (b*K,1,64,64)

        prediction = prediction.view(batch_size, 8, 1, map_size, map_size)
        map_mask = map_mask.view(batch_size, 8, 1, map_size, map_size)
        return prediction, map_mask

    def extract_features(self, textual_input, textual_mask, visual_input):
        vis_h = self.frame_layer(visual_input.transpose(1, 2))
        map_h, map_mask = self.prop_layer(vis_h)

        fused_h = self.fusion_layer(textual_input, textual_mask, map_h, map_mask)
        fused_h = self.map_layer(fused_h, map_mask)
        prediction = self.pred_layer(fused_h) * map_mask

        return fused_h, prediction, map_mask
