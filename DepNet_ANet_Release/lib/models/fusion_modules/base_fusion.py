import torch
from torch import nn
import torch.nn.functional as F
from IPython import embed

class BaseFusion(nn.Module):

    def __init__(self, cfg):
        super(BaseFusion, self).__init__()
        self.cfg = cfg
        hidden_size = cfg.HIDDEN_SIZE
        txt_input_size = cfg.TXT_INPUT_SIZE
        txt_hidden_size = cfg.TXT_HIDDEN_SIZE
        self.textual_encoder = nn.LSTM(txt_input_size, txt_hidden_size//2 if cfg.LSTM.BIDIRECTIONAL else txt_hidden_size,
                                       num_layers=cfg.LSTM.NUM_LAYERS, bidirectional=cfg.LSTM.BIDIRECTIONAL, batch_first=True)
        self.tex_linear = nn.Linear(txt_hidden_size, hidden_size)
        self.vis_conv = nn.Conv2d(hidden_size, hidden_size, 1, 1)

    def forward(self, textual_input, textual_mask, sentence_mask, map_h, map_mask):
        # textual_input:(b,8,seq,300)
        # textual_mask: (b,seq,1)
        # map_h: (b, 512, 64, 64)
        # map_mask: (b,1,64,64)
        self.textual_encoder.flatten_parameters()
        batch_size = textual_input.size(0)
        seq = textual_input.size(2)

        # To LSTM
        # Single sentence:
        # txt_h = self.textual_encoder(textual_input)[0] * textual_mask # txt_h:(b,seq,512)
        textual_input = textual_input.view((batch_size * 8, seq, 300))  # textual_input: (b,8,seq,300)->(b*8,seq,300)
        txt_h = self.textual_encoder(textual_input)[0]  # txt_h:(b*8,seq,512)
        txt_h = txt_h.view((batch_size, 8, seq, 512))  # txt_h:(b,8,seq,512)
        txt_h = txt_h * textual_mask  # txt_h:(b,8,seq,512), textual_mask:(b,8,seq,1)
        txt_h = txt_h.view((batch_size * 8, seq, 512))
        textual_mask = textual_mask.view((batch_size * 8, seq, 1))

        # get LSTM's last output
        # Single sentence:
        # txt_h = torch.stack([txt_h[i][torch.sum(mask).long() - 1] for i, mask in enumerate(textual_mask)]) # txt_h:(b,512)
        txt_h_ = torch.zeros(batch_size * 8, 512).cuda()  # txt_h_ (b*8,512)
        for i, mask in enumerate(textual_mask):
            cur_seq = torch.sum(mask).long()
            if cur_seq > 0:
                txt_h_[i] = txt_h[i][cur_seq - 1]

        # Single sentence:
        # txt_h = self.tex_linear(txt_h)[:,:,None,None] # txt_h:(b,512,1,1)
        txt_h = self.tex_linear(txt_h_)
        txt_h = txt_h.view(batch_size, 8, 512)
        txt_h = txt_h[:, :, :, None, None]

        # fusion_layer: Vision
        # Single sentence:
        # map_h = self.vis_conv(map_h)  # map_h: (b, 512, 64, 64)
        map_h = self.vis_conv(map_h)  # map_h: (b, 512, 64, 64)
        map_h = map_h.unsqueeze(1).repeat((1, 8, 1, 1, 1))  # map_h (b,8,512,64,64)
        map_mask = map_mask.unsqueeze(1).repeat((1, 8, 1, 1, 1))  # map_mask (b,8,1,64,64)

        # fusion_layer: Fusion
        # Single sentence:
        # fused_h = F.normalize(txt_h * map_h) * map_mask
        fused_h = F.normalize(txt_h * map_h, dim=2) * map_mask # fused_h (b,8,512,64,64)
        return fused_h, map_mask

