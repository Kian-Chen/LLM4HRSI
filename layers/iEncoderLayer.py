import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.ttt import TTTConfig, TTTLinear, TTTMLP

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        return x


class EncoderLayer(nn.Module):
    def __init__(self, configs=None):
        super(EncoderLayer, self).__init__()

        self.seq_len = configs.seq_len
        self.len_dff = configs.len_dff
        self.d_ff = configs.d_ff
        self.n_heads = configs.n_heads
        self.d_lower = configs.d_lower
        self.d_inner = self.d_lower
        self.d_k = self.d_lower
        self.d_v = self.d_lower
        self.attn_dropout = configs.dropout

        self.is_invert = configs.is_invert

        if self.is_invert:
            self.layer_norm = nn.LayerNorm(self.len_dff)
            self.pos_ffn = PositionWiseFeedForward(self.len_dff, self.d_inner, configs.dropout)
            self.ttt_config = TTTConfig(
                hidden_size=self.len_dff,
            )
        else:
            self.layer_norm = nn.LayerNorm(self.d_ff)
            self.pos_ffn = PositionWiseFeedForward(self.d_ff, self.d_inner, configs.dropout)
            self.ttt_config = TTTConfig(
                hidden_size=self.d_ff
            )

        #self.seq_block = MultiHeadAttention(self.n_heads, self.d_ff, self.d_k, self.d_v, self.attn_dropout)
        if configs.ttt_style == 'TTTLinear':
            self.seq_block = TTTLinear(self.ttt_config)
        else:
            self.seq_block = TTTMLP(self.ttt_config)
        self.dropout = nn.Dropout(configs.dropout)

        if configs.use_gpu:
            device = torch.device('cuda:{}'.format(0))
            self.seq_block.to(device)
            self.pos_ffn.to(device)


    def forward(self, enc_input):

        residual = enc_input
        # here we apply LN before attention cal, namely Pre-LN, refer paper https://arxiv.org/abs/2002.04745
        enc_input = self.layer_norm(enc_input)
        #enc_output, attn_weights = self.seq_block(enc_input, enc_input, enc_input, attn_mask=mask_time)
        B, L = enc_input.shape[:2]
        attention_mask = torch.ones((B, L))
        position_ids = torch.arange(L, dtype=torch.long).unsqueeze(0).expand(B, L)
        enc_output, attn_weights = self.seq_block(hidden_states=enc_input,
                                                  attention_mask=attention_mask,
                                                  position_ids=position_ids)
        enc_output = self.dropout(enc_output)
        enc_output += residual

        enc_output = self.pos_ffn(enc_output)
        return enc_output, attn_weights