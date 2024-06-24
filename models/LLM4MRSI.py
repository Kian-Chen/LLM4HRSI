from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from transformers import GPT2ForSequenceClassification
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import BertTokenizer, BertModel
from einops import rearrange


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.is_ln = configs.ln
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        self.patch_num = (configs.seq_len + self.pred_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1

        self.gpt2 = GPT2Model.from_pretrained('./gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            elif 'mlp' in name and configs.mlp == 1:
                param.requires_grad = True
            else:
                param.requires_grad = False

        if configs.use_gpu:
            device = torch.device('cuda:{}'.format(0))
            self.gpt2.to(device=device)

        self.hidden_size = self.gpt2.config.hidden_size
        self.in_layer = nn.Linear(configs.c_out * 2, self.hidden_size)
        self.s_in_layer = nn.Linear(configs.seq_len * 2, self.hidden_size)
        self.weight_layer = nn.Linear(configs.c_out * 2, configs.c_out)

        self.mlp = nn.Sequential(
            nn.Linear(configs.c_out * 2, configs.c_out),
            nn.ReLU(),
            nn.Linear(configs.c_out, configs.c_out),
        )
        self.drop = nn.Dropout(0.05)

        self.ln_proj = nn.LayerNorm(self.hidden_size * 2)
        self.s_ln_proj = nn.LayerNorm(self.hidden_size)
        self.out_layer = nn.Linear(self.hidden_size * 2, configs.c_out, bias=True)
        self.s_out_layer = nn.Linear(self.hidden_size, self.seq_len, bias=True)

        self.nvars = 3
        self.ffn_pw1 = nn.Conv1d(in_channels=self.nvars * self.c_out, out_channels=self.nvars * self.d_ff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=self.c_out)
        self.ffn_act = nn.GELU()
        self.ffn_pw2 = nn.Conv1d(in_channels=self.nvars * self.d_ff, out_channels=self.nvars * self.c_out, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=self.c_out)
        self.ffn_drop1 = nn.Dropout(0.1)
        self.ffn_drop2 = nn.Dropout(0.1)

    def forward(self, x_enc, mask=None):
        dec_out = self.imputation(x_enc, mask)
        return dec_out

    def imputation(self, x_enc, mask):
        # * B is batch size
        # * L is the len of the seq
        # * M is the spatial data nums
        # * S is the source nums
        B, L, M, S = x_enc.shape
        #print("The shape of x_enc is: ", x_enc.shape)
        
        dec_outs = torch.zeros(B, L, M, S, device=x_enc.device)

        for source_idx in range(S):
            # Select right now data
            x_enc_source = x_enc[..., source_idx]
            mask_source = mask[..., source_idx]

            #print("The shape of x_enc_source is: ", x_enc_source.shape)
            #print("The shape of mask_source is: ", mask_source.shape)
            x_enc_source = x_enc_source.masked_fill(mask_source == 0, 0)

            # temporal
            x_m_enc = torch.cat([x_enc_source, mask_source], dim=-1)
            #print("Shape before in_layer: ", x_m_enc.shape)
            x_m_enc = self.in_layer(x_m_enc)
            #print("Shape after in_layer: ", x_m_enc.shape)
            x_m_output = self.gpt2(inputs_embeds=x_m_enc).last_hidden_state

            # backward imputation
            re_x_m_enc = torch.flip(torch.cat([x_enc_source, mask_source], dim=-1), dims=(0, 1))
            re_x_m_enc = self.in_layer(re_x_m_enc)
            re_m_outputs = self.gpt2(inputs_embeds=re_x_m_enc).last_hidden_state
            re_dec_out = torch.flip(re_m_outputs, dims=(0, 1))

            tem_out = torch.cat([x_m_output, re_dec_out], dim=2)
            tem_out = self.ln_proj(tem_out)
            tem_out = self.out_layer(tem_out)

            # spatial
            x_s_enc = rearrange(tem_out, 'b l m -> b m l')
            s_mask = rearrange(mask_source, 'b l m -> b m l')
            x_m_s_enc = torch.cat([x_s_enc, s_mask], dim=-1)
            x_m_s_enc = self.s_in_layer(x_m_s_enc)
            x_m_s_output = self.gpt2(inputs_embeds=x_m_s_enc).last_hidden_state
            s_outputs = self.s_ln_proj(x_m_s_output)
            s_dec_out = self.s_out_layer(s_outputs)
            spa_out = rearrange(s_dec_out, 'b m l -> b l m')

            dec_out = torch.cat([tem_out, spa_out], dim=2)
            dec_out = self.weight_layer(dec_out)

            # Store in dec_outs
            dec_outs[..., source_idx] = dec_out
        
        #print("The shape of dec_outs is: ", dec_outs.shape)
        B, L, M, S = dec_outs.shape
        # Flatten the last two dims(flatten the vars) 
        dec_outs = dec_outs.view(B, L, M * S)

        # Apply ConvFFN
        dec_outs = dec_outs.transpose(1, 2)
        dec_outs = self.ffn_pw1(dec_outs)
        dec_outs = self.ffn_act(dec_outs)
        dec_outs = self.ffn_drop1(dec_outs)
        dec_outs = self.ffn_pw2(dec_outs)
        dec_outs = self.ffn_drop2(dec_outs)

        # Reshape to the original shape
        dec_outs = dec_outs.transpose(1, 2)
        dec_outs = dec_outs.view(B, L, M, S)
        return dec_outs