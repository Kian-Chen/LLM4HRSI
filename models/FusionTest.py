from typing import Optional
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from collections import OrderedDict
from transformers import GPT2ForSequenceClassification
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import BertTokenizer, BertModel
from einops import rearrange
from models import LLM4HRSI


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
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

        # Load pretrained models
        self.pretrained_models = self.load_pretrained_models('pretrained/', configs.use_gpu)

        self.nvars = len(configs.source_names)
        self.ffn_pw1 = nn.Conv1d(in_channels=self.nvars * self.c_out, out_channels=self.nvars * self.d_ff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=self.c_out)
        self.ffn_act = nn.GELU()
        self.ffn_pw2 = nn.Conv1d(in_channels=self.nvars * self.d_ff, out_channels=self.nvars * self.c_out, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=self.c_out)
        self.ffn_drop1 = nn.Dropout(0.1)
        self.ffn_drop2 = nn.Dropout(0.1)

        self.final_linear = nn.Linear(self.nvars * self.c_out, self.nvars * self.c_out)


    def load_pretrained_models(self, pretrained_dir, use_gpu):
        pretrained_models = []

        # Get list of files in the pretrained directory
        mask_rate = self.configs.mask_rate
        model_files = sorted(os.listdir(pretrained_dir))

        for model_file in model_files:
            if f"mask_{mask_rate}_" in model_file and model_file.endswith(self.configs.pretrain_postfix):
                model_path = os.path.join(pretrained_dir, model_file)
                model = self.load_model(model_path, use_gpu)
                pretrained_models.append(model)

        return pretrained_models

    def load_model(self, model_path, use_gpu):
        model = LLM4HRSI.Model(self.configs).float()

        map_location = torch.device('cuda:0') if use_gpu else torch.device('cpu')

        state_dict = torch.load(model_path, map_location=map_location)

        # Remove 'module.' which generated in multi-gpu
        if 'module.' in list(state_dict.keys())[0]:
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)

        # Freeze the model parameters
        for param in model.parameters():
            param.requires_grad = False

        if use_gpu:
            device = torch.device('cuda:0')
            model.to(device)

        if self.configs.use_multi_gpu and self.configs.use_gpu:
            model = nn.DataParallel(model, device_ids=self.configs.device_ids)

        return model

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

            dec_out = self.pretrained_models[source_idx](x_enc_source, mask_source)

            # Store in dec_outs
            dec_outs[..., source_idx] = dec_out
        
        #print("The shape of dec_outs is: ", dec_outs.shape)
        B, L, M, S = dec_outs.shape
        # Flatten the last two dims(flatten the vars)
        dec_outs = dec_outs.view(B, L, M * S)

        # Apply ConvFFN
        dec_outs = dec_outs.transpose(1, 2)
        residual = dec_outs
        dec_outs = self.ffn_pw1(dec_outs)
        dec_outs = self.ffn_act(dec_outs)
        dec_outs = self.ffn_drop1(dec_outs)
        dec_outs = self.ffn_pw2(dec_outs)
        dec_outs = self.ffn_drop2(dec_outs) + residual

        # Reshape to the original shape
        dec_outs = dec_outs.transpose(1, 2)
        dec_outs = self.final_linear(dec_outs)
        dec_outs = dec_outs.view(B, L, M, S)
        return dec_outs