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
from layers.last_fusion import (WithoutFusion, V_DAB, ChannelAttention,
                                GroupConv, STARm, STARc, ShuffleConv,
                                TSConv2d, TSDeformConv2d)

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

        self.fusion_layers = self._build_fusion_layers()

    def _build_fusion_layers(self, layer_name=None):
        fusion_layers_dict = {
            'WithoutFusion': WithoutFusion,
            'V_DAB': V_DAB,
            'ChannelAttention': ChannelAttention,
            'GroupConv': GroupConv,
            'STARm': STARm,
            'STARc': STARc,
            'ShuffleConv': ShuffleConv,
            'TSConv2d': TSConv2d,
            'TSDeformConv2d': TSDeformConv2d,
        }
        if layer_name is None:
            fusion_layer = fusion_layers_dict[self.configs.last_fusion](self.configs).float()
        else:
            fusion_layer = fusion_layers_dict[layer_name](self.configs).float()

        if self.configs.use_gpu:
            device = torch.device('cuda:{}'.format(0))
            fusion_layer.to(device=device)
        return fusion_layer

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
            x_enc_source = x_enc_source.masked_fill(mask_source == 0, 0)

            dec_out = self.pretrained_models[source_idx](x_enc_source, mask_source)

            # Store in dec_outs
            dec_outs[..., source_idx] = dec_out

        #print("The shape of dec_outs is: ", dec_outs.shape)
        B, L, M, S = dec_outs.shape
        dec_outs = self.fusion_layers(dec_outs)
        return dec_outs