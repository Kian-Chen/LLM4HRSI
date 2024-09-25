import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torchvision.ops import DeformConv2d

class WithoutFusion(nn.Module):
    def __init__(self, configs):
        super(WithoutFusion, self).__init__()

    def forward(self, x):
        return x


class VariableDeformableAttentionBlock(nn.Module):
    def __init__(self, d, l, s, k):
        super(VariableDeformableAttentionBlock, self).__init__()
        self.d = d
        self.l = l
        self.s = s
        self.k = k

        self.WQ = nn.Linear(d, d)
        self.WK = nn.Linear(d, d)
        self.WV = nn.Linear(d, d)
        self.Wi = nn.Linear(d, d)
        self.Wv = nn.Linear(d, d)

        self.conv1 = nn.Conv2d(in_channels=d, out_channels=d, kernel_size=k, padding=k // 2)
        self.conv2 = nn.Conv2d(in_channels=d, out_channels=2, kernel_size=1)
        self.tanh = nn.Tanh()
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def segment_sequence(self, x):
        B, L, d = x.shape
        pad_length = (self.s - (L - self.l) % self.s) % self.s
        x = F.pad(x, (0, 0, 0, pad_length))
        L = x.shape[1]
        n = (L - self.l) // self.s + 1
        patches = torch.zeros((B, n, self.l, d), device=x.device)
        for i in range(n):
            patches[:, i, :, :] = x[:, i * self.s:i * self.s + self.l, :]
        return patches, L + pad_length

    def bilinear_sample(self, patch, p):
        h, w = patch.shape[:2]
        x, y = p
        x0, x1 = int(x), min(int(x) + 1, h - 1)
        y0, y1 = int(y), min(int(y) + 1, w - 1)

        Ia = patch[x0, y0] if 0 <= x0 < h and 0 <= y0 < w else 0
        Ib = patch[x1, y0] if 0 <= x1 < h and 0 <= y0 < w else 0
        Ic = patch[x0, y1] if 0 <= x0 < h and 0 <= y1 < w else 0
        Id = patch[x1, y1] if 0 <= x1 < h and 0 <= y1 < w else 0

        wa = (x1 - x) * (y1 - y)
        wb = (x - x0) * (y1 - y)
        wc = (x1 - x) * (y - y0)
        wd = (x - x0) * (y - y0)

        return wa * Ia + wb * Ib + wc * Ic + wd * Id

    def bilinear_interpolation(self, Zp, delta_p):
        B, n, l, d = Zp.shape
        Zd = torch.zeros_like(Zp)

        for i in range(B):
            for j in range(n):
                for k in range(l):
                    base_pos = np.array([k, 0])  # 只考虑时间步
                    offset = delta_p[i, j, k]
                    p = base_pos + offset.detach().cpu().numpy()
                    Zd[i, j, k] = torch.tensor(self.bilinear_sample(Zp[i, j].detach().cpu().numpy(), p))

        return Zd.to(Zp.device)

    def forward(self, x):
        B, L, d = x.shape

        Zp, padded_length = self.segment_sequence(x)  # 形状为 (B, n, l, d)

        Qp = self.WQ(Zp)  # 形状为 (B, n, l, d)
        Qp_perm = Qp.permute(0, 3, 1, 2)

        offset = self.conv1(Qp_perm)
        offset = self.conv2(offset)
        offset = self.tanh(offset)
        delta_p = self.alpha * offset
        delta_p = delta_p.permute(0, 2, 3, 1)

        Zd = self.bilinear_interpolation(Zp, delta_p)

        Kd = self.WK(Zd)
        Vd = self.WV(Zd)

        scores = torch.matmul(Qp, Kd.transpose(-2, -1)) / torch.sqrt(torch.tensor(d, dtype=torch.float32))
        attn = F.softmax(scores, dim=-1)
        Ai = torch.matmul(attn, Vd)
        Ai = self.Wi(Ai)

        A = Ai.view(B, -1, d)

        # 处理最后输出以确保形状正确
        Zv = self.Wv(A)
        Zv = Zv[:, :L, :]  # 截取到原始长度
        return Zv

class V_DAB(nn.Module):
    def __init__(self, configs):
        super(V_DAB, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.c_out = configs.c_out
        self.d_ff = configs.d_ff
        self.nvars = len(configs.source_names)

        if configs.use_gpu:
            device = torch.device('cuda:{}'.format(0))
            self.device = device

        self.drop = nn.Dropout(0.05)

        self.v_dab = VariableDeformableAttentionBlock(d=self.c_out, l=10, s=5, k=3)
        if configs.use_gpu:
            self.v_dab.to(device=self.device)

        self.recover = nn.Linear(self.c_out, self.c_out * self.nvars)
        self.fc = nn.Linear(self.c_out * self.nvars, self.c_out)
        self.layer_norm = nn.LayerNorm(self.c_out * self.nvars)

        self.mlp = nn.Sequential(
            nn.Linear(configs.c_out * self.nvars, configs.d_ff * self.nvars),
            nn.ReLU(),
            nn.Linear(configs.d_ff * self.nvars, configs.c_out * self.nvars),
        )

    def forward(self, x_input):
        B, L, M, S = x_input.shape

        x_input = x_input.view(B, L, M*S)
        res = x_input

        x_input = self.fc(x_input)
        x_output = self.v_dab(x_input)
        x_output = self.recover(self.drop(x_output)) + res
        res = x_output

        x_output = self.layer_norm(x_output)
        x_output = self.mlp(x_output)
        x_output = self.drop(x_output) + res

        x_output = x_output.reshape(B, L, M, S)
        return x_output


class ChannelAttention(nn.Module):
    def __init__(self, configs):
        super(ChannelAttention, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.c_out = configs.c_out
        self.d_ff = configs.d_ff
        self.nvars = len(configs.source_names)

        if configs.use_gpu:
            device = torch.device('cuda:{}'.format(0))
            self.device = device

        # Layers in Channel Attention Module
        self.gelu = nn.GELU()
        self.fc = nn.Linear(self.c_out * self.nvars * 2, self.c_out)
        self.max_pool = nn.MaxPool1d(kernel_size=self.seq_len)
        self.avg_pool = nn.AvgPool1d(kernel_size=self.seq_len)
        self.reduce_ratio = 6
        self.reduce_channel = nn.Linear(self.nvars * self.c_out, self.nvars * self.c_out // self.reduce_ratio)
        self.inverse_reduce = nn.Linear(self.nvars * self.c_out // self.reduce_ratio, self.nvars * self.c_out)
        self.sigmoid = nn.Sigmoid()
        self.fuse_parallel = nn.Linear(self.nvars * self.c_out * 2, self.nvars * self.c_out)

    def forward(self, x_input):
        B, L, M, S = x_input.shape

        # Channel Attention Module
        x_input = x_input.view(B, L, M * S)
        x_input = rearrange(x_input, 'b l ms -> b ms l')
        x_max = self.max_pool(x_input)
        x_avg = self.avg_pool(x_input)
        x_max = rearrange(x_max, 'b ms l -> b l ms')
        x_avg = rearrange(x_avg, 'b ms l -> b l ms')
        x_input = rearrange(x_input, 'b ms l -> b l ms')
        x_max = self.gelu(self.reduce_channel(x_max))
        x_avg = self.gelu(self.reduce_channel(x_avg))
        x_signal = self.inverse_reduce(x_max) + self.inverse_reduce(x_avg)
        x_attention = self.sigmoid(x_signal)
        x_output = x_input * x_attention + x_input
        return x_output


class GroupConv(nn.Module):
    def __init__(self, configs):
        super(GroupConv, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.c_out = configs.c_out
        self.d_ff = configs.d_ff
        self.nvars = len(configs.source_names)

        if configs.use_gpu:
            device = torch.device('cuda:{}'.format(0))
            self.device = device

        self.ffn_pw1 = nn.Conv1d(in_channels=self.nvars, out_channels=96,
                                 kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=self.nvars)
        self.ffn_act = nn.GELU()
        self.ffn_pw2 = nn.Conv1d(in_channels=96 , out_channels=self.nvars,
                                 kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=self.nvars)
        self.ffn_drop1 = nn.Dropout(0.1)
        self.ffn_drop2 = nn.Dropout(0.1)

    def forward(self, x_input):
        B, L, M, S = x_input.shape

        x_input = rearrange(x_input, 'b l m s -> b s m l')
        x_input = x_input.reshape(B, S, M * L)
        #x_input = rearrange(x_input, 'b l ms -> b ms l')
        residual = x_input
        x_input = self.ffn_pw1(x_input)
        x_input = self.ffn_act(x_input)
        x_input = self.ffn_drop1(x_input)
        x_input = self.ffn_pw2(x_input)
        x_input = self.ffn_drop2(x_input) + residual

        # Reshape to the original shape
        x_output = x_input.reshape(B, S, M, L)

        x_output = rearrange(x_output, 'b s m l -> b l m s')

        return x_output


class STAR(nn.Module):
    def __init__(self, configs):
        super(STAR, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.c_out = configs.c_out
        self.d_ff = configs.d_ff
        self.nvars = len(configs.source_names)

        if configs.use_gpu:
            device = torch.device('cuda:{}'.format(0))
            self.device = device

        '''
        self.conv1 = nn.Conv1d(in_channels=self.c_out, out_channels=self.d_ff, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(in_channels=self.d_ff * self.nvars, out_channels=self.c_out, kernel_size=1, stride=1)
        self.conv3 = nn.Conv1d(in_channels=self.c_out * 2, out_channels=self.c_out, kernel_size=1, stride=1)
        '''
        self.seq_core = 64
        self.gen1 = nn.Linear(self.seq_len, self.seq_len)
        self.gen2 = nn.Linear(self.seq_len, self.seq_core)
        self.gen3 = nn.Linear(self.seq_len + self.seq_core, self.seq_len)
        self.gen4 = nn.Linear(self.seq_len, self.seq_len)

        self.drop = nn.Dropout(0.05)
        self.gelu = nn.GELU()

    def forward(self, x_input):
        B, L, M, S = x_input.shape
        residual = x_input.clone()

        x_input = rearrange(x_input, 'b l m s -> b (m s) l')

        combined_mean = F.gelu(self.gen1(x_input))
        combined_mean = self.gen2(combined_mean)

        # stochastic pooling
        if self.configs.is_training:
            ratio = F.softmax(combined_mean, dim=1)
            ratio = ratio.permute(0, 2, 1)
            ratio = ratio.reshape(-1, M*S)
            indices = torch.multinomial(ratio, 1)
            indices = indices.view(B, -1, 1).permute(0, 2, 1)
            combined_mean = torch.gather(combined_mean, 1, indices)
            combined_mean = combined_mean.repeat(1, M*S, 1)
        else:
            weight = F.softmax(combined_mean, dim=1)
            combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True).repeat(1, M*S, 1)

        # mlp fusion
        combined_mean_cat = torch.cat([x_input, combined_mean], -1)
        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat))
        combined_mean_cat = self.gen4(combined_mean_cat)
        x_output = combined_mean_cat
        '''
        x1 = x_input[..., 0]
        x2 = x_input[..., 1]
        x3 = x_input[..., 2]

        x1_1 = self.drop(self.gelu(self.conv1(x1)))
        x2_1 = self.drop(self.gelu(self.conv1(x2)))
        x3_1 = self.drop(self.gelu(self.conv1(x3)))

        x_hid = torch.cat([x1_1, x2_1, x3_1], dim=1)
        x_hid = self.drop(self.gelu(self.conv2(x_hid)))

        x1_cat = torch.cat([x1, x_hid], dim=1)
        x2_cat = torch.cat([x2, x_hid], dim=1)
        x3_cat = torch.cat([x3, x_hid], dim=1)

        x1 = self.drop(self.gelu(self.conv3(x1_cat)))
        x2 = self.drop(self.gelu(self.conv3(x2_cat)))
        x3 = self.drop(self.gelu(self.conv3(x3_cat)))

        # 不使用就地操作，创建一个新的张量来存储结果
        new_dec_outs = x_input.clone()
        new_dec_outs[..., 0] = x1
        new_dec_outs[..., 1] = x2
        new_dec_outs[..., 2] = x3

        x_output = new_dec_outs + residual
        '''

        x_output = rearrange(x_output, 'b ms l -> b l ms').view(B, L, M, S)
        return x_output + residual


class ShuffleConv(nn.Module):
    def __init__(self, configs):
        super(ShuffleConv, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.c_out = configs.c_out
        self.d_ff = configs.d_ff
        self.nvars = len(configs.source_names)

        if configs.use_gpu:
            device = torch.device('cuda:{}'.format(0))
            self.device = device

        # 1x1 Grouped Convolution
        self.gconv1 = nn.Conv1d(in_channels=self.nvars * self.c_out, out_channels=self.nvars * self.c_out,
                                kernel_size=1, stride=1, padding=0, groups=self.nvars)
        self.bn1 = nn.BatchNorm1d(self.nvars * self.c_out)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

        # 3x3 Depthwise Convolution (stride=1 to keep the same length)
        self.dwconv = nn.Conv1d(in_channels=self.nvars * self.c_out, out_channels=self.nvars * self.c_out,
                                kernel_size=3, stride=1, padding=1, groups=self.nvars * self.c_out)
        self.bn2 = nn.BatchNorm1d(self.nvars * self.c_out)

        # Second 1x1 Grouped Convolution
        self.gconv2 = nn.Conv1d(in_channels=self.nvars * self.c_out, out_channels=self.nvars * self.c_out,
                                kernel_size=1, stride=1, padding=0, groups=self.nvars)
        self.bn3 = nn.BatchNorm1d(self.nvars * self.c_out)

        # Dropout layers
        self.ffn_drop1 = nn.Dropout(0.1)
        self.ffn_drop2 = nn.Dropout(0.1)

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height = x.size()
        channels_per_group = num_channels // groups

        # reshape
        x = x.view(batchsize, groups, channels_per_group, height)
        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height)
        return x

    def forward(self, x_input):
        B, L, M, S = x_input.shape

        # Flatten the last two dims(flatten the vars)
        x_input = x_input.view(B, L, M * S)

        # Apply the improved ConvFFN
        x_input = rearrange(x_input, 'b l ms -> b ms l')
        residual = x_input

        # 1x1 Grouped Convolution -> BatchNorm -> ReLU
        x_input = self.gconv1(x_input)
        x_input = self.bn1(x_input)
        x_input = self.relu(x_input)

        # Channel Shuffle
        x_input = self.channel_shuffle(x_input, self.nvars)

        # 3x3 Depthwise Convolution -> BatchNorm
        x_input = self.dwconv(x_input)
        x_input = self.bn2(x_input)

        # 1x1 Grouped Convolution -> BatchNorm
        x_input = self.gconv2(x_input)
        x_input = self.bn3(x_input)

        # GELU activation + residual
        x_input = self.gelu(x_input)
        x_output = self.ffn_drop2(x_input) + residual
        x_output = rearrange(x_output, 'b ms l -> b l ms')

        x_output = x_output.view(B, L, M, S)
        return x_output


class TSConv2d(nn.Module):
    def __init__(self, configs):
        super(TSConv2d, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.c_out = configs.c_out
        self.d_ff = configs.d_ff
        self.nvars = len(configs.source_names)

        if configs.use_gpu:
            device = torch.device('cuda:{}'.format(0))
            self.device = device

        self.conv2d1 = nn.Conv2d(in_channels=self.nvars, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2d1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.2)
        self.conv2d2 = nn.Conv2d(in_channels=64, out_channels=self.nvars, kernel_size=3, stride=1, padding=1)

    def forward(self, x_input):
        B, L, M, S = x_input.shape

        res = x_input
        x_input = rearrange(x_input, 'b l m s -> b s l m')
        x_input = self.conv2d1(x_input)
        x_input = self.bn2d1(x_input)
        x_input = self.dropout(self.relu(x_input))
        x_input = self.conv2d2(x_input)
        x_input = rearrange(x_input, 'b s l m -> b l m s')
        x_output = x_input + res
        return x_output


class TSDeformConv2d(nn.Module):
    def __init__(self, configs):
        super(TSDeformConv2d, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.c_out = configs.c_out
        self.d_ff = configs.d_ff
        self.nvars = len(configs.source_names)

        if configs.use_gpu:
            device = torch.device('cuda:{}'.format(0))
            self.device = device

        self.n_offset_grps = 1

        # 生成偏移量的卷积层
        self.offset_conv1 = nn.Conv2d(in_channels=self.nvars, out_channels=2 * 3 * 3 * 1, kernel_size=3, stride=1,
                                      padding=1)
        self.offset_conv2 = nn.Conv2d(in_channels=64, out_channels=2 * 3 * 3 * 1, kernel_size=3, stride=1,
                                      padding=1)

        # 可变形卷积层
        self.conv2d1 = DeformConv2d(in_channels=self.nvars, out_channels=64, kernel_size=3, stride=1, padding=1,
                                    groups=self.n_offset_grps)
        self.bn2d1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.2)
        self.conv2d2 = DeformConv2d(in_channels=64, out_channels=self.nvars, kernel_size=3, stride=1, padding=1,
                                    groups=self.n_offset_grps)

    def forward(self, x_input):
        B, L, M, S = x_input.shape

        res = x_input
        x_input = rearrange(x_input, 'b l m s -> b s l m')
        offset1 = self.offset_conv1(x_input)
        x_input = self.conv2d1(x_input, offset1)
        x_input = self.bn2d1(x_input)
        x_input = self.dropout(self.relu(x_input))
        offset2 = self.offset_conv2(x_input)
        x_input = self.conv2d2(x_input, offset2)
        x_input = rearrange(x_input, 'b s l m -> b l m s')
        x_output = x_input + res
        return x_output