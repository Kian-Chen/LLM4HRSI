import torch
import torch.nn as nn


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h'):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        x = self.value_embedding(x)
        return x