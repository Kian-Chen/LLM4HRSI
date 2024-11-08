import os
import torch
# from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
#     Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN
from models import LLM4HRSI, LLM4MRSI, FusionTest, TTTITS, TTT4MHRS


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'LLM4HRSI': LLM4HRSI,
            'LLM4MRSI': LLM4MRSI,
            'FusionTest': FusionTest,
            'TTTITS': TTTITS,
            'TTT4MHRS': TTT4MHRS
        }
        self.expert_dict = {
            'W': 'WithoutFusion',
            'V': 'V_DAB',
            'A': 'ChannelAttention',
            'G': 'GroupConv',
            'S': 'STARc',
            'M': 'STARm',
            'H': 'ShuffleConv',
            'C': 'TSConv2d',
            'D': 'TSDeformConv2d'
        }
        self.args.expert_names = [self.expert_dict[expert] for expert in self.args.expert_ids]
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
