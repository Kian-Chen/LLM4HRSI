import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.utils import normalization_grud as normalization

import warnings
import math

warnings.filterwarnings('ignore')


def top_k_indices(nums, k):
    """
    返回给定列表中前k个最大的元素的索引值
    :param nums: 给定列表
    :param k: 返回的元素个数
    :return: 前k个最大元素的索引值列表
    """
    indices = sorted(range(len(nums)), key=lambda i: nums[i], reverse=True)
    return indices[:k]


# m is row, n is col
def sample_MN(raw_M, m, n, p):
    # p是missing rate，0.7--> 有70的数据是0
    np.random.seed(10)
    # 生成0,1正态分布，形状与observed_mask相同，而后与raw_M相乘,1的位置保持原样，0的位置变为0
    rand_for_mask = np.random.uniform(np.nextafter(0, 1), np.nextafter(1, 0), size=[m, n]) * raw_M
    # 统计每列1的个数（即有观察值的个数），形状为（K,),值为对应列1的个数
    num_observed = (raw_M == 1).astype(int).sum(axis=0)
    # TODO? why set raw mask to 0, artificially missing to -1
    for i in range(n):
        # 统计i列需要缺失的个数（有值个数*缺失率）
        num_masked = round(num_observed[i] * p)
        # 找出矩阵i列有值位置前num_masked个大的值及其对应的索引
        indices = top_k_indices(rand_for_mask[:, i], num_masked)
        rand_for_mask[indices, i] = -1
    # rand_for_mask > 0表示存在剩余观察值，0或者为负表示原始就无观察值或者删除观察值

    cond_mask = (rand_for_mask > 0).astype(int)
    return cond_mask


def sample_mask(shape, p=0.002, p_noise=0., max_seq=1, min_seq=1, rng=None):
    if rng is None:
        rand = np.random.random
        randint = np.random.randint
    else:
        rand = rng.random
        randint = rng.integers

    mask = rand(shape) < p
    for col in range(mask.shape[1]):
        idxs = np.flatnonzero(mask[:, col])
        if not len(idxs):
            continue
        fault_len = min_seq
        if max_seq > min_seq:
            fault_len = fault_len + int(randint(max_seq - min_seq))
        idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
        idxs = np.unique(idxs_ext)
        idxs = np.clip(idxs, 0, shape[0] - 1)
        mask[idxs, col] = True
    mask = mask | (rand(mask.shape) < p_noise)
    return mask.astype('uint8')


class Dataset_Custom(Dataset):
    def __init__(self, configs, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, timeenc=0, freq='h',
                 seasonal_patterns=None, artificially_missing_rate=0.5, percent=10):
        # size [seq_len, label_len, pred_len]
        self.configs = configs

        # info
        if size == None:
            # TODO? why set 24 * 4 * 4
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val', 'gen']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'gen': 3}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.artificially_missing_rate = artificially_missing_rate

        self.percent = percent
        self.root_path = root_path
        self.data_path = data_path

        SEED = 56789

        self.rng = np.random.default_rng(SEED)

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = np.loadtxt(os.path.join(self.root_path,
                                         self.data_path), dtype=float, delimiter=",")
        # mask = ~np.isnan(df_raw.values)

        row, col = df_raw.shape
        # print("Raw data shape: ", row,col)
        # TODO? what is the relation among missing_mask, mask and eval_mask
        # first, mask equals 0 iff where origin data is 0
        # missing_mask is mask using sample_MN mask again
        # equal_mask equals 0 iff mask and missing_mask=0
        # second set mask = missing_mask
        mask = np.ones(shape=(row, col))
        mask[df_raw == 0] = 0

        data, norm_parameters = normalization(df_raw)
        data[mask == 0] = 0

        missing_mask = sample_MN(mask, row, col, self.artificially_missing_rate)
        # print("Missing rate: ", self.artificially_missing_rate)

        eval_mask = (mask - missing_mask).astype('uint8')
        mask = missing_mask.astype('uint8')

        num_train = int(len(df_raw) * 0.8)
        num_test = int(len(df_raw) * 0.1)
        num_vali = len(df_raw) - num_train - num_test
        # print("Train/Vali/Test: ", num_train,num_test,num_vali,len(df_raw))
        # TODO? why set these borders that minues seq_len
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len, 0]
        border2s = [num_train, num_train + num_vali, len(df_raw), len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        # print("Border1/Border2: ", border1, border2)

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.mask = mask[border1:border2]
        self.eval_mask = eval_mask[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        # print("s_begin, s_end, r_begin, r_end: ", s_begin, s_end, r_begin, r_end)
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_mask = self.mask[s_begin:s_end]
        seq_eval_mask = self.eval_mask[s_begin:s_end]

        return seq_x, seq_y, seq_mask, seq_eval_mask

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Multisource(Dataset):
    def __init__(self, configs, root_path, file_list=None, limit_size=None, flag='train', size=None,
                 data_path='ETTh1.csv', seasonal_patterns=None,
                 features='S', target='OT', scale=False, timeenc=0, freq='h',
                 artificially_missing_rate=0.5, percent=10):

        # size [seq_len, label_len, pred_len]
        self.configs = configs
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val', 'gen']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'gen': 3}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.artificially_missing_rate = artificially_missing_rate

        self.percent = percent
        self.root_path = root_path
        self.source_names = configs.source_names
        self.file_list = file_list if file_list else [
            os.path.join(root_path, f"{source_name}_{data_path}") for source_name in
            self.source_names]
        self.limit_size = limit_size

        SEED = 56789
        self.rng = np.random.default_rng(SEED)

        self.__read_data__()

    def __read_data__(self):
        data_frames = [pd.read_csv(file_path).values for file_path in self.file_list]

        # Add new dimension for each data source
        data_sources = [df[np.newaxis, :] for df in data_frames]

        combined_data = np.concatenate(data_sources, axis=0)
        # print("Combined data shape: ", combined_data.shape)

        if self.limit_size:
            combined_data = combined_data[:, :int(self.limit_size * combined_data.shape[1]), :]

        num_sources, seq_len, num_features = combined_data.shape
        # print("Raw data shape: ", num_sources, seq_len, num_features)

        masks = np.ones_like(combined_data)
        masks[combined_data == 0] = 0

        data_list = []
        norm_params_list = []
        mask_list = []
        eval_mask_list = []

        for i in range(num_sources):
            data, norm_parameters = normalization(combined_data[i])
            data[masks[i] == 0] = 0

            missing_mask = sample_MN(masks[i], seq_len, num_features, self.artificially_missing_rate)
            # print(f"Missing rate for source {i}: ", self.artificially_missing_rate)

            eval_mask = (masks[i] - missing_mask).astype('uint8')
            mask = missing_mask.astype('uint8')

            data_list.append(data)
            norm_params_list.append(norm_parameters)
            mask_list.append(mask)
            eval_mask_list.append(eval_mask)

        data = np.array(data_list)
        mask = np.array(mask_list)
        eval_mask = np.array(eval_mask_list)

        num_train = int(seq_len * 0.8)
        num_test = int(seq_len * 0.1)
        num_vali = seq_len - num_train - num_test
        # print("Train/Vali/Test: ", num_train, num_test, num_vali, seq_len)
        border1s = [0, num_train - self.seq_len, seq_len - num_test - self.seq_len, 0]
        border2s = [num_train, num_train + num_vali, seq_len, seq_len]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        # print("Border1/Border2: ", border1, border2)

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        self.data_x = data[:, border1:border2, :]
        self.data_y = data[:, border1:border2, :]
        self.mask = mask[:, border1:border2, :]
        self.eval_mask = eval_mask[:, border1:border2, :]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[:, s_begin:s_end, :]
        seq_y = self.data_y[:, r_begin:r_end, :]
        seq_mask = self.mask[:, s_begin:s_end, :]
        seq_eval_mask = self.eval_mask[:, s_begin:s_end, :]

        return seq_x, seq_y, seq_mask, seq_eval_mask

    def __len__(self):
        return self.data_x.shape[1] - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        data = data.reshape(-1, data.shape[-1])
        return self.scaler.inverse_transform(data).reshape(data.shape)


class Dataset_Trad:
    def __init__(self, configs):
        root_path = configs.root_path
        cur_source = configs.source_names[0]
        data_path = configs.data_path
        filename = f"{cur_source}_{data_path}"
        self.data_path = os.path.join(root_path, filename)
        self.artificially_missing_rate = configs.mask_rate
        self.lat = configs.lat
        self.lon = configs.lon
        self.__read_data__()

    def __read_data__(self):
        data = np.loadtxt(self.data_path, delimiter=",")

        time_steps = data.shape[0]
        M = data.shape[1]
        assert M == self.lat * self.lon, "列数必须等于 lat * lon 的乘积。"

        # (time_steps, lat, lon)
        data = data.reshape(time_steps, self.lat, self.lon)

        raw_mask = np.ones_like(data)
        raw_mask[data == 0] = 0  # origin mask

        self.missing_mask = sample_MN(raw_mask.reshape(time_steps, self.lat * self.lon), time_steps,
                                      self.lat * self.lon, self.artificially_missing_rate)
        self.missing_mask = self.missing_mask.reshape(time_steps, self.lat, self.lon)
        self.eval_mask = (raw_mask - self.missing_mask).astype(int)

        self.data = np.copy(data)

    def get_data(self):
        return self.data, self.missing_mask, self.eval_mask
