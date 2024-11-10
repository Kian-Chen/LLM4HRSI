import numpy as np
from scipy.sparse.linalg import svds
from tqdm import trange
from utils.dineof import (center_3d_tensor, rectify_tensor, decenter_3d_tensor,
                          unrectify_mat)


class DINEOF:
    def __init__(self, configs, tensor_shape, early_stopping=True):
        self.K = configs.rank  # SVD rank
        self.nitemax = configs.nitemax
        self.tol = configs.tol
        self.to_center = configs.to_center
        self.keep_non_negative_only = configs.keep_non_negative_only
        self.tensor_shape = tensor_shape  # (lat, lon, L)
        self.early_stopping = early_stopping
        self.reconstructed_tensor = None

    def fit(self, data, missing_mask):
        """
        输入的数据 `data` 为三维张量，`missing_mask` 为指示缺失值位置的布尔矩阵
        """
        mat = data.astype(float)
        mat[missing_mask == 0] = 0
        nan_mask = missing_mask == 0

        # 中心化处理
        if self.to_center:
            mat, m0, m1, m2 = center_3d_tensor(mat)

        # SVD迭代补全
        rectified_mat = rectify_tensor(mat)
        rectified_mask = rectify_tensor(missing_mask)  # 将掩码展平为与rectified_mat相同的形状
        pbar = trange(self.nitemax, desc='Reconstruction')
        conv_error = 0
        for i in pbar:
            u, s, vt = svds(rectified_mat, k=self.K, tol=self.tol)
            mat_hat = u @ np.diag(s) @ vt
            mat_hat[rectified_mask == 1] = rectified_mat[rectified_mask == 1]
            new_error = np.sqrt(np.mean((mat_hat[rectified_mask == 0] - rectified_mat[rectified_mask == 0]) ** 2)) / np.std(rectified_mat[rectified_mask == 1])

            pbar.set_postfix(error=new_error, rel_error=abs(new_error - conv_error))
            grad_conv_error = abs(new_error - conv_error)
            conv_error = new_error

            if self.early_stopping and (conv_error <= self.tol or grad_conv_error < self.tol):
                break

            rectified_mat = mat_hat

        # 恢复为三维结构并去中心化
        mat = unrectify_mat(rectified_mat, self.tensor_shape[:-1])
        if self.to_center:
            mat = decenter_3d_tensor(mat, (m0, m1), m2)

        # 保持非负约束
        if self.keep_non_negative_only:
            mat[mat < 0] = 0

        self.reconstructed_tensor = mat

    def predict(self, indices):
        """
        根据缺失数据的索引返回重构的补全值
        """
        return np.array([self.reconstructed_tensor[x[0], x[1], x[2]] for x in indices])
