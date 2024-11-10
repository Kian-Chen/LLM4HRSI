import numpy as np


def tensorify(X, y, shape):
    tensor = np.full(shape, np.nan)
    for i, d in enumerate(X):
        lat, lon, t = d.astype(int)
        tensor[lat, lon, t] = y[i]
    return tensor

def center_3d_tensor(tensor):
    temp_tensor = tensor.copy()
    m0 = np.nanmean(temp_tensor, axis=0)
    m0[np.isnan(m0)] = 0
    for i in range(temp_tensor.shape[0]):
        temp_tensor[i, :, :] -= m0

    m1 = np.nanmean(temp_tensor, axis=1)
    m1[np.isnan(m1)] = 0
    for i in range(temp_tensor.shape[1]):
        temp_tensor[:, i, :] -= m1

    m2 = np.nanmean(temp_tensor, axis=2)
    m2[np.isnan(m2)] = 0
    for i in range(temp_tensor.shape[2]):
        temp_tensor[:, :, i] -= m2

    return temp_tensor, m0, m1, m2

def decenter_3d_tensor(tensor, spatial_means, m2):
    temp_tensor = tensor.copy()
    m0, m1 = spatial_means
    for i in range(temp_tensor.shape[0]):
        temp_tensor[i, :, :] += m0
    for i in range(temp_tensor.shape[1]):
        temp_tensor[:, i, :] += m1
    for i in range(temp_tensor.shape[2]):
        temp_tensor[:, :, i] += m2
    return temp_tensor

def rectify_tensor(tensor):
    rect_mat = []
    for t in range(tensor.shape[-1]):
        rect_mat.append(tensor[:, :, t].flatten())
    rect_mat = np.array(rect_mat)
    rect_mat = np.moveaxis(rect_mat, 0, -1)
    return rect_mat

def unrectify_mat(mat, spatial_shape):
    tensor = []
    for t in range(mat.shape[-1]):
        col = mat[:, t]
        unrectified_col = col.reshape(spatial_shape)
        tensor.append(unrectified_col)
    tensor = np.array(tensor)
    tensor = np.moveaxis(tensor, 0, -1)
    return tensor

def nrmse(y_hat, y):
    root_meaned_sqd_diff = np.sqrt(np.mean(np.power(y_hat - y, 2)))
    return root_meaned_sqd_diff / np.std(y)
