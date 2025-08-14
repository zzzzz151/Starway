from settings import *
import ctypes
import numpy as np
import torch

class Batch(ctypes.Structure):
    _fields_ = [
        ('num_active_features', ctypes.c_size_t),
        ('active_features_stm', ctypes.POINTER(ctypes.c_int16)),
        ('active_features_ntm', ctypes.POINTER(ctypes.c_int16)),
        ('stm_scores', ctypes.POINTER(ctypes.c_int16)),
        ('stm_WDLs', ctypes.POINTER(ctypes.c_float)),
        ('best_move_idx1882', ctypes.POINTER(ctypes.c_int16)),
        ('total_legal_moves', ctypes.c_size_t),
        ('legal_moves_idxs1882', ctypes.POINTER(ctypes.c_int16)),
    ]

    # For active_features_stm, active_features_ntm and legal_moves_idxs1882
    def get_2d_tensor(self, field, count_field, num_cols, dtype):
        tensor = torch.zeros(BATCH_SIZE, num_cols, device=DEVICE, dtype=dtype)

        arr = np.ctypeslib.as_array(field, shape=(count_field, 2))

        indices_tensor = torch.from_numpy(arr).int()
        value = True if dtype == torch.bool else 1

        tensor[indices_tensor[:, 0], indices_tensor[:, 1]] = value

        return tensor

    # For stm_scores, stm_WDLs and best_move_idx1882
    def get_tensor(self, field, shape, dtype):
        arr = np.ctypeslib.as_array(field, shape=shape)
        return torch.from_numpy(arr).to(DEVICE, dtype=dtype)
