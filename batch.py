from settings import *
import ctypes
import numpy as np
import torch

class Batch(ctypes.Structure):
    _fields_ = [
        ('num_active_features', ctypes.c_uint32),
        ('active_features_white', ctypes.POINTER(ctypes.c_int16)),
        ('active_features_black', ctypes.POINTER(ctypes.c_int16)),
        ('is_white_stm', ctypes.POINTER(ctypes.c_bool)),
        ('stm_scores', ctypes.POINTER(ctypes.c_int16)),
        ('stm_WDLs', ctypes.POINTER(ctypes.c_float))
    ]

    def get_features_tensor(self, is_white_perspective: bool, dtype = torch.bool):
        features_tensor = torch.zeros(BATCH_SIZE, 768, device=DEVICE, dtype=dtype)

        arr = np.ctypeslib.as_array(
            self.active_features_white if is_white_perspective else self.active_features_black,
            shape=(self.num_active_features, 2)
        )

        indices_tensor = torch.from_numpy(arr).int()
        value = True if dtype == torch.bool else 1

        features_tensor[indices_tensor[:, 0], indices_tensor[:, 1]] = value

        return features_tensor

    @staticmethod
    def to_tensor(x):
        arr = np.ctypeslib.as_array(x, shape=(BATCH_SIZE, 1))

        if arr.dtype == np.bool_:
            dtype = torch.bool
        elif arr.dtype == np.int16:
            dtype = torch.int16
        elif arr.dtype == np.float32:
            dtype = torch.float32
        else:
            raise ValueError(f"Unsupported dtype: {arr.dtype}")

        return torch.from_numpy(arr).to(DEVICE, dtype=dtype)
