from settings import *
import ctypes
import numpy as np
import torch

MAX_PIECES_PER_POS = 32

class Batch(ctypes.Structure):
    _fields_ = [
        ('active_features_stm', ctypes.POINTER(ctypes.c_int16)),
        ('active_features_ntm', ctypes.POINTER(ctypes.c_int16)),
        ('stm_scores', ctypes.POINTER(ctypes.c_int16)),
        ('stm_WDLs', ctypes.POINTER(ctypes.c_float)),
        ('target_logits', ctypes.POINTER(ctypes.c_int16)),
    ]

    def get_features_tensor(self, is_stm: bool):
        field = self.active_features_stm if is_stm else self.active_features_ntm
        arr = np.ctypeslib.as_array(field, shape=(BATCH_SIZE, MAX_PIECES_PER_POS))
        return torch.from_numpy(arr).to(DEVICE, dtype=torch.int32)

    def get_scores_tensor(self):
        arr = np.ctypeslib.as_array(self.stm_scores, shape=(BATCH_SIZE, 1))
        return torch.from_numpy(arr).to(DEVICE, dtype=torch.float32)

    def get_wdl_tensor(self):
        arr = np.ctypeslib.as_array(self.stm_WDLs, shape=(BATCH_SIZE, 1))
        return torch.from_numpy(arr).to(DEVICE, dtype=torch.float32)

    def get_target_logits_tensor(self):
        arr = np.ctypeslib.as_array(self.target_logits, shape=(BATCH_SIZE, POLICY_OUTPUT_SIZE))
        return torch.from_numpy(arr).to(DEVICE, dtype=torch.float32)
