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
        ('legal_moves_idxs', ctypes.POINTER(ctypes.c_int16)),
        ('best_move_idx', ctypes.POINTER(ctypes.c_uint8)),
    ]

    def get_features_tensor(self, is_stm: bool):
        field = self.active_features_stm if is_stm else self.active_features_ntm
        arr = np.ctypeslib.as_array(field, shape=(BATCH_SIZE, MAX_PIECES_PER_POS))
        return torch.from_numpy(arr).to(DEVICE, dtype=torch.int32)

    def get_legal_moves_idxs_tensor(self):
        arr = np.ctypeslib.as_array(self.legal_moves_idxs, shape=(BATCH_SIZE, MAX_MOVES_PER_POS))
        return torch.from_numpy(arr).to(DEVICE, dtype=torch.int32)

    def get_stm_scores_tensor(self):
        arr = np.ctypeslib.as_array(self.stm_scores, shape=(BATCH_SIZE, 1))
        return torch.from_numpy(arr).to(DEVICE, dtype=torch.float32)

    def get_stm_wdl_tensor(self):
        arr = np.ctypeslib.as_array(self.stm_WDLs, shape=(BATCH_SIZE, 1))
        return torch.from_numpy(arr).to(DEVICE, dtype=torch.float32)

    def get_target_policy_tensor(self):
        arr = np.ctypeslib.as_array(self.best_move_idx, shape=(BATCH_SIZE, 1))
        best_move_idx_tensor = torch.from_numpy(arr).to(DEVICE, dtype=torch.int)

        result_tensor = torch.zeros(
            BATCH_SIZE, MAX_MOVES_PER_POS, dtype=torch.float32, device=DEVICE
        )

        result_tensor.scatter_(1, best_move_idx_tensor, 1.0)

        return result_tensor
