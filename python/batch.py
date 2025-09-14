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
        ('total_legal_moves', ctypes.c_size_t),
        ('legal_moves_idxs_and_visits', ctypes.POINTER(ctypes.c_uint32)),
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
        target_logits_tensor = torch.full(
            size=(BATCH_SIZE, POLICY_OUTPUT_SIZE),
            fill_value=ILLEGAL_LOGITS_VALUE,
            dtype=torch.float32,
            device=DEVICE
        )

        # legal_moves_idxs_and_visits_tensor stores tuples (entry_idx, move_idx, visits)
        arr = np.ctypeslib.as_array(self.legal_moves_idxs_and_visits, shape=(self.total_legal_moves, 3))
        legal_moves_idxs_and_visits_tensor = torch.from_numpy(arr).int().to(DEVICE)

        # In target_logits_tensor, set legal moves to their visits
        target_logits_tensor[
            legal_moves_idxs_and_visits_tensor[:, 0],
            legal_moves_idxs_and_visits_tensor[:, 1]
        ] = legal_moves_idxs_and_visits_tensor[:, 2].float()

        return target_logits_tensor
