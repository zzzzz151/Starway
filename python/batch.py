from settings import *
import ctypes
import numpy as np
import torch

MAX_PIECES_PER_POS = 32

class Batch(ctypes.Structure):
    _fields_ = [
        ('active_features_stm', ctypes.POINTER(ctypes.c_int16)),
        ('active_features_ntm', ctypes.POINTER(ctypes.c_int16)),
        ('stm_scores_sigmoided', ctypes.POINTER(ctypes.c_float)),
        ('stm_WDLs', ctypes.POINTER(ctypes.c_float)),
        ('total_legal_moves', ctypes.c_size_t),
        ('legal_moves_idxs_and_visits_percent', ctypes.POINTER(ctypes.c_float)),
    ]

    def get_features_tensor(self, is_stm: bool):
        field = self.active_features_stm if is_stm else self.active_features_ntm
        arr = np.ctypeslib.as_array(field, shape=(BATCH_SIZE, MAX_PIECES_PER_POS))
        return torch.from_numpy(arr).to(DEVICE, dtype=torch.int32)

    def get_stm_scores_sigmoided_tensor(self):
        arr = np.ctypeslib.as_array(self.stm_scores_sigmoided, shape=(BATCH_SIZE, 1))
        return torch.from_numpy(arr).to(DEVICE, dtype=torch.float32)

    def get_stm_wdl_tensor(self):
        arr = np.ctypeslib.as_array(self.stm_WDLs, shape=(BATCH_SIZE, 1))
        return torch.from_numpy(arr).to(DEVICE, dtype=torch.float32)

    def get_legals_idxs_and_target_policy_tensors(self):
        # First tensor to return
        legal_moves_idxs_tensor = torch.zeros(
            (BATCH_SIZE, POLICY_OUTPUT_SIZE),
            dtype=torch.bool,
            device=DEVICE
        )

        # Second tensor to return
        target_policy_tensor = torch.zeros(
            (BATCH_SIZE, POLICY_OUTPUT_SIZE),
            dtype=torch.float32,
            device=DEVICE
        )

        arr = np.ctypeslib.as_array(
            self.legal_moves_idxs_and_visits_percent,
            shape=(self.total_legal_moves, 3)
        )

        # [entry_idx][move_idx][visits_percent]
        legal_moves_idxs_and_visits_percent_tensor = torch.from_numpy(arr).to(DEVICE)
        assert legal_moves_idxs_and_visits_percent_tensor.dtype == torch.float32

        legal_moves_idxs_tensor[
            legal_moves_idxs_and_visits_percent_tensor[:, 0].int(),
            legal_moves_idxs_and_visits_percent_tensor[:, 1].int()
        ] = True

        target_policy_tensor[
            legal_moves_idxs_and_visits_percent_tensor[:, 0].int(),
            legal_moves_idxs_and_visits_percent_tensor[:, 1].int()
        ] = legal_moves_idxs_and_visits_percent_tensor[:, 2]

        return legal_moves_idxs_tensor, target_policy_tensor
