import ctypes
import numpy as np
import torch
import math
import time
import sys
import os
from settings import *
from batch import Batch
from model import NetValuePolicy

if __name__ == "__main__":
    SUPERBATCHES = END_SUPERBATCH - START_SUPERBATCH + 1
    NUM_DATA_ENTRIES = os.path.getsize(DATA_FILE_PATH) / 32
    POLICY_LOSS_WEIGHT = 1.0 - VALUE_LOSS_WEIGHT

    print("Device:", "CPU" if DEVICE == torch.device("cpu") else torch.cuda.get_device_name(0))
    print("Net name:", NET_NAME)
    print("Net arch: (768x2HM -> {})x2 -> 1883".format(HIDDEN_SIZE))
    print("Checkpoint to load:", CHECKPOINT_TO_LOAD)
    print("Superbatches: {} to {} ({} total)"
        .format(START_SUPERBATCH, END_SUPERBATCH, SUPERBATCHES))
    print("Save interval: every {} superbatches".format(SAVE_INTERVAL))
    print("Data file:", DATA_FILE_PATH)
    print("Data entries:", NUM_DATA_ENTRIES)
    print("Batch size:", BATCH_SIZE)
    print("Threads:", THREADS)
    print("LR: start {} multiply by {} every {} superbatches"
        .format(LR, LR_MULTIPLIER, LR_DROP_INTERVAL))
    print("Scale:", SCALE)
    print("WDL:", WDL)
    print("FT weights/biases clipping: [{}, {}]".format(-FT_MAX_WEIGHT_BIAS, FT_MAX_WEIGHT_BIAS))
    print()

    # Launch dataloader
    dll_exists = os.path.exists("./dataloader.dll")
    so_exists = os.path.exists("./dataloader.so")
    assert dll_exists or so_exists
    dataloader = ctypes.CDLL("./dataloader.dll" if dll_exists else "./dataloader.so")

    # Define dataloader functions
    dataloader.init.restype = None # void
    dataloader.init.argtypes = [ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32]
    dataloader.nextBatch.restype = ctypes.POINTER(Batch)

    # Init dataloader
    dataloader.init(
        ctypes.c_char_p(DATA_FILE_PATH.encode('utf-8')),
        BATCH_SIZE,
        THREADS
    )

    net = NetValuePolicy().to(DEVICE)

    #optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)

    net = torch.compile(net)

    # Load checkpoint if resuming training
    if CHECKPOINT_TO_LOAD:
        print("Resuming training from checkpoint", CHECKPOINT_TO_LOAD)

        checkpoint = torch.load(
            CHECKPOINT_TO_LOAD,
            map_location = lambda storage,
            loc: storage.cuda(torch.cuda.current_device()),
            weights_only=False
        )

        net.load_state_dict(checkpoint["model"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        assert(len(optimizer.param_groups) == 1 and "lr" in optimizer.param_groups[0])
        optimizer.param_groups[0]["lr"] = LR

    # 1 superbatch = 100M positions
    BATCHES_PER_SUPERBATCH = math.ceil(100_000_000.0 / float(BATCH_SIZE))

    ce_fn = torch.nn.CrossEntropyLoss()

    for superbatch_num in range(START_SUPERBATCH, END_SUPERBATCH + 1):
        superbatch_start_time = time.time()
        sb_value_loss = 0.0
        sb_policy_loss = 0.0

        # Drop learning rate
        should_drop_lr = (superbatch_num - START_SUPERBATCH) % LR_DROP_INTERVAL == 0
        if superbatch_num > START_SUPERBATCH and should_drop_lr:
            LR *= LR_MULTIPLIER
            assert(len(optimizer.param_groups) == 1 and "lr" in optimizer.param_groups[0])
            optimizer.param_groups[0]["lr"] = LR
            print("LR dropped to {}".format(LR))

        for batch_num in range(1, BATCHES_PER_SUPERBATCH + 1):
            batch = dataloader.nextBatch().contents

            optimizer.zero_grad(set_to_none=True)

            pred_value, pred_logits = net.forward(
                batch.get_features_tensor(batch.active_features_stm),
                batch.get_features_tensor(batch.active_features_ntm),
                batch.get_legal_moves_idxs1882_tensor(torch.bool)
            )

            assert pred_value.dtype == torch.float32
            assert pred_logits.dtype == torch.float32

            stm_scores = batch.get_tensor(batch.stm_scores, (BATCH_SIZE, 1), torch.float32)
            stm_WDLs = batch.get_tensor(batch.stm_WDLs, (BATCH_SIZE, 1), torch.float32)

            assert stm_scores.dtype == torch.float32
            assert stm_WDLs.dtype == torch.float32

            expected_value = torch.sigmoid(stm_scores / float(SCALE)) * (1.0 - WDL)
            expected_value += stm_WDLs * WDL

            value_abs_diff = torch.abs(torch.sigmoid(pred_value) - expected_value)

            best_move_tensor = batch.get_tensor(
                batch.best_move_idx1882, (BATCH_SIZE,), torch.long
            )

            value_loss = torch.pow(value_abs_diff, 2.5).mean()
            policy_loss = ce_fn(pred_logits, best_move_tensor)

            assert value_loss.dtype == torch.float32
            assert policy_loss.dtype == torch.float32

            loss = value_loss * VALUE_LOSS_WEIGHT + policy_loss * POLICY_LOSS_WEIGHT
            assert loss.dtype == torch.float32

            sb_value_loss += value_loss.item()
            sb_policy_loss += policy_loss.item()

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                net.ft.weight.clamp_(-FT_MAX_WEIGHT_BIAS, FT_MAX_WEIGHT_BIAS)
                net.ft.bias.clamp_(-FT_MAX_WEIGHT_BIAS, FT_MAX_WEIGHT_BIAS)

            # Log every N batches
            if batch_num == 1 or batch_num == BATCHES_PER_SUPERBATCH or batch_num % 64 == 0:
                positions_seen_this_superbatch = batch_num * BATCH_SIZE
                elapsed = time.time() - superbatch_start_time
                positions_per_sec = positions_seen_this_superbatch / elapsed

                log_template = "Superbatch {}/{}, " \
                    "batch {}/{}, " \
                    "sb value loss = {:.4f}*{:.2f} = {:.4f}, " \
                    "sb policy loss = {:.4f}*{:.2f} = {:.4f}, " \
                    "{} positions/s"

                log = "\r" + log_template.format(
                    superbatch_num,
                    END_SUPERBATCH,
                    batch_num,
                    BATCHES_PER_SUPERBATCH,
                    sb_value_loss / batch_num,
                    VALUE_LOSS_WEIGHT,
                    sb_value_loss * VALUE_LOSS_WEIGHT / batch_num,
                    sb_policy_loss / batch_num,
                    POLICY_LOSS_WEIGHT,
                    sb_policy_loss * POLICY_LOSS_WEIGHT / batch_num,
                    round(positions_per_sec)
                )

                if batch_num == BATCHES_PER_SUPERBATCH:
                    print(log)
                else:
                    sys.stdout.write(log)
                    sys.stdout.flush()

        # Save checkpoint as .pt (pytorch file)
        mod = (superbatch_num - START_SUPERBATCH + 1) % SAVE_INTERVAL
        if mod == 0 or superbatch_num == END_SUPERBATCH:
            checkpoint = {
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict()
            }

            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")

            pt_file_path = "checkpoints/{}-{}.pt".format(NET_NAME, superbatch_num)
            torch.save(checkpoint, pt_file_path)
            print("Checkpoint saved", pt_file_path)
