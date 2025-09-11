from settings import *
from batch import Batch
from model import NetValuePolicy
import ctypes
import numpy as np
import torch
import math
import time
import sys
import os

SUPERBATCHES = END_SUPERBATCH - START_SUPERBATCH + 1
BATCHES_PER_SUPERBATCH = math.ceil(100_000_000.0 / float(BATCH_SIZE))
SCORE_WEIGHT = 1.0 - WDL_WEIGHT
POLICY_LOSS_WEIGHT = 1.0 - VALUE_LOSS_WEIGHT

if __name__ == "__main__":
    net = NetValuePolicy().to(DEVICE)
    net.print_info()

    print("Device:", "CPU" if DEVICE == torch.device("cpu") else torch.cuda.get_device_name(0))
    print("Checkpoint to load:", CHECKPOINT_TO_LOAD)

    print("Superbatches: {} to {} ({} total)"
        .format(START_SUPERBATCH, END_SUPERBATCH, SUPERBATCHES))

    print("Save interval: every {} superbatches".format(SAVE_INTERVAL))
    print("Data file:", DATA_FILE_PATH)
    print("Batch size:", BATCH_SIZE)
    print("Dataloader threads:", THREADS)

    print("LR: start {} multiply by {} every {} superbatches"
        .format(LR, LR_MULTIPLIER, LR_DROP_INTERVAL))

    print("Value scale:", SCALE)
    print("WDL weight for value head:", WDL_WEIGHT)
    print("FT params clipping: [{}, {}]".format(-FT_MAX_WEIGHT_BIAS, FT_MAX_WEIGHT_BIAS))
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

    net = torch.compile(net)

    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=LR_DROP_INTERVAL,
        gamma=LR_MULTIPLIER
    )

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
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    ce_fn = torch.nn.CrossEntropyLoss()

    for superbatch_num in range(START_SUPERBATCH, END_SUPERBATCH + 1):
        superbatch_start_time = time.time()
        sb_value_loss = 0.0
        sb_policy_loss = 0.0

        for param_group in optimizer.param_groups:
            print("LR for superbatch #{}:".format(superbatch_num), round(param_group['lr'], 5))

        for batch_num in range(1, BATCHES_PER_SUPERBATCH + 1):
            batch = dataloader.nextBatch().contents

            optimizer.zero_grad(set_to_none=True)

            target_logits = batch.get_target_logits_tensor()

            pred_value, pred_logits = net.forward(
                batch.get_features_tensor(True),
                batch.get_features_tensor(False),
                target_logits
            )

            expected_value = torch.sigmoid(batch.get_scores_tensor() / float(SCALE)) * SCORE_WEIGHT
            expected_value += batch.get_wdl_tensor() * WDL_WEIGHT

            value_abs_diff = torch.abs(torch.sigmoid(pred_value) - expected_value)

            def softmax(x):
                return torch.nn.functional.softmax(x, dim=1)

            value_loss = torch.pow(value_abs_diff, 2.5).mean()
            policy_loss = ce_fn(softmax(pred_logits), softmax(target_logits))

            loss = value_loss * VALUE_LOSS_WEIGHT + policy_loss * POLICY_LOSS_WEIGHT

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

        lr_scheduler.step()

        # Save checkpoint as .pt (pytorch file)
        mod = (superbatch_num - START_SUPERBATCH + 1) % SAVE_INTERVAL
        if mod == 0 or superbatch_num == END_SUPERBATCH:
            checkpoint = {
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict()
            }

            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")

            pt_file_path = "checkpoints/{}-{}.pt".format(NET_NAME, superbatch_num)
            torch.save(checkpoint, pt_file_path)
            print("Checkpoint saved", pt_file_path)
