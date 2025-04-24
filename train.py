from settings import *
from batch import Batch
from model import PerspectiveNet768x2
import ctypes
import numpy as np
import torch
import math
import time
import sys
import os

torch.set_float32_matmul_precision('medium')

if __name__ == "__main__":
    SUPERBATCHES = END_SUPERBATCH - START_SUPERBATCH + 1
    NUM_DATA_ENTRIES = os.path.getsize(DATA_FILE_PATH) / 32

    print("Device:", "CPU" if DEVICE == torch.device("cpu") else torch.cuda.get_device_name(0))
    print("Net name:", NET_NAME)
    print("Net arch: (768x2x{} -> {})x2 -> 1, vertical axis mirroring"
        .format(NUM_INPUT_BUCKETS, HIDDEN_SIZE))
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
    print("Weights/biases clipping: [{}, {}]".format(-MAX_WEIGHT_BIAS, MAX_WEIGHT_BIAS))
    print()

    # Launch dataloader
    dll_exists = os.path.exists("./dataloader.dll")
    so_exists = os.path.exists("./dataloader.so")
    assert dll_exists or so_exists
    dataloader = ctypes.CDLL("./dataloader.dll" if dll_exists else "./dataloader.so")

    # Define dataloader's functions return types
    dataloader.init.restype = None # void
    dataloader.nextBatch.restype = ctypes.POINTER(Batch)

    # Define dataloader's init() arguments types
    dataloader.init.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t)
    ]

    # Init dataloader
    dataloader.init(
        ctypes.c_char_p(DATA_FILE_PATH.encode('utf-8')),
        BATCH_SIZE,
        THREADS,
        NUM_INPUT_BUCKETS,
        (ctypes.c_size_t * len(INPUT_BUCKETS_MAP))(*INPUT_BUCKETS_MAP)
    )

    net = PerspectiveNet768x2().to(DEVICE)

    #optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)

    scaler = torch.cuda.amp.GradScaler()

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

        scaler.load_state_dict(checkpoint["scaler"])

    # 1 superbatch = 100M positions
    BATCHES_PER_SUPERBATCH = math.ceil(100_000_000.0 / float(BATCH_SIZE))

    for superbatch_num in range(START_SUPERBATCH, END_SUPERBATCH + 1):
        superbatch_start_time = time.time()
        superbatch_total_loss = 0.0

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

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = net.forward(
                    batch.get_features_tensor(True),
                    batch.get_features_tensor(False),
                    Batch.to_tensor(batch.is_white_stm)
                )

                assert prediction.dtype == torch.bfloat16

                expected = torch.sigmoid(Batch.to_tensor(batch.stm_scores) / float(SCALE)) * (1.0 - WDL)
                expected += Batch.to_tensor(batch.stm_WDLs) * WDL
                assert expected.dtype == torch.float32

                loss = torch.pow(torch.abs(torch.sigmoid(prediction) - expected), 2.5).mean()
                assert loss.dtype == torch.float32

            superbatch_total_loss += loss.item()

            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()

            scaler.step(optimizer)
            scaler.update()

            net.clamp_weights_biases()

            # Log every N batches
            if batch_num == 1 or batch_num == BATCHES_PER_SUPERBATCH or batch_num % 64 == 0:
                positions_seen_this_superbatch = batch_num * BATCH_SIZE
                positions_per_sec = positions_seen_this_superbatch / (time.time() - superbatch_start_time)

                log = "\rSuperbatch {}/{}, batch {}/{}, superbatch train loss {:.4f}, {} positions/s".format(
                    superbatch_num,
                    END_SUPERBATCH,
                    batch_num,
                    BATCHES_PER_SUPERBATCH,
                    superbatch_total_loss / batch_num,
                    round(positions_per_sec)
                )

                if batch_num == BATCHES_PER_SUPERBATCH:
                    print(log)
                else:
                    sys.stdout.write(log)
                    sys.stdout.flush()

        # Save checkpoint as .pt (pytorch file)
        if (superbatch_num - START_SUPERBATCH + 1) % SAVE_INTERVAL == 0 or superbatch_num == END_SUPERBATCH:
            checkpoint = {
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict()
            }

            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")

            pt_file_path = "checkpoints/{}-{}.pt".format(NET_NAME, superbatch_num)
            torch.save(checkpoint, pt_file_path)
            print("Checkpoint saved", pt_file_path)
