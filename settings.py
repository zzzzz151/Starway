import os
import torch

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

NET_NAME = "net768x2_hm"
CHECKPOINT_TO_LOAD = None # Set to a .pt file to resume training, else set to None

HIDDEN_SIZE = 1024 # The final hidden layer is twice as big

# 1 superbatch = 100 million positions
# Total superbatches = END_SUPERBATCH - START_SUPERBATCH + 1
START_SUPERBATCH = 1
END_SUPERBATCH = 500

SAVE_INTERVAL = 100 # Save net every SAVE_INTERVAL superbatches

DATA_FILE_PATH = "data.bin" # .bin data file
BATCH_SIZE = 16384
THREADS = 8

# Learning rate schedule
LR = 0.001 * (0.99**(START_SUPERBATCH - 1))
LR_DROP_INTERVAL = 1
LR_MULTIPLIER = 0.99

SCALE = 400
WDL = 0.0
MAX_WEIGHT_BIAS = 1.99

QA, QB = 256, 64

assert NET_NAME != ""
if CHECKPOINT_TO_LOAD: assert os.path.exists(CHECKPOINT_TO_LOAD)
assert HIDDEN_SIZE > 0
assert START_SUPERBATCH > 0
if not CHECKPOINT_TO_LOAD: assert START_SUPERBATCH == 1
assert END_SUPERBATCH > 0
assert START_SUPERBATCH <= END_SUPERBATCH
assert SAVE_INTERVAL > 0
assert os.path.exists(DATA_FILE_PATH)
assert BATCH_SIZE > 0
assert THREADS > 0
assert LR > 0.0 and LR_DROP_INTERVAL > 0 and LR_MULTIPLIER > 0.0
assert SCALE > 0
assert WDL >= 0.0 and WDL <= 1.0
assert MAX_WEIGHT_BIAS > 0.0
assert QA > 0 and QB > 0
