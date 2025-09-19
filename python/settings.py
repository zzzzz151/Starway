import os
import torch

torch.backends.fp32_precision = "ieee"
torch.backends.cuda.matmul.fp32_precision = "ieee"
torch.backends.cudnn.fp32_precision = "ieee"

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

NET_NAME = "net"

# Set to a .pt file to resume training, else set to None
CHECKPOINT_TO_LOAD = None

INPUT_SIZE = 768 * 2
HIDDEN_SIZE = 128 # The final hidden layer is twice as big
POLICY_OUTPUT_SIZE = 1882

# 1 superbatch = 100 million positions
# Total superbatches = END_SUPERBATCH - START_SUPERBATCH + 1
START_SUPERBATCH = 1
END_SUPERBATCH = 600

SAVE_INTERVAL = 30 # Save net checkpoint every SAVE_INTERVAL superbatches

DATA_FILE_PATH = "interleaved_data.sw"
BATCH_OFFSETS_FILE_PATH = "interleaved_batch_offsets.bin"
BATCH_SIZE = 16384
THREADS = 12

# Learning rate schedule
LR = 0.001 * (0.99**(START_SUPERBATCH - 1))
LR_DROP_INTERVAL = 1
LR_MULTIPLIER = 0.99

SCALE = 400
WDL_WEIGHT = 0.0

VALUE_LOSS_WEIGHT = 0.99 # vlw

# To fit in i16: 33 * 5.48 * 181 <= 32767
FT_MAX_WEIGHT_BIAS = 5.48
FT_Q = 181

assert NET_NAME != ""
if CHECKPOINT_TO_LOAD: assert os.path.exists(CHECKPOINT_TO_LOAD)
assert HIDDEN_SIZE > 0
assert POLICY_OUTPUT_SIZE > 0
assert START_SUPERBATCH > 0
if not CHECKPOINT_TO_LOAD: assert START_SUPERBATCH == 1
assert END_SUPERBATCH > 0
assert START_SUPERBATCH <= END_SUPERBATCH
assert SAVE_INTERVAL > 0
assert os.path.exists(DATA_FILE_PATH)
assert os.path.exists(BATCH_OFFSETS_FILE_PATH)
assert BATCH_SIZE > 0
assert THREADS > 0
assert LR > 0.0 and LR_DROP_INTERVAL > 0 and LR_MULTIPLIER > 0.0
assert SCALE > 0
assert WDL_WEIGHT >= 0.0 and WDL_WEIGHT <= 1.0
assert VALUE_LOSS_WEIGHT >= 0.0 and VALUE_LOSS_WEIGHT <= 1.0
assert FT_MAX_WEIGHT_BIAS > 0.0
assert FT_Q > 0
