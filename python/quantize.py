from settings import *
from model import NetValuePolicy
from display_net_output import FENS, print_net_output
import os
import torch
import numpy as np
import struct
import chess

if __name__ == "__main__":
    net = NetValuePolicy().to(DEVICE)
    net.print_info()

    print("Device:", "CPU" if DEVICE == torch.device("cpu") else torch.cuda.get_device_name(0))
    print("Checkpoint to load:", CHECKPOINT_TO_LOAD)
    print("FT weights/biases clipping: [{}, {}]".format(-FT_MAX_WEIGHT_BIAS, FT_MAX_WEIGHT_BIAS))
    print("FT quantization:", FT_Q)
    print("Value scale:", VALUE_SCALE)
    print()

    assert CHECKPOINT_TO_LOAD != None and os.path.exists(CHECKPOINT_TO_LOAD)

    net = torch.compile(net)

    checkpoint = torch.load(CHECKPOINT_TO_LOAD, weights_only=False)
    net.load_state_dict(checkpoint["model"])

    FT_Q = float(FT_Q)

    # Quantize FT
    with torch.no_grad():
        net.ft.weight.data = torch.round(net.ft.weight.data * FT_Q)
        net.ft.bias.data = torch.round(net.ft.bias.data * FT_Q)

    # Write weights and biases to binary file
    out_file_path = CHECKPOINT_TO_LOAD[:-3] + ".bin"
    with open(out_file_path, 'wb') as out_file:
        def write_bin(weights_or_biases, np_type):
            flattened = weights_or_biases.detach().cpu().numpy().astype(np_type).flatten().tolist()
            letter = {np.int16: 'h', np.int32: 'i', np.float32: 'f'}[np_type]
            out_file.write(struct.pack('<' + letter * len(flattened), *flattened))

        # Write features weights and hidden biases
        write_bin(net.ft.weight, np.int16)
        write_bin(net.ft.bias, np.int16)

        mid = int(HIDDEN_SIZE/2)

        # Write value head weights and biases
        write_bin(net.hidden_to_out_value.weight[:, :mid], np.float32)
        write_bin(net.hidden_to_out_value.weight[:, mid:], np.float32)
        write_bin(net.hidden_to_out_value.bias, np.float32)

        # Write policy head weights and biases
        for i in range(1882):
            write_bin(net.hidden_to_out_policy.weight[i][:mid], np.float32)
            write_bin(net.hidden_to_out_policy.weight[i][mid:], np.float32)
        write_bin(net.hidden_to_out_policy.bias, np.float32)

        # Pad file so size is multiple of 64 bytes
        current_size = out_file.tell()
        padding_needed = (64 - (current_size % 64)) % 64
        if padding_needed:
            pad_pattern = b'STARWAY'
            full_repeats = padding_needed // len(pad_pattern)
            remainder = padding_needed % len(pad_pattern)
            out_file.write(pad_pattern * full_repeats + pad_pattern[:remainder])

        print(out_file_path)
        print()

    # Unquantize FT
    with torch.no_grad():
        net.ft.weight.data /= FT_Q
        net.ft.bias.data /= FT_Q

        assert net.ft.weight.dtype == torch.float32
        assert net.ft.bias.dtype == torch.float32

    # Print quantized output for some positions
    print_net_output(net, FENS)
