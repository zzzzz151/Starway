from settings import *
from model import PerspectiveNet768x2
import os
import torch
import numpy as np
import struct

if __name__ == "__main__":
    print("Device:", "CPU" if DEVICE == torch.device("cpu") else torch.cuda.get_device_name(0))
    print("Net name:", NET_NAME)
    print("Net arch: (768x2 -> {})x2 -> 1, vertical axis mirroring".format(HIDDEN_SIZE))
    print("Checkpoint to load:", CHECKPOINT_TO_LOAD)
    print("Scale:", SCALE)
    print("Quantization: {}, {}".format(QA, QB))
    print()

    assert CHECKPOINT_TO_LOAD != None and os.path.exists(CHECKPOINT_TO_LOAD)

    net = PerspectiveNet768x2().to(DEVICE)
    net = torch.compile(net)

    checkpoint = torch.load(CHECKPOINT_TO_LOAD, weights_only=False)
    net.load_state_dict(checkpoint["model"])

    QA, QB = float(QA), float(QB)

    # Quantize net
    with torch.no_grad():
        net.ft_white.weight.data = torch.round(net.ft_white.weight.data * QA)
        net.ft_white.bias.data = torch.round(net.ft_white.bias.data * QA)

        net.ft_black.weight.data = torch.round(net.ft_black.weight.data * QA)
        net.ft_black.bias.data = torch.round(net.ft_black.bias.data * QA)

        net.hidden_to_out.weight.data = torch.round(net.hidden_to_out.weight.data * QB)
        net.hidden_to_out.bias.data = torch.round(net.hidden_to_out.bias.data * QA * QB)

    # Write quantized weights and biases to binary file

    out_file_path = CHECKPOINT_TO_LOAD[:-3] + ".bin"

    with open(out_file_path, 'wb') as out_file:
        def write_bin(weights_or_biases, np_type):
            flattened = weights_or_biases.detach().cpu().numpy().astype(np_type).flatten().tolist()
            letter = {np.int16: 'h', np.int32: 'i'}[np_type]
            out_file.write(struct.pack('<' + letter * len(flattened), *flattened))

        # Write features weights
        write_bin(net.ft_white.weight, np.int16)
        write_bin(net.ft_black.weight, np.int16)

        # Write hidden biases
        write_bin(net.ft_white.bias, np.int16)
        write_bin(net.ft_black.bias, np.int16)

        # Write output weights and output bias
        write_bin(net.hidden_to_out.weight[:, :HIDDEN_SIZE], np.int16)
        write_bin(net.hidden_to_out.weight[:, HIDDEN_SIZE:], np.int16)
        write_bin(net.hidden_to_out.bias, np.int32)

        print(out_file_path)

    # Remove quantization
    with torch.no_grad():
        net.ft_white.weight.data /= QA
        net.ft_white.bias.data /= QA

        net.ft_black.weight.data /= QA
        net.ft_black.bias.data /= QA

        net.hidden_to_out.weight.data /= QB
        net.hidden_to_out.bias.data /= QA * QB

    # Print some quantized evals

    print("\nQuantized evals:")

    FENS = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k1r1/8/ppp5/3p4/8/1B6/1B4N1/4K1R1 w - - 0 1",
        "r3k1r1/8/ppp5/3p4/8/1B6/1B4N1/4K1R1 b - - 0 1",
        "r5r1/3k4/ppp5/3p4/1K6/1B6/1B4N1/6R1 w - - 0 1",
        "r5r1/3k4/ppp5/3p4/1K6/1B6/1B4N1/6R1 b - - 0 1",
        "r5r1/3k4/ppp5/3p4/8/1B6/1B3KN1/6R1 w - - 0 1",
        "r5r1/3k4/ppp5/3p4/8/1B6/1B3KN1/6R1 b - - 0 1",
        "r5r1/8/ppp5/3p4/8/1B6/1B4Nk/1K4R1 w - - 0 1",
        "r5r1/8/ppp5/3p4/8/1B6/1B4Nk/1K4R1 b - - 0 1",
        "b2qk3/6pp/8/8/8/5Q1N/R7/4K3 w - - 0 1",
        "b2qk3/6pp/8/8/8/5Q1N/R7/4K3 b - - 0 1",
        "b3k3/6pp/8/8/8/1Q5N/Rq6/4K3 w - - 0 1",
        "b3k3/6pp/8/8/8/1Q5N/Rq6/4K3 b - - 0 1",
        "b3k3/6pp/5Q2/8/8/7N/R7/4K3 w - - 0 1",
        "b3k3/6pp/5Q2/8/8/7N/R7/4K3 b - - 0 1",
        "b3k3/6pp/8/8/3q4/7N/R7/4K3 w - - 0 1",
        "b3k3/6pp/8/8/3q4/7N/R7/4K3 b - - 0 1",
        "b3k3/5qpp/8/8/8/2Q4N/R1Q5/4K3 w - - 0 1",
        "b3k3/5qpp/8/8/8/2Q4N/R1Q5/4K3 b - - 0 1",
        "b3k1q1/6pp/8/8/8/1q5N/R7/4K3 w - - 0 1",
        "b3k1q1/6pp/8/8/8/1q5N/R7/4K3 b - - 0 1"
    ]

    for fen in FENS:
        eval_quantized = round(net.evaluate(fen) * float(SCALE))
        print('{', '"' + fen + '",', eval_quantized, '},')
