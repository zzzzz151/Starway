from settings import *
from model import NetValuePolicy
import os
import torch
import numpy as np
import struct
import chess

# Flip board vertically if black to move
def get_board_oriented(board):
    if board.turn == chess.WHITE:
        return board.copy()

    new_board = chess.Board(None)
    new_board.turn = chess.WHITE

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            flipped_piece = chess.Piece(piece.piece_type, not piece.color)
            new_board.set_piece_at(square ^ 56, flipped_piece)

    if board.has_kingside_castling_rights(chess.BLACK):
        new_board.castling_rights |= chess.BB_H1

    if board.has_queenside_castling_rights(chess.BLACK):
        new_board.castling_rights |= chess.BB_A1

    if board.ep_square:
        new_board.ep_square = board.ep_square ^ 56

    return new_board

with open("moves_map_1880.bin", "rb") as f:
    MOVES_MAP_1880 = np.frombuffer(f.read(), dtype=np.int16)
    MOVES_MAP_1880 = MOVES_MAP_1880.reshape((64, 64, 7))

def get_move_idx1882(move):
    promo_idx = move.promotion - 1 if move.promotion else 6
    idx1880 = MOVES_MAP_1880[move.from_square][move.to_square][promo_idx]
    assert idx1880 >= 0 and idx1880 < 1880

    if board.is_castling(move):
        return 1880 + (move.to_square > move.from_square)

    return idx1880

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    print("Device:", "CPU" if DEVICE == torch.device("cpu") else torch.cuda.get_device_name(0))
    print("Net name:", NET_NAME)
    print("Net arch: (768 -> {})x2 -> 1883".format(HIDDEN_SIZE))
    print("Checkpoint to load:", CHECKPOINT_TO_LOAD)
    print("Scale:", SCALE)
    print("FT weights/biases clipping: [{}, {}]".format(-FT_MAX_WEIGHT_BIAS, FT_MAX_WEIGHT_BIAS))
    print("FT quantization:", FT_Q)
    print()

    assert CHECKPOINT_TO_LOAD != None and os.path.exists(CHECKPOINT_TO_LOAD)

    net = NetValuePolicy().to(DEVICE)
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
        write_bin(net.ft.weight.T, np.int16)
        write_bin(net.ft.bias, np.int16)

        # Write value head weights and biases
        write_bin(net.hidden_to_out_value.weight[:, :HIDDEN_SIZE], np.float32)
        write_bin(net.hidden_to_out_value.weight[:, HIDDEN_SIZE:], np.float32)
        write_bin(net.hidden_to_out_value.bias, np.float32)

        # Write policy head weights and biases
        for i in range(1882):
            write_bin(net.hidden_to_out_policy.weight[i][:HIDDEN_SIZE], np.float32)
            write_bin(net.hidden_to_out_policy.weight[i][HIDDEN_SIZE:], np.float32)
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

    # Remove FT quantization
    with torch.no_grad():
        net.ft.weight.data /= FT_Q
        net.ft.bias.data /= FT_Q

    # Finally, print the quantized outputs for some FENs

    print("\nQuantized outputs:")

    FENS = [
        # Start pos (and flipped version)
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
        # Kiwipete modified (and flipped version)
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w Kq - 0 1",
        "r3k2r/pppbbppp/2n2q1P/1P2p3/3pn3/BN2PNP1/P1PPQPB1/R3K2R b Qk - 0 1",
        # Mate in 1 with promos and en passant (and flipped version)
        "6q1/5P2/8/2Pp4/8/6R1/8/5K1k w - d6 0 1",
        "5k1K/8/6r1/8/2pP4/8/5p2/6Q1 b - d3 0 1"
    ]

    for fen in FENS:
        print()
        print(fen)

        board = chess.Board(fen.strip())
        board_oriented = get_board_oriented(board)

        stm_features_tensor = torch.zeros(768, device=DEVICE, dtype=torch.float32)
        ntm_features_tensor = torch.zeros(768, device=DEVICE, dtype=torch.float32)

        for square, colored_piece in board_oriented.piece_map().items():
            piece_type = colored_piece.piece_type - 1 # Minus 1 so it's 0-5

            idx = (colored_piece.color == chess.BLACK) * 384 + piece_type * 64 + square
            stm_features_tensor[idx] = 1

            idx = (colored_piece.color == chess.WHITE) * 384 + piece_type * 64 + (square ^ 56)
            ntm_features_tensor[idx] = 1

        legal_moves_idx1882_tensor = torch.zeros(1882, device=DEVICE, dtype=torch.bool)

        for move in board_oriented.legal_moves:
            move_idx1882 = get_move_idx1882(move)
            legal_moves_idx1882_tensor[move_idx1882] = True

        pred_value, pred_logits = net.forward(
            stm_features_tensor, ntm_features_tensor, legal_moves_idx1882_tensor
        )

        pred_policy = torch.nn.functional.softmax(pred_logits, dim=0, dtype=pred_logits.dtype)

        legal_moves_policy = []
        for move in board.legal_moves:
            oriented_move = chess.Move(move.from_square, move.to_square, move.promotion)
            if board.turn == chess.BLACK:
                oriented_move.from_square ^= 56
                oriented_move.to_square ^= 56

            move_idx1882 = get_move_idx1882(oriented_move)

            logit = float(pred_logits[move_idx1882])
            move_policy = float(pred_policy[move_idx1882])

            legal_moves_policy.append((move, logit, move_policy))

        legal_moves_policy.sort(key=lambda x: x[2], reverse=True)

        print("Value:", round(float(pred_value) * float(SCALE)))

        for (move, logit, move_policy) in legal_moves_policy:
            print("{:<5}: {:6.2f} | {:.2f}".format(move.uci(), logit, move_policy))
