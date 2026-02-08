from settings import *
from model import NetValuePolicy
import numpy as np
import chess
import torch

FENS = [
    # Start pos
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # Kiwipete modified
    "r2k3r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w K - 0 1",
    # Mate in 1 with promos and en passant
    "3q4/4P3/8/5Pp1/8/2R5/2K5/k7 w - g6 0 1",
    # In check (test net bucket)
    "3b2k1/4P1p1/1K6/8/8/8/8/8 w - - 0 1",
]

def get_flipped_board(original_board: chess.Board) -> chess.Board:
    flipped_board = chess.Board.empty()

    for square, piece in original_board.piece_map().items():
        flipped_piece = chess.Piece(piece.piece_type, not piece.color)
        flipped_board.set_piece_at(square ^ 56, flipped_piece)

    flipped_board.turn = not original_board.turn

    # Flip castling rights
    if original_board.has_kingside_castling_rights(chess.WHITE):
        flipped_board.castling_rights |= chess.BB_H8
    if original_board.has_queenside_castling_rights(chess.WHITE):
        flipped_board.castling_rights |= chess.BB_A8
    if original_board.has_kingside_castling_rights(chess.BLACK):
        flipped_board.castling_rights |= chess.BB_H1
    if original_board.has_queenside_castling_rights(chess.BLACK):
        flipped_board.castling_rights |= chess.BB_A1

    if original_board.ep_square is not None:
        flipped_board.ep_square = original_board.ep_square ^ 56

    flipped_board.halfmove_clock = original_board.halfmove_clock
    flipped_board.fullmove_number = original_board.fullmove_number

    return flipped_board

def ranks_flipped(move: chess.Move) -> chess.Move:
    return chess.Move(move.from_square ^ 56, move.to_square ^ 56, move.promotion)

def files_flipped(move: chess.Move) -> chess.Move:
    return chess.Move(move.from_square ^ 7, move.to_square ^ 7, move.promotion)

def get_move_idx(move: chess.Move, board_oriented: chess.Board) -> int:
    assert board_oriented.turn == chess.WHITE

    piece_moved = board_oriented.piece_at(move.from_square)
    assert piece_moved
    pt_moved = piece_moved.piece_type - 1 # 0 inclusive to 5 inclusive
    assert pt_moved >= 0 and pt_moved < 6

    dst_for_idx = move.to_square

    if move.promotion and move.promotion != chess.QUEEN:
        dst_for_idx ^= 56

    if board_oriented.king(board_oriented.turn) % 8 <= 3:
        dst_for_idx ^= 7

    if board_oriented.is_en_passant(move):
        pt_captured = 0
    elif board_oriented.is_capture(move):
        piece_captured = board_oriented.piece_at(move.to_square)
        assert piece_captured
        pt_captured = piece_captured.piece_type - 1
        assert pt_captured >= 0 and pt_captured < 5
    else:
        pt_captured = 5

    idx = pt_moved * 6 * 64 + dst_for_idx * 6 + pt_captured
    assert idx >= 0 and idx < POLICY_OUTPUT_SIZE
    return idx

# The policy is a dictionary that maps chess.Move to tuple (logit: float, move_policy: float)
def get_value_and_policy(net: NetValuePolicy, board: chess.Board) -> (int, dict):
    board_oriented = board.copy() if board.turn == chess.WHITE else get_flipped_board(board)

    stm_features_tensor = torch.zeros(32, device=DEVICE, dtype=torch.int32) - 1
    ntm_features_tensor = stm_features_tensor.clone()

    stm_xor = 7 if board_oriented.king(chess.WHITE) % 8 <= 3 else 0
    ntm_xor = 56 ^ (7 if board_oriented.king(chess.BLACK) % 8 <= 3 else 0)

    for i, (square, colored_piece) in enumerate(board_oriented.piece_map().items()):
        piece_type = colored_piece.piece_type - 1 # Minus 1 so it's 0-5 inclusive

        idx = (colored_piece.color == chess.BLACK) * 384 + piece_type * 64 + (square ^ stm_xor)
        stm_features_tensor[i] = board.is_check() * 768 + idx

        idx = (colored_piece.color == chess.WHITE) * 384 + piece_type * 64 + (square ^ ntm_xor)
        ntm_features_tensor[i] = board.is_check() * 768 + idx

    num_legal_moves = len(list(board.legal_moves))
    assert num_legal_moves > 0 and num_legal_moves <= MAX_MOVES_PER_POS

    legal_logits_idxs_tensor = torch.zeros(MAX_MOVES_PER_POS, dtype=torch.int32, device=DEVICE) - 1

    for i, move in enumerate(board_oriented.legal_moves):
        legal_logits_idxs_tensor[i] = get_move_idx(move, board_oriented)

    pred_value, pred_logits = net.forward(
        stm_features_tensor.unsqueeze(0),
        ntm_features_tensor.unsqueeze(0),
        legal_logits_idxs_tensor.unsqueeze(0)
    )

    pred_value = float(pred_value[0].detach())
    pred_policy = torch.nn.functional.softmax(pred_logits[0], dim=0).tolist()
    pred_logits = pred_logits[0].detach().tolist()

    legal_moves_policy = dict()

    for i, move in enumerate(board_oriented.legal_moves):
        if board.turn == chess.BLACK:
            move = ranks_flipped(move)

        legal_moves_policy[move] = (pred_logits[i], pred_policy[i])

    return pred_value, legal_moves_policy

def print_net_output(net: NetValuePolicy, fens: list[str]):
    for fen in fens:
        fen = fen.strip()
        print(fen)

        board = chess.Board(fen)
        value, legal_moves_policy = get_value_and_policy(net, board)

        print("Value:", round(value * float(VALUE_SCALE)))

        for move, (logit, move_policy) in sorted(
            legal_moves_policy.items(),
            key=lambda item: item[1][1],
            reverse=True
        ):
            print("{:<5}: {:6.2f} | {:.2f}".format(move.uci(), logit, move_policy))

        value2, legal_moves_policy2 = get_value_and_policy(net, get_flipped_board(board))

        assert abs(value - value2) < 0.001

        for move, (logit, move_policy) in legal_moves_policy.items():
            move = ranks_flipped(move)

            assert move in legal_moves_policy2
            assert abs(logit - legal_moves_policy2[move][0]) < 0.001
            assert abs(move_policy - legal_moves_policy2[move][1]) < 0.001

        print()

if __name__ == "__main__":
    net = NetValuePolicy().to(DEVICE)
    net.print_info()

    print("Device:", "CPU" if DEVICE == torch.device("cpu") else torch.cuda.get_device_name(0))
    print("Checkpoint to load:", CHECKPOINT_TO_LOAD)
    print("Value scale:", VALUE_SCALE)
    print()

    assert CHECKPOINT_TO_LOAD != None and os.path.exists(CHECKPOINT_TO_LOAD)

    net = torch.compile(net)

    checkpoint = torch.load(CHECKPOINT_TO_LOAD, weights_only=False)
    net.load_state_dict(checkpoint["model"])

    print_net_output(net, FENS)
