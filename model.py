from settings import *
import torch
from feature_transformer import FeatureTransformerSlice

class PerspectiveNet768x2(torch.nn.Module):
    def __init__(self):
        super().__init__()

        NUM_INPUT_BUCKETS = max(INPUT_BUCKETS_MAP) + 1

        # Features transformers (input layer -> hidden layer)
        self.ft_white = FeatureTransformerSlice(768 * NUM_INPUT_BUCKETS, HIDDEN_SIZE)
        self.ft_black = FeatureTransformerSlice(768 * NUM_INPUT_BUCKETS, HIDDEN_SIZE)

        # Hidden layer -> output layer
        self.hidden_to_out = torch.nn.Linear(HIDDEN_SIZE * 2, 1)

        # Init hidden_to_out random weights and biases
        # Feature transformers weights and biases are initialized in FeatureTransformerSlice constructor
        torch.manual_seed(42)
        with torch.no_grad():
            torch.nn.init.uniform_(self.hidden_to_out.weight, -0.1, 0.1)
            torch.nn.init.uniform_(self.hidden_to_out.bias, -0.1, 0.1)

    # The arguments should be dense tensors and not sparse tensors, as the former are way faster
    @torch.compile
    def forward(self,
        features_tensor_white: torch.Tensor,
        features_tensor_black: torch.Tensor,
        is_white_stm_tensor: torch.Tensor):

        assert features_tensor_white.dtype == torch.int32
        assert features_tensor_black.dtype == torch.int32
        assert is_white_stm_tensor.dtype == torch.bool

        hidden_white = self.ft_white(features_tensor_white)
        hidden_black = self.ft_black(features_tensor_black)

        assert len(features_tensor_white.size()) == len(features_tensor_black.size())
        dim = len(features_tensor_white.size()) - 1

        # stm accumulator first
        hidden_layer  =  is_white_stm_tensor * torch.cat([hidden_white, hidden_black], dim=dim)
        hidden_layer += ~is_white_stm_tensor * torch.cat([hidden_black, hidden_white], dim=dim)

        # SCReLU activation
        hidden_layer = torch.pow(torch.clamp(hidden_layer, 0, 1), 2)

        return self.hidden_to_out(hidden_layer)

    def clamp_weights_biases(self):
        with torch.no_grad():
            self.ft_white.weight.clamp_(-MAX_WEIGHT_BIAS, MAX_WEIGHT_BIAS)
            self.ft_white.bias.clamp_(-MAX_WEIGHT_BIAS, MAX_WEIGHT_BIAS)

            self.ft_black.weight.clamp_(-MAX_WEIGHT_BIAS, MAX_WEIGHT_BIAS)
            self.ft_black.bias.clamp_(-MAX_WEIGHT_BIAS, MAX_WEIGHT_BIAS)

            self.hidden_to_out.weight.clamp_(-MAX_WEIGHT_BIAS, MAX_WEIGHT_BIAS)
            self.hidden_to_out.bias.clamp_(-MAX_WEIGHT_BIAS, MAX_WEIGHT_BIAS)

    def evaluate(self, fen: str):
        import chess

        features_tensor_white = torch.zeros((1, 32), device=DEVICE, dtype=torch.int32) - 1
        features_tensor_black = torch.zeros((1, 32), device=DEVICE, dtype=torch.int32) - 1

        board = chess.Board(fen.strip())

        mirror_v_axis_white = board.king(chess.WHITE) % 8 > 3
        mirror_v_axis_black = board.king(chess.BLACK) % 8 > 3

        white_queens = board.pieces(chess.QUEEN, chess.WHITE)
        white_queen_square = None if len(white_queens) != 1 else next(iter(white_queens), None)

        black_queens = board.pieces(chess.QUEEN, chess.BLACK)
        black_queen_square = None if len(black_queens) != 1 else next(iter(black_queens), None)

        if mirror_v_axis_white and black_queen_square != None:
            black_queen_square ^= 7

        if mirror_v_axis_black and white_queen_square != None:
            white_queen_square ^= 7

        white_bucket = 0 if black_queen_square == None else INPUT_BUCKETS_MAP[black_queen_square]
        black_bucket = 0 if white_queen_square == None else INPUT_BUCKETS_MAP[white_queen_square]

        pieces_processed = 0

        for square, colored_piece in board.piece_map().items():
            piece_type = colored_piece.piece_type # P=1, N=2, B=3, R=4, Q=5, K=6

            feature_idx_white = white_bucket * 768
            feature_idx_white += (colored_piece.color == chess.BLACK) * 384 + (piece_type - 1) * 64
            feature_idx_white += square ^ 7 if mirror_v_axis_white else square

            feature_idx_black = black_bucket * 768
            feature_idx_black += (colored_piece.color == chess.BLACK) * 384 + (piece_type - 1) * 64
            feature_idx_black += square ^ 7 if mirror_v_axis_black else square

            features_tensor_white[0][pieces_processed] = feature_idx_white
            features_tensor_black[0][pieces_processed] = feature_idx_black

            pieces_processed += 1

        prediction = self.forward(
            features_tensor_white,
            features_tensor_black,
            torch.tensor(board.turn == chess.WHITE, device=DEVICE, dtype=torch.bool)
        )

        return float(prediction)
