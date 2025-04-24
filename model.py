from settings import *
import torch
from feature_transformer import FeatureTransformerSlice

class PerspectiveNet768x2(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Features transformers (input layer -> hidden layer)
        self.ft_white = FeatureTransformerSlice(768 * NUM_INPUT_BUCKETS, HIDDEN_SIZE)
        self.ft_black = FeatureTransformerSlice(768 * NUM_INPUT_BUCKETS, HIDDEN_SIZE)

        # Hidden layer -> output layer
        self.hidden_to_out = torch.nn.Linear(HIDDEN_SIZE * 2, 1)

        # Init hidden_to_out random weights and biases
        # Feature transformers weights and biases
        # are initialized in FeatureTransformerSlice constructor
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

        features_tensor_w = torch.zeros((1, 32), device=DEVICE, dtype=torch.int32) - 1
        features_tensor_b = features_tensor_w.clone()

        board = chess.Board(fen.strip())

        mirror_v_axis_w = board.king(chess.WHITE) % 8 > 3
        mirror_v_axis_b = board.king(chess.BLACK) % 8 > 3

        def get_input_bucket(color):
            opp_queens = board.pieces(chess.QUEEN, not color)

            if len(opp_queens) == 0:
                return 0
            elif len(opp_queens) > 1:
                return NUM_INPUT_BUCKETS - 1
            else:
                opp_q_sq = next(iter(opp_queens), None)

                if board.king(color) % 8 > 3:
                    opp_q_sq ^= 7

                return INPUT_BUCKETS_MAP[opp_q_sq]

        bucket_w = get_input_bucket(chess.WHITE)
        bucket_b = get_input_bucket(chess.BLACK)

        for i, (square, colored_piece) in enumerate(board.piece_map().items()):
            piece_type = colored_piece.piece_type - 1

            base_feature = (colored_piece.color == chess.BLACK) * 384 + piece_type * 64

            sq_w = square ^ 7 if mirror_v_axis_w else square
            sq_b = square ^ 7 if mirror_v_axis_b else square

            features_tensor_w[0][i] = bucket_w * 768 + base_feature + sq_w
            features_tensor_b[0][i] = bucket_b * 768 + base_feature + sq_b

        prediction = self.forward(
            features_tensor_w,
            features_tensor_b,
            torch.tensor(board.turn == chess.WHITE, device=DEVICE, dtype=torch.bool)
        )

        return float(prediction)
