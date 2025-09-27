from settings import *
from feature_transformer import FeatureTransformer
import torch

class NetValuePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Features transformer (input layer -> hidden layer)
        self.ft = FeatureTransformer(INPUT_SIZE, HIDDEN_SIZE)

        # Hidden layer -> output layer
        self.hidden_to_out_value = torch.nn.Linear(HIDDEN_SIZE, 1)
        self.hidden_to_out_policy = torch.nn.Linear(HIDDEN_SIZE, POLICY_OUTPUT_SIZE)

        # Init random weights and biases
        torch.manual_seed(42)
        with torch.no_grad():
            torch.nn.init.uniform_(self.ft.weight, -0.1, 0.1)
            torch.nn.init.uniform_(self.ft.bias, -0.1, 0.1)

            torch.nn.init.uniform_(self.hidden_to_out_value.weight, -0.1, 0.1)
            torch.nn.init.uniform_(self.hidden_to_out_value.bias, -0.1, 0.1)

            torch.nn.init.uniform_(self.hidden_to_out_policy.weight, -0.1, 0.1)
            torch.nn.init.uniform_(self.hidden_to_out_policy.bias, -0.1, 0.1)

    def forward(self, stm_features_tensor, ntm_features_tensor, legal_moves_idxs_tensor):
        assert stm_features_tensor.dtype == ntm_features_tensor.dtype
        assert len(stm_features_tensor.size()) == len(ntm_features_tensor.size())
        assert legal_moves_idxs_tensor.dtype == torch.bool

        # [BATCH_SIZE, HIDDEN_SIZE]
        hidden_stm = self.ft(stm_features_tensor)
        hidden_ntm = self.ft(ntm_features_tensor)

        dim = len(stm_features_tensor.size()) - 1

        # [BATCH_SIZE, HIDDEN_SIZE * 2]
        hidden_layer = torch.cat([hidden_stm, hidden_ntm], dim=dim)

        # CReLU activation
        hidden_layer = torch.clamp(hidden_layer, 0, 1)

        # [BATCH_SIZE, HIDDEN_SIZE, 2]
        hidden_layer = hidden_layer.view(hidden_layer.shape[0], HIDDEN_SIZE, 2)

        # Activation: pairwise mul
        # [BATCH_SIZE, HIDDEN_SIZE]
        hidden_layer = hidden_layer[:, :, 0] * hidden_layer[:, :, 1]

        # [BATCH_SIZE, POLICY_OUTPUT_SIZE]
        pred_logits = self.hidden_to_out_policy(hidden_layer)
        pred_logits[legal_moves_idxs_tensor == False] = -10_000

        # Return predicted value and logits
        return self.hidden_to_out_value(hidden_layer), pred_logits

    def print_info(self):
        print("Net name:", NET_NAME)

        print("Net arch: ({}->{})x2--pairwise->{}->(1+{})".format(
            INPUT_SIZE,
            HIDDEN_SIZE,
            HIDDEN_SIZE,
            POLICY_OUTPUT_SIZE
        ))
