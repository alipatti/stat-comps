from itertools import pairwise
from typing import Literal

from torch import nn
from torch.nn.utils.rnn import PackedSequence
import torch


def pack_data_like(data: torch.Tensor, like: PackedSequence):
    return PackedSequence(
        data=data,
        batch_sizes=like.batch_sizes,
        sorted_indices=like.sorted_indices,
        unsorted_indices=like.unsorted_indices,
    )


class MLP(nn.Module):
    """Multi-layer perceptron"""

    def __init__(
        self,
        layer_dimensions: list[int],
        activation_function=nn.ReLU,
    ):
        super().__init__()

        layers: list[nn.Module] = []

        for d_in, d_out in pairwise(layer_dimensions):
            layers.append(activation_function())
            layers.append(nn.Linear(d_in, d_out))

        self.mlp = nn.Sequential(*layers[1:])  # ignore the first non-linearity

    def forward(self, x):
        return self.mlp(x)


class SportSequenceModel(nn.Module):
    def __init__(
        self,
        D_seq,
        D_rnn=64,
        bottom_hidden_layers=[],  # directly from events to RNN
        top_hidden_layers=[],  # directly from RNN to probabilities
        rnn_type: Literal["gru", "lstm", "rnn"] = "gru",
        rnn_layers=1,
    ):
        super().__init__()

        # module for mapping x_t into the RNN
        self.top_mlp = MLP([D_seq] + top_hidden_layers + [D_rnn])

        # the RNN itself
        self.rnn = (
            nn.GRU(D_rnn, D_rnn, num_layers = rnn_layers)
            if rnn_type == "gru"
            else nn.LSTM(D_rnn, D_rnn, num_layers = rnn_layers)
            if rnn_type == "lstm"
            else nn.RNN(D_rnn, D_rnn, num_layers = rnn_layers)
        )

        # module for taking hidden state to win probabilities
        self.bottom_mlp = MLP([D_rnn] + bottom_hidden_layers + [1])

    def forward(self, x: PackedSequence) -> PackedSequence:
        """Takes a PackedSequences of dimension D_in (representing the game events) and returns a PackedSequence of dimension 1 (representing the logits of the win probabilties)."""

        # send the events through the top MLP
        x = pack_data_like(
            data=self.top_mlp(x.data),
            like=x,
        )

        # send the transformed events through the rnn
        h, _ = self.rnn(x)

        # send the hidden states through the bottom mlp to get probabilities
        p = pack_data_like(
            data=self.bottom_mlp(h.data).reshape(-1),
            like=h,
        )

        return p
