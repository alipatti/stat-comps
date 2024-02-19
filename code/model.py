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


class SportSequenceModel(nn.Module):
    D_seq: int  # dimension of event sequence
    D_rnn: int  # dimension of RNN hidden state
    N: int  # batch size

    def __init__(self, D_seq, D_rnn):
        super().__init__()

        self.rnn = nn.GRU(D_seq, D_rnn)
        self.rnn_to_logit = nn.Linear(D_rnn, 1)

    def forward(self, packed_seqs: PackedSequence) -> PackedSequence:
        """Takes list of sequences of dimension D_in and returns a list of sequences of dimension 1."""

        packed_rnn_output, _ = self.rnn(packed_seqs)

        # apply dense layer to transform RNN output to a win-probability logit
        packed_logits = pack_data_like(
            data=self.rnn_to_logit(packed_rnn_output.data).reshape(-1),
            like=packed_rnn_output,
        )

        return packed_logits
