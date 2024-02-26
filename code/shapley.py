from pathlib import Path

from captum.attr import GradientShap
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam
import numpy as np

from training import SportSequenceDataset, _load_latest_checkpoint, train_test_loaders
from nba import NBA_TENSOR_PATH
from model import SportSequenceModel


class WrapperModel(nn.Module):
    """Wrapper around our model that takes input as batch-first tensors.
    This is needed because the GradientShap class wants our model to take tensors,
    not packed sequences."""

    def __init__(self, model: SportSequenceModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """B x L x D -> B x L"""
        lengths = x.size(1) * torch.ones(x.size(0))
        packed_x = pack_padded_sequence(x, lengths, batch_first=True)
        packed_p = self.model.forward(packed_x)
        p, _ = pad_packed_sequence(packed_p, batch_first=True)
        return p


def get_model(d_seq):
    checkpoint_path = Path("../checkpoints/nba/rnn-md")

    rnn_md = SportSequenceModel(
        d_seq,
        rnn_type="rnn",
        D_rnn=128,
        bottom_hidden_layers=[128],
        top_hidden_layers=[128],
        rnn_layers=1,
    )

    _load_latest_checkpoint(checkpoint_path, rnn_md, Adam(rnn_md.parameters()))

    return WrapperModel(rnn_md)


def get_data(n_background=1000, n_test=10) -> tuple[torch.Tensor, torch.Tensor]:
    nba_data = SportSequenceDataset(NBA_TENSOR_PATH)

    test_loader, _ = train_test_loaders(nba_data, batch_size=n_background + n_test)

    packed_seqs, _ = next(iter(test_loader))

    padded_seqs, lengths = pad_packed_sequence(packed_seqs, batch_first=True)
    shortest_sequence_length = lengths.min()
    truncated_seqs = padded_seqs[:, :shortest_sequence_length]

    background = truncated_seqs[:n_background]
    test = truncated_seqs[n_background:]

    return background, test


def attributions():
    background, test_samples = get_data()
    model = get_model(background.size(2))
    explainer = GradientShap(model)

    # explain the 200th event in each sequence
    attributions = explainer.attribute(test_samples, background, target=200)

    # take the average contribution from each component over all games and time steps
    return attributions.mean(1).mean(0)


if __name__ == "__main__":
    attrs = attributions()
    np.savetxt(".shapley-vals.txt", attrs.numpy())
