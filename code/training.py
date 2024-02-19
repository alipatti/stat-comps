# TODO: clean up imports

from pathlib import Path
from typing import Callable, Iterable
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm import tqdm
import torch
import numpy as np

from params import DEVICE, MODEL_CHECKPOINT_PATH
from nba import NBADataset
from model import SportSequenceModel

type LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def pack_sequences(
    seqs_and_labels: Iterable[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[PackedSequence, PackedSequence]:
    """Converts an iterable of sequences and sequence labels
    into PackedSequences for RNN training."""
    seqs, labels = zip(*seqs_and_labels)

    packed_seqs = torch.nn.utils.rnn.pack_sequence(seqs, enforce_sorted=False)
    packed_labels = torch.nn.utils.rnn.pack_sequence(labels, enforce_sorted=False)

    return packed_seqs, packed_labels


def train_test_loaders(
    dataset: Dataset,
    batch_size: int,
    train_prop=0.2,
    shuffle=True,
) -> tuple[DataLoader, DataLoader]:
    train_data, test_data = random_split(
        dataset,
        [1 - train_prop, train_prop],
        torch.Generator().manual_seed(67),  # reproducable train/test split
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=pack_sequences,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=pack_sequences,
    )

    return train_loader, test_loader


def _train_loop(
    model: nn.Module,
    training_data: DataLoader,
    optimizer: Optimizer,
    loss_function: LossFunction,
):
    model.train()  # set to training mode

    losses = []
    for x, y in tqdm(
        training_data,
        total=len(training_data),
        leave=False,
        desc=" - Training",
    ):
        x: PackedSequence
        y: PackedSequence

        # move to GPU (if it exists)
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # forward pass
        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_function(outputs.data, y.data)

        # backward pass
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print(f" - Average training loss: {np.mean(losses)}")


def _test_loop(
    model: nn.Module,
    test_data: DataLoader,
    loss_function: LossFunction,
):
    model.eval()  # set to eval mode
    losses = []

    for x, y in tqdm(
        test_data,
        total=len(test_data),
        leave=False,
        desc=" - Testing",
    ):
        # forward pass
        outputs = model(x)
        loss = loss_function(outputs.data, y.data)
        losses.append(loss.item())

    print(f" - Average test loss: {np.mean(losses)}")


def train(
    model: nn.Module,
    data: Dataset,
    optimizer_class=torch.optim.Adam,
    batch_size=500,
    epochs=10,
    save_checkpoints_every=10,
):
    model = model.to(DEVICE)
    optimizer = optimizer_class(model.parameters())

    # TODO: weight to prioritize being right towards the end of games
    loss_function = nn.BCEWithLogitsLoss()

    training_data, test_data = train_test_loaders(data, batch_size)

    print(f"Training {model} on {len(data)} sequences.")  # type: ignore
    print(f"Using {DEVICE} device.")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        _train_loop(model, training_data, optimizer, loss_function)
        _test_loop(model, test_data, loss_function)

        # TODO: save model checkpoint


if __name__ == "__main__":
    # TODO: use all data
    data = Subset(NBADataset(), list(range(200)))

    sequence_dimension = data[0][0].shape[1]  # dimension of event representations
    rnn_hidden_dimension = 32

    model = SportSequenceModel(sequence_dimension, rnn_hidden_dimension)

    train(model, data, batch_size=50)
