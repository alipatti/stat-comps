import os
from pathlib import Path
from typing import Callable, Iterable, Any, Literal
import json

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from tqdm import tqdm
from statsbomb import STATSBOMB_TENSOR_PATH

from params import DEVICE
from model import SportSequenceModel
from nba import NBA_TENSOR_PATH

type LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class SportSequenceDataset(Dataset):
    def __init__(
        self,
        tensor_path: Path | str,
    ) -> None:
        self.paths = list(map(str, Path(tensor_path).glob("*.pt")))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[i]
        seq = torch.load(path)
        labels = (
            torch.ones(seq.size(0))
            if "home_win" in path
            else torch.zeros(seq.size(0))
            if "away_win" in path
            else 0.5 * torch.ones(seq.size(0))  # draw
        )

        return seq, labels


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
) -> dict[str, Any]:
    model.train()  # set to training mode

    losses = []
    n_correct, total = 0, 0

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
        predictions = model(x)
        loss = loss_function(predictions.data, y.data)

        # backward pass
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        n_correct += sum(y.data == (predictions.data > 0))
        total += len(predictions.data)

    avg_loss = np.mean(losses)
    accuracy = (n_correct / total).item()  # type: ignore
    print(f" - Average training loss: {avg_loss}")
    print(f" - Average training accuracy: {accuracy :.2%}")

    return dict(avg_train_loss=avg_loss, train_accuracy=accuracy)


def _test_loop(
    model: nn.Module,
    test_data: DataLoader,
    loss_function: LossFunction,
) -> dict[str, Any]:
    model.eval()  # set to eval mode

    losses = []
    n_correct, total = 0, 0

    for x, y in tqdm(
        test_data,
        total=len(test_data),
        leave=False,
        desc=" - Testing",
    ):
        # move to GPU (if it exists)
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # forward pass
        predictions = model(x)
        loss = loss_function(predictions.data, y.data)

        # update stats
        losses.append(loss.item())
        n_correct += sum(y.data == (predictions.data > 0))
        total += len(predictions.data)

    avg_loss = np.mean(losses)
    accuracy = (n_correct / total).item()  # type: ignore
    print(f" - Average test loss: {avg_loss}")
    print(f" - Average test accuracy: {accuracy :.2%}")

    return dict(avg_test_loss=avg_loss, test_accuracy=accuracy)


def _save_model(
    base_path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
) -> Path:
    os.makedirs(base_path, exist_ok=True)

    path = base_path / f"epoch-{epoch:>04}.pt"
    torch.save(
        dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            epoch=epoch,
        ),
        path,
    )

    return path


def _load_latest_checkpoint(
    base_path: Path | None,
    model: nn.Module,
    optimizer: Optimizer,
) -> tuple[Path | None, int]:
    if not base_path:
        return None, 1

    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
        return None, 1

    try:
        path = sorted(list(base_path.glob("epoch-*")))[-1]
    except IndexError:
        return None, 1  # no checkpoints in the directory

    state = torch.load(path)

    # load model state in place
    res = model.load_state_dict(state["model"])
    assert not res.missing_keys and not res.unexpected_keys

    res = optimizer.load_state_dict(state["optimizer"])
    assert not res

    return path, state["epoch"] + 1


def train(
    model: SportSequenceModel,
    data: Dataset,
    checkpoint_path: Path,
    optimizer_class=torch.optim.Adam,
    batch_size=500,
    epochs=80,
    checkpoint_every=5,
):
    model = model.to(DEVICE)
    optimizer = optimizer_class(model.parameters())

    initial_checkpoint, starting_epoch = _load_latest_checkpoint(
        checkpoint_path, model, optimizer
    )

    if initial_checkpoint:
        print(f"Loaded checkpoint from {initial_checkpoint}.")

    print("\n -:- MODEL -:- \n")
    print(model)

    # TODO: weight to prioritize being right towards the end of games
    loss_function = nn.BCEWithLogitsLoss()

    training_data, test_data = train_test_loaders(data, batch_size)

    print("\n -:- TRAINING -:- \n")
    print(f"Training for {epochs} epochs using {DEVICE.upper()}.")
    print(f"Saving checkpoints to {checkpoint_path}")
    print(f"Data: {len(data)} sequences of dimension {data[0][0].shape[1]}")  # type: ignore
    print(f"Starting on epoch {starting_epoch}.")

    for epoch in range(starting_epoch, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_stats = _train_loop(model, training_data, optimizer, loss_function)
        test_stats = _test_loop(model, test_data, loss_function)

        if checkpoint_path:
            with (checkpoint_path / "_training_stats.jsonl").open("at") as f:
                stats = dict(epoch=epoch) | train_stats | test_stats
                json.dump(stats, f)
                f.write("\n")

        if checkpoint_path and (epoch % checkpoint_every == 0):
            path = _save_model(checkpoint_path, model, optimizer, epoch)
            print(f" - Saved checkpoint to {path}")


def main_nba():
    EPOCHS = 20  # things seem to converge by this point

    nba_data = SportSequenceDataset(NBA_TENSOR_PATH)
    d_seq = nba_data[0][0].shape[1]  # dimension of event representations

    model_sizes: dict[str, tuple[int, list[int], int]] = {
        "xs": (32, [], 1),
        "sm": (64, [64], 1),
        "md": (128, [128], 1),
        "lg": (128, [128], 2),
        "xl": (128, [128, 128], 4),
    }

    rnn_types = ["lstm", "gru", "rnn"]

    for size, (D_rnn, hidden_layers, rnn_layers) in model_sizes.items():
        for rnn_type in rnn_types:
            model = SportSequenceModel(
                d_seq,
                rnn_type=rnn_type,  # type: ignore
                D_rnn=D_rnn,
                bottom_hidden_layers=hidden_layers,
                top_hidden_layers=hidden_layers,
                rnn_layers=rnn_layers,
            )

            model_name = f"{rnn_type}-{size}"

            train(
                model,
                nba_data,
                checkpoint_path=Path(f"../checkpoints/nba/{model_name}/"),
                epochs=EPOCHS,
            )


if __name__ == "__main__":
    main_nba()
