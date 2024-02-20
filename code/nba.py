import glob
import os
from pathlib import Path
from zipfile import ZipFile

import polars as pl
from polars.dataframe.group_by import GroupBy
import torch

from params import DATA_ROOT

NBA_DATA_PATH = DATA_ROOT / "nba-raw"
NBA_TENSOR_PATH = DATA_ROOT / "nba-tensors"


def raw_event_df() -> pl.DataFrame:
    print("Loading raw play-by-play data...")

    if not NBA_DATA_PATH.exists():
        print("Extracting NBA zip file ...")
        with ZipFile(DATA_ROOT / "kaggle-bbref-pbp.zip") as f:
            f.extractall(NBA_DATA_PATH)

    glob_path = str(NBA_DATA_PATH / "*.csv")
    seasons = [pl.read_csv(filepath) for filepath in glob.glob(glob_path)]

    return pl.concat(seasons, how="diagonal_relaxed").drop("")


def game_dfs(events: pl.DataFrame) -> GroupBy:
    numeric_cols = [
        "SecLeft",
        "AwayScore",
        "HomeScore",
        "ShotDist",
    ]
    categorical_cols = [
        "ShotType",
        "ShotOutcome",
        "FreeThrowOutcome",
        "FoulType",
        "ReboundType",
        "ViolationType",
        "TurnoverType",
        "TurnoverCause",
    ]

    print("Cleaning and aggregating...")

    filtered_events = (
        events.with_columns(
            id=events["URL"].str.extract("/boxscores/(.*).html"),
            home_wins=events["HomeTeam"] == events["WinningTeam"],
        )
        .select("id", "home_wins", *categorical_cols, *numeric_cols)
        .cast({pl.String: pl.Categorical})
        .to_dummies(categorical_cols, drop_first=True)
        .fill_nan(None)  # replace nan with null
        .fill_null(0)  # fill null with zeros
    )

    games = filtered_events.group_by(("id",))

    n_events = games.len()["len"].sum()
    n_games = len(games.first())
    dimension = next(iter(games))[1].shape[1]

    print(f" - Found {n_events} relevant events across {n_games} games.")
    print(f" - Event dimension: {dimension - 2}")

    return games


def write_tensors(games: GroupBy):
    """Converts raw play-by-play data into sequences of tensors."""

    os.makedirs(NBA_TENSOR_PATH, exist_ok=True)

    print(f"Writing tensors to {NBA_TENSOR_PATH}...")

    for group, game in games:
        seq = game.drop("id", "home_wins").to_numpy().astype("float32")
        game_id: str = group[0]  # type: ignore
        filename = (
            game_id + ("_home_win" if game["home_wins"][0] else "_away_win") + ".pt"
        )
        torch.save(torch.from_numpy(seq), NBA_TENSOR_PATH / filename)

    print("Done!")


def naive_accuracy(games: GroupBy) -> float:
    """Predicts whoever is winning at that moment."""

    n_correct, total = 0, 0

    for _, game in games:
        predictions = game["HomeScore"] > game["AwayScore"]

        n_correct += (predictions == game["home_wins"]).sum()
        total += len(game)

    return n_correct / total


if __name__ == "__main__":
    events = raw_event_df()
    games = game_dfs(events)

    print(f"Naive accuracy: {naive_accuracy(games):.3%}")

    if not os.path.exists(NBA_TENSOR_PATH):
        write_tensors(games)
