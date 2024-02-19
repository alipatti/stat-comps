import glob
import os
from pathlib import Path
from zipfile import ZipFile

import polars as pl
from polars.dataframe.group_by import GroupBy
import torch
from torch.utils.data import Dataset

DATA_ROOT = Path("../data")
IN_PATH = DATA_ROOT / "nba-raw"
OUT_PATH = DATA_ROOT / "nba-tensors"


def raw_event_df() -> pl.DataFrame:
    print("Loading raw play-by-play data...")

    if not IN_PATH.exists():
        print("Extracting NBA zip file ...")
        with ZipFile(DATA_ROOT / "kaggle-bbref-pbp.zip") as f:
            f.extractall(IN_PATH)

    glob_path = str(IN_PATH / "*.csv")
    seasons = [pl.read_csv(filepath) for filepath in glob.glob(glob_path)]
    seasons[1].drop_in_place("")  # weird extra column for some reason

    return pl.concat(seasons, how="vertical_relaxed")


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

    os.makedirs(OUT_PATH, exist_ok=True)

    print(f"Writing tensors to {OUT_PATH}...")

    for group, game in games:
        seq = game.drop("id", "home_wins").to_numpy().astype("float32")
        game_id: str = group[0]  # type: ignore
        filename = (
            game_id + ("_home_win" if game["home_wins"][0] else "_away_win") + ".pt"
        )
        torch.save(torch.from_numpy(seq), OUT_PATH / filename)

    print("Done!")


class NBADataset(Dataset):
    def __init__(
        self,
        seq_path: Path | str = OUT_PATH,
    ) -> None:
        self.paths = list(map(str, Path(OUT_PATH).glob("*.pt")))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[i]
        seq = torch.load(path)
        labels = (
            torch.ones(seq.size(0))
            if str(path).split("_")[1] == "home"
            else torch.zeros(seq.size(0))
        )

        return seq, labels


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

    if not os.path.exists(OUT_PATH):
        write_tensors(games)
