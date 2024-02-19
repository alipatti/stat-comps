import glob
from pathlib import Path

import polars as pl
import torch

from torch.utils.data import Dataset, DataLoader, random_split

from params import DEVICE

DATA_ROOT = Path("../data")
IN_PATH = DATA_ROOT / "nba-raw"
OUT_PATH = DATA_ROOT / "nba-tensors"


def make_nba_tensors():
    """Converts raw play-by-play data into sequences of tensors."""

    print("Loading raw play-by-play data...")

    glob_path = str(IN_PATH / "*.csv")
    seasons = [pl.read_csv(filepath) for filepath in glob.glob(glob_path)]
    seasons[1].drop_in_place("")  # weird extra column for some reason

    df = pl.concat(seasons, how="vertical_relaxed")

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
    ]

    print("Cleaning and aggregating...")

    games = list(
        df.with_columns(
            id=df["URL"].str.extract("/boxscores/(.*).html"),
            home_wins=df["HomeTeam"] == df["WinningTeam"],
        )
        .select("id", "home_wins", *categorical_cols, *numeric_cols)
        .cast({pl.String: pl.Categorical})
        .to_dummies(categorical_cols, drop_first=True)
        .fill_null(0)
        .fill_nan(0)
        .group_by(("id",))
    )

    print(f" - Found {len(df)} events across {len(games)} games.")

    print("Writing tensors...")

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


if __name__ == "__main__":
    make_nba_tensors()
