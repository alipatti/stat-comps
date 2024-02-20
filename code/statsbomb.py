import os
from typing import Iterable
from zipfile import ZipFile
import json
from urllib.request import urlretrieve
import itertools

import polars as pl
import polars.selectors as cs
from kloppy import statsbomb
from tqdm import tqdm
import torch

from params import DATA_ROOT

STATSBOMB_ZIP_URL = (
    "https://github.com/statsbomb/open-data/archive/refs/heads/master.zip"
)

LOCAL_ZIP_PATH = DATA_ROOT / "statsbomb-events.zip"
LOCAL_DF_PATH = DATA_ROOT / "statsbomb-events.parquet"
STATSBOMB_TENSOR_PATH = DATA_ROOT / "statsbomb-tensors/"


def all_comps(zip) -> list[dict]:
    return json.load(zip.open("open-data-master/data/competitions.json"))


def matches_in_comp(comp_dict, zip):
    path = f"open-data-master/data/matches/{comp_dict['competition_id']}/{comp_dict['season_id']}.json"
    return json.load(zip.open(path))


def events_in_match(match_dict, zip: ZipFile) -> pl.DataFrame:
    match_id = match_dict["match_id"]
    event_path = f"open-data-master/data/events/{match_id}.json"
    lineup_path = f"open-data-master/data/lineups/{match_id}.json"

    sb_obj = statsbomb.load(zip.open(event_path), zip.open(lineup_path))
    df = sb_obj.to_df(engine="polars")

    home_team = str(match_dict["home_team"]["home_team_id"])
    score_diff = match_dict["home_score"] - match_dict["away_score"]
    result = "away_win" if score_diff < 0 else "home_win" if score_diff > 0 else "draw"

    # convert cols with team_id into home/away
    # e.g. 217 -> true, 318 -> false
    return df.with_columns(df.select(cs.contains("team")) == home_team).with_columns(
        pl.lit(f"{match_id}_{result}").alias("match_id")
    )


def events_in_matches(match_list: Iterable[dict], zip: ZipFile) -> pl.DataFrame:
    match_dfs = (
        events_in_match(i, zip) for i in tqdm(match_list, desc="Loading matches")
    )

    return (
        pl.concat(match_dfs, how="diagonal")
        .drop("event_id", "receiver_player_id", "player_id")
        .cast({pl.String: pl.Categorical})
        .to_dummies(cs.by_dtype(pl.Categorical).exclude("match_id"), drop_first=True)
        .cast({cs.numeric() | cs.boolean(): pl.Float32})  # type: ignore
    )


if __name__ == "__main__":
    if not os.path.exists(LOCAL_ZIP_PATH):
        print("Downloading statsbomb data (this may take a while)")
        print(f" - Source: {STATSBOMB_ZIP_URL}")
        print(f" - Local path: {LOCAL_ZIP_PATH}")
        path, response = urlretrieve(STATSBOMB_ZIP_URL, LOCAL_ZIP_PATH)

    if not os.path.exists(LOCAL_DF_PATH):
        zip = ZipFile(LOCAL_ZIP_PATH)

        comps = all_comps(zip)
        matches = list(
            itertools.chain.from_iterable(matches_in_comp(comp, zip) for comp in comps)
        )
        events = events_in_matches(matches, zip)

        events.write_parquet(LOCAL_DF_PATH)

    if not os.path.exists(STATSBOMB_TENSOR_PATH):
        events = pl.read_parquet(LOCAL_DF_PATH)
        matches = events.group_by(("match_id",))

        os.makedirs(STATSBOMB_TENSOR_PATH, exist_ok=True)

        for group, match_df in tqdm(
            matches,
            desc="Writing tensors",
            total=len(matches.first()),
        ):
            path: str = STATSBOMB_TENSOR_PATH / (group[0] + ".pt")  # type: ignore

            tensor = torch.from_numpy(match_df.drop("match_id").to_numpy())
            torch.save(tensor, path)
