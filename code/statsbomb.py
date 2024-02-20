import os
from typing import Iterable
from zipfile import ZipFile
import json
from pprint import pprint

import polars as pl
import polars.selectors as cs
from kloppy import statsbomb
from tqdm import tqdm
import torch

from nba import DATA_ROOT

IN_PATH = DATA_ROOT / "statsbomb-events.zip"
OUT_PATH = DATA_ROOT / "statsbomb-tensors/"


def all_comps(zip):
    return json.load(zip.open("open-data-master/data/competitions.json"))


def matches_in_comp(comp_dict, zip):
    path = f"open-data-master/data/matches/{comp_dict['competition_id']}/{comp_dict['season_id']}.json"
    return json.load(zip.open(path))


def events_in_match(match_dict, zip: ZipFile) -> pl.DataFrame:
    match_id = match_dict["match_id"]
    event_path = f"open-data-master/data/events/{match_id}.json"
    lineup_path = f"open-data-master/data/lineups/{match_id}.json"

    sb_obj = statsbomb.load(zip.open(event_path), zip.open(lineup_path))
    df = pl.from_pandas(sb_obj.to_df())

    home_team = str(match_dict["home_team"]["home_team_id"])
    score_diff = match_dict["home_score"] - match_dict["away_score"]
    result = "away_win" if score_diff < 0 else "home_win" if score_diff > 0 else "draw"

    return (
        # convert team_id to home/away
        df.with_columns(df.select(cs.contains("team")) == home_team).with_columns(
            pl.lit(f"{match_id}_{result}").alias("match_id")
        )
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
    zip = ZipFile(IN_PATH)

    comps = all_comps(zip)
    matches = matches_in_comp(comps[0], zip)[:100]
    events = events_in_matches(matches, zip)

    match_groups = tqdm(
        events.group_by(("match_id",)),
        desc="Writing tensors",
        total=len(matches),
    )

    os.makedirs(OUT_PATH, exist_ok=True)

    for group, match_df in match_groups:
        path: str = OUT_PATH / (group[0] + ".pt")  # type: ignore

        tensor = torch.from_numpy(match_df.drop("match_id").to_numpy())
        torch.save(tensor, path)
