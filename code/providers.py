import abc
import typing
from itertools import chain, islice
from functools import cache
import warnings

import httpx_cache
import kloppy.domain.models
import kloppy.statsbomb
import kloppy.wyscout
import pandas as pd


class EventProvider(abc.ABC):
    base_url: str
    client: httpx_cache.Client

    def __init__(self) -> None:
        self.client = httpx_cache.Client(cache=httpx_cache.FileCache())

    @cache
    def _fetch(self, path: str, json=True) -> typing.Any:
        if json:
            return self.client.get(f"{self.base_url}/{path}.json").json()

        return self.client.get(f"{self.base_url}/{path}").text

    @abc.abstractmethod
    def all_match_ids(self, n=None) -> list[int]:
        pass

    @abc.abstractmethod
    def events(self, match_id: int | str) -> kloppy.domain.EventDataset:
        pass


class Wyscout(EventProvider):
    base_url = "https://raw.githubusercontent.com/koenvo/wyscout-soccer-match-event-dataset/main/processed/"

    def all_match_ids(self, n=None) -> list[int]:
        df = pd.read_table(
            "https://raw.githubusercontent.com/koenvo/wyscout-soccer-match-event-dataset/main/processed/README.md",
            sep="|",
        )
        match_id_strings = df[" ID "].str.extract(r"\[(\d*)\]")

        # convert to list of ints
        return match_id_strings.dropna().astype(int).pop(0).tolist()[:n]  # type: ignore

    def events(self, match_id: int | str = 2499841, **kwargs):
        return kloppy.wyscout.load_open_data(
            match_id, **kwargs, coordinates="statsbomb"
        )


class StatsBomb(EventProvider):
    base_url = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

    def competitions(self):
        return self._fetch("competitions")

    def matches(self, comp_id: int, season_id: int) -> list[dict]:
        return self._fetch(f"matches/{comp_id}/{season_id}")

    def match_ids(self, comp_id: int, season_id: int) -> list[int]:
        return [m["match_id"] for m in self.matches(comp_id, season_id)]

    def all_match_ids(self, n=None) -> list[int]:
        everything = chain.from_iterable(
            self.match_ids(comp["competition_id"], comp["season_id"])
            for comp in self.competitions()
        )
        return list(islice(everything, n))

    def events(self, match_id: int | str = 15946, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")  # suppress data use TOS warning
            return kloppy.statsbomb.load_open_data(match_id, **kwargs)


if __name__ == "__main__":
    sb = StatsBomb()
    ws = Wyscout()

    print("Statsbomb matches:", len(sb.all_match_ids()))
    print("Wyscout matches:", len(ws.all_match_ids()))
