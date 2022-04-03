import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict

import pandas as pd
from tqdm import tqdm

DEFAULT_POLIMORFIK_PATH = Path("data/polimorfologik-2.1/polimorfologik-2.1.txt")
DEFAULT_LEMMAS_PATH = Path("data/lemmas.pickle")


class Lemmas:
    def __init__(self, lemmas_dict):
        self._words_to_lemmas = lemmas_dict

    def lemmatize(self, word: str):
        return self._words_to_lemmas.get(word, [word])

    def save(self, filepath: Path = DEFAULT_LEMMAS_PATH):
        if filepath.suffix == ".json":
            self._save_json(filepath)
        elif filepath.suffix == ".pickle":
            self._save_pickle(filepath)
        else:
            raise ValueError(f"Unsupported suffix ({filepath.suffix}).")

    def _save_json(self, filepath: Path):
        with filepath.open("wt") as f:
            json.dump(self._words_to_lemmas, f)

    def _save_pickle(self, filepath: Path):
        with filepath.open("wb") as f:
            pickle.dump(self._words_to_lemmas, f)

    @classmethod
    def from_file(cls, filepath: Path = DEFAULT_LEMMAS_PATH):
        if filepath.suffix == ".json":
            return cls._from_json(filepath)
        elif filepath.suffix == ".pickle":
            return cls._from_pickle(filepath)
        else:
            raise ValueError(f"Unsupported suffix ({filepath.suffix}).")

    @classmethod
    def _from_json(cls, filepath: Path):
        with filepath.open("rt") as f:
            return cls(json.load(f))

    @classmethod
    def _from_pickle(cls, filepath: Path):
        with filepath.open("rb") as f:
            return cls(pickle.load(f))

    @classmethod
    def from_polimorfik(cls, filepath: Path = DEFAULT_POLIMORFIK_PATH):
        return cls(create_dict(filepath))


def get_lemmas_df(filepath: Path = DEFAULT_POLIMORFIK_PATH) -> pd.DataFrame:
    return pd.read_csv(filepath, sep=";", names=["lemma", "word"], usecols=[0, 1])


def create_dict(filepath: Path = DEFAULT_POLIMORFIK_PATH) -> Dict:
    df = get_lemmas_df(filepath)
    d = defaultdict(list)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        d[row["word"].lower()].append(row["lemma"].lower())
    return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, default=DEFAULT_POLIMORFIK_PATH)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_LEMMAS_PATH)
    parser.add_argument("-f", "--force", action="store_true")

    args = parser.parse_args()
    if args.force or not args.output.exists():
        lemmas = Lemmas.from_polimorfik(args.input)
        lemmas.save(args.output)
    else:
        print("Lemmas dictionary already exists.")
        print("Use -f/--force flag to force recreation.")
