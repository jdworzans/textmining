import argparse
import hashlib
import json
import pickle
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Set

from tqdm import tqdm

from textmining.lemmatization import Lemmas
from textmining.tokenization import tokenize
import bisect

DEFAULT_INDEX_DIR = Path("data/index")
DEFAULT_POSITION_INDEX_DIR = Path("data/position_index")


def get_hash(word: str):
    return hashlib.md5(word.encode()).hexdigest()


@dataclass
class Document:
    title: str
    content: str


class Index:
    def __init__(self, lemmatize):
        self.lemmatize = lemmatize

        self.documents = []
        self.inverse_mapping = defaultdict(list)

    def add(self, document: Document):
        document_idx = len(self.documents)
        self.documents.append(document)

        lemmas = set()
        for element in [document.title, document.content]:
            for token in tokenize(element.lower()):
                lemmas.update(self.lemmatize(token))

        for lemma in lemmas:
            self.inverse_mapping[lemma].append(document_idx)

    def _get_docs(self, token):
        lemmas = self.lemmatize(token)
        docs = set()
        for term in lemmas:
            docs.update(self.inverse_mapping[term])
        return docs

    def search(self, query: str) -> Set:
        docs = set()

        tokens = tokenize(query.lower())
        if tokens:
            docs = self._get_docs(tokens[0])
        else:
            return set()

        for token in tokens[1:]:
            if docs:
                docs.intersection_update(self._get_docs(token))
            else:
                return docs
        return docs

    def save(self, dir: Path = DEFAULT_INDEX_DIR):
        dir.mkdir(exist_ok=True)

        index_dir = dir / "index"
        index_dir.mkdir(exist_ok=True)
        print("Saving inverse mapping to", index_dir)
        for lemma, docs_idxs in tqdm(
            self.inverse_mapping.items(), total=len(self.inverse_mapping)
        ):
            with (index_dir / str(get_hash(lemma))).with_suffix(".pickle").open(
                "wb"
            ) as f:
                pickle.dump(docs_idxs, f)

        docs_dir = dir / "docs"
        docs_dir.mkdir(exist_ok=True)
        print("Saving docs to", docs_dir)
        for doc_idx, document in tqdm(
            enumerate(self.documents), total=len(self.documents)
        ):
            with (docs_dir / str(doc_idx)).with_suffix(".json").open("wt") as f:
                json.dump(asdict(document), f)

class PositionIndex:
    def __init__(self, lemmatize):
        self.lemmatize = lemmatize
        self.word_idx = 0
        self.beginnings = []
        self.documents = []
        self.inverse_mapping = defaultdict(list)

    def add(self, document: Document):
        self.beginnings.append(self.word_idx)
        self.documents.append(document)

        for element in [document.title, document.content]:
            for token in tokenize(element.lower()):
                for lemma in self.lemmatize(token):
                    self.inverse_mapping[lemma].append(self.word_idx)
                self.word_idx += 1
        self.word_idx += 1

    def _get_term_positions(self, term):
        return self.inverse_mapping[term]

    def _get_positions(self, token):
        lemmas = self.lemmatize(token)
        positions = set()
        for term in lemmas:
            positions.update(self._get_term_positions(term))
        return positions


    def _get_docs_idxs(self, query: str) -> Set[int]:
        tokens = tokenize(query.lower())
        if tokens:
            positions = self._get_positions(tokens[0])
        else:
            return set()

        for idx, token in enumerate(tokens[1:], 1):
            if positions:
                new_positions = self._get_positions(token)
                positions.intersection_update({pos - idx for pos in new_positions})
            else:
                return set()

        docs_idxs = set()
        for pos in positions:
            doc_idx = bisect.bisect(self.beginnings, pos) - 1
            docs_idxs.add(doc_idx)
        return docs_idxs

    def load_doc(self, doc_idx: int) -> Document:
        return self.documents[doc_idx]

    def search(self, query: str) -> Set:
        query = query.lower()
        docs_idxs = self._get_docs_idxs(query)

        docs = [self.load_doc(doc_idx) for doc_idx in sorted(docs_idxs)]
        return docs

    def save(self, dir: Path = DEFAULT_POSITION_INDEX_DIR):
        dir.mkdir(exist_ok=True)

        index_dir = dir / "index"
        index_dir.mkdir(exist_ok=True)
        print("Saving inverse mapping to", index_dir)
        for lemma, words_idxs in tqdm(
            self.inverse_mapping.items(), total=len(self.inverse_mapping)
        ):
            with (index_dir / str(get_hash(lemma))).with_suffix(".pickle").open(
                "wb"
            ) as f:
                pickle.dump(words_idxs, f)

        docs_dir = dir / "docs"
        docs_dir.mkdir(exist_ok=True)
        print("Saving docs to", docs_dir)
        for doc_idx, document in tqdm(
            enumerate(self.documents), total=len(self.documents)
        ):
            with (docs_dir / str(doc_idx)).with_suffix(".json").open("wt") as f:
                json.dump(asdict(document), f)
        beginings_path = dir / "beginings.pickle"
        print("Saving beginnings to", beginings_path)
        with beginings_path.open("wb") as f:
            pickle.dump(self.beginnings, f)

class DiskPositionIndex(PositionIndex):
    def __init__(self, lemmatize, dir: Path = DEFAULT_POSITION_INDEX_DIR):
        self.dir = dir
        self.lemmatize = lemmatize
        beginings_path = self.dir / "beginings.pickle"
        with beginings_path.open("rb") as f:
            self.beginnings = pickle.load(f)

    def _get_term_positions(self, term):
        index_dir = self.dir / "index"
        term_filepath = (index_dir / str(get_hash(term))).with_suffix(".pickle")
        if term_filepath.exists():
            with term_filepath.open("rb") as f:
                return pickle.load(f)
        else:
            return set()

    def load_doc(self, doc_idx: int) -> Document:
        doc_dir = self.dir / "docs"
        doc_filepath = (doc_dir / str(doc_idx)).with_suffix(".json")
        with doc_filepath.open("rt") as f:
            doc = Document(**json.load(f))
            doc.id = doc_idx
            return doc

class DiskIndex:
    def __init__(self, lemmatize, dir: Path = DEFAULT_INDEX_DIR):
        self.lemmatize = lemmatize
        self.dir = dir

    def _get_term_docs(self, term):
        index_dir = self.dir / "index"
        term_filepath = (index_dir / str(get_hash(term))).with_suffix(".pickle")
        if term_filepath.exists():
            with term_filepath.open("rb") as f:
                return pickle.load(f)
        else:
            return set()

    def _get_token_docs_idxs(self, token):
        lemmas = self.lemmatize(token)
        docs = set()
        for term in lemmas:
            term_docs = self._get_term_docs(term)
            docs.update(term_docs)
        return docs

    def _get_docs_idxs(self, query: str) -> Set[int]:
        docs_idxs = set()

        tokens = tokenize(query)
        if tokens:
            docs_idxs = self._get_token_docs_idxs(tokens[0])
        else:
            return set()

        for token in tokens[1:]:
            if docs_idxs:
                docs_idxs.intersection_update(self._get_token_docs_idxs(token))
            else:
                return docs_idxs
        return docs_idxs

    def load_doc(self, doc_idx: int) -> Document:
        doc_dir = self.dir / "docs"
        doc_filepath = (doc_dir / str(doc_idx)).with_suffix(".json")
        with doc_filepath.open("rt") as f:
            doc = Document(**json.load(f))
            doc.id = doc_idx
            return doc

    def search(self, query: str) -> Set:
        query = query.lower()
        docs_idxs = self._get_docs_idxs(query)

        docs = [self.load_doc(doc_idx) for doc_idx in sorted(docs_idxs)]
        return docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("-p", "--position", action="store_true")


    args = parser.parse_args()

    if args.output is None:
        if args.position:
            output = DEFAULT_POSITION_INDEX_DIR
        else:
            output = DEFAULT_INDEX_DIR
    else:
        output = args.output

    if args.force or not output.exists():
        print("Loading lemmas")
        lemmas = Lemmas.from_file()
        print("Lemmas loaded")

        if args.position:
            index = PositionIndex(lemmas.lemmatize)
        else:
            index = Index(lemmas.lemmatize)

        with open("data/fp_wiki.txt", "rt") as f:
            lines = iter(tqdm(f.readlines()))
            try:
                while (first_line := next(lines)).startswith("TITLE: "):
                    title = next(lines).strip()
                    content_lines = []
                    while l := next(lines).strip():
                        content_lines.append(l)
                    content = " ".join(content_lines)
                    document = Document(title, content)
                    index.add(document)
            except StopIteration:
                print("Index created")

            print("Saving index")
            index.save(output)
            print("Index saved")
    else:
        print("Index already exists.")
        print("Use -f/--force flag to force recreation.")
