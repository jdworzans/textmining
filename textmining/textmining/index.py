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

DEFAULT_INDEX_DIR = Path("data/index")


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
            return Document(**json.load(f))

    def search(self, query: str) -> Set:
        query = query.lower()
        docs_idxs = self._get_docs_idxs(query)

        docs = [self.load_doc(doc_idx) for doc_idx in docs_idxs]
        return docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("-f", "--force", action="store_true")

    args = parser.parse_args()
    if args.force or not args.output.exists():
        print("Loading lemmas")
        lemmas = Lemmas.from_file()
        print("Lemmas loaded")

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
            index.save(Path("data/temp_index"))
            print("Index saved")
    else:
        print("Index already exists.")
        print("Use -f/--force flag to force recreation.")
