import argparse
import re
from pathlib import Path
from typing import List

from colorama import Fore, Style

from textmining.index import DEFAULT_INDEX_DIR, DiskIndex, DiskPositionIndex, Document, Index
from textmining.lemmatization import Lemmas
from textmining.tokenization import tokenize


def get_lemmas(text: str, lemmatize):
    return {lemma for token in tokenize(text) for lemma in lemmatize(token)}


class SearchEngine:
    def __init__(self, index=None):
        if index is None:
            self.index = Index()
        else:
            self.index = index

    def process(self, docs: List[Document], query: str, color=False):
        qtokens = tokenize(query.lower())
        qlemmas = {lemma for token in qtokens for lemma in self.index.lemmatize(token)}
        for doc in docs:
            doc.title_matching = 0
            doc.exact_matching = 0
            for token in tokenize(doc.title):
                lemmas = set(self.index.lemmatize(token.lower()))
                if qlemmas.intersection(lemmas):
                    doc.title_matching += 1
                    if color:
                        doc.title = re.sub(
                            rf"(\b){token}(\b)",
                            rf"\1{Fore.RED + token + Style.RESET_ALL}\2",
                            doc.title,
                        )
                if token.lower() in qtokens:
                    doc.exact_matching += 1

            for token in tokenize(doc.content):
                lemmas = set(self.index.lemmatize(token.lower()))
                if qlemmas.intersection(lemmas):
                    if color:
                        doc.content = re.sub(
                            rf"(\b){token}(\b)",
                            rf"\1{Fore.RED + token + Style.RESET_ALL}\2",
                            doc.content,
                        )

                if token.lower() in qtokens:
                    doc.exact_matching += 1

        return sorted(
            docs, reverse=True, key=lambda d: (d.title_matching, d.exact_matching)
        )

    def search(self, query: str, color=True):
        docs = self.index.search(query)
        return self.process(docs, query, color)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Wyszukiwarka")
    parser.add_argument("-d", "--dir", default=None)
    parser.add_argument("-p", "--position", action="store_true")
    args = parser.parse_args()

    print("Ładowanie lematów")
    lemmas = Lemmas.from_file()
    print("Lematy gotowe")
    if args.position:
        if args.dir is not None:
            index = DiskPositionIndex(lemmas.lemmatize, dir)
        else:
            index = DiskPositionIndex(lemmas.lemmatize)
    else:
        if args.dir is not None:
            index = DiskIndex(lemmas.lemmatize, dir)
        else:
            index = DiskIndex(lemmas.lemmatize)
    se = SearchEngine(index)

    prompt = "Naciśnij ENTER, aby zobaczyć kolejny dokument..."
    try:
        while True:
            query = input("Wprowadź zapytanie: ")
            docs = se.search(query)
            if docs:
                print(f"[{docs[0].id}]  {docs[0].title}", docs[0].content, sep="\n", end="\n\n")

            for doc in docs[1:]:
                if not (response := input(prompt)):
                    print("\033[A", len(prompt) * " ", "\033[A")
                    print(f"[{doc.id}]  {doc.title}", doc.content, sep="\n", end="\n\n")
                else:
                    break
    except KeyboardInterrupt:
        pass
