import re
from itertools import chain
from pathlib import Path

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from textmining.lemmatization import Lemmas
from textmining.tokenization import tokenize


PROVERBS_PATH = Path("data/proverbs.txt")
BERT_TOKENIZER = AutoTokenizer.from_pretrained("henryk/bert-base-multilingual-cased-finetuned-polish-squad2")
BERT = AutoModelForQuestionAnswering.from_pretrained("henryk/bert-base-multilingual-cased-finetuned-polish-squad2")

def normalize(text: str):
    return " ".join(tokenize(text))

class ProverbsHandler:
    def __init__(self, lemmas):
        with PROVERBS_PATH.open("rt") as f:
            self.PROVERBS = f.read().strip().lower()
        self.lemmas = lemmas
        self.TFIDF_VECTORIZER = TfidfVectorizer(tokenizer=self._tfidf_tokenize)
        self.PROVERBS_TFIDF = self.TFIDF_VECTORIZER.fit_transform(self.PROVERBS.splitlines())

    def _tfidf_tokenize(self, text: str):
        return list(chain.from_iterable([self.lemmas.lemmatize(token) for token in tokenize(text.lower())]))

    def get_most_similar_proverbs(self, question: str, k: int = 10):
        question_tfidf = self.TFIDF_VECTORIZER.transform([question])
        terms_idxs = question_tfidf.nonzero()[1]
        proverbs_tfidf_sum = self.PROVERBS_TFIDF[:, terms_idxs].sum(1).A1
        best_proverbs_idxs = proverbs_tfidf_sum.argsort()[-1:-k:-1]
        return [self.PROVERBS.splitlines()[idx] for idx in best_proverbs_idxs]


    def handle_proverb_question(self, question):
        question_lemmas = set(chain.from_iterable(map(self.lemmas.lemmatize, tokenize(question.lower()))))

        if question_lemmas.intersection(self.lemmas.lemmatize("dokończ")):
            match = re.search("„(?P<name>.*)...”", question.lower())
            if match:
                beginning = match["name"]
                beginning_match = re.search(f"^{beginning}(?P<name>.*)$", self.PROVERBS, flags=re.MULTILINE)
                if beginning_match:
                    return normalize(beginning_match["name"])

        context = " [SEP] ".join(self.get_most_similar_proverbs(question.lower()))
        inputs = BERT_TOKENIZER(question.lower(), context, return_tensors="pt")
        with torch.no_grad():
            outputs = BERT(**inputs)

        answer_start_index = outputs.start_logits.argmax()
        if answer_start_index != 0:
            offset = outputs.end_logits[0, answer_start_index:].argmax()

            predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_start_index + offset + 1]
            return(BERT_TOKENIZER.decode(predict_answer_tokens))
        return None
