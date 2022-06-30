from itertools import product
import multiprocessing as mp
import editdistance
from tqdm import tqdm

from textmining.index import DiskIndex
from textmining.lemmatization import Lemmas
from textmining.search import SearchEngine
from textmining.tokenization import tokenize
from textmining.proverbs import ProverbsHandler

lemmas = Lemmas.from_file()
index = DiskIndex(lemmas.lemmatize)
se = SearchEngine(index)
proverbs_handler = ProverbsHandler(lemmas)


def scaled_editdist(ans, cor):
    ans = ans.lower()
    cor = cor.lower()

    return editdistance.eval(ans, cor) / len(cor)


def answer(question):
    question = question.strip()
    proverbs_answer = proverbs_handler.handle_proverb_question(question)
    if proverbs_answer is not None:
        return proverbs_answer
    question_tokens = [token for token in tokenize(question.lower()) if len(token) > 1]
    while question_tokens:
        query = " ".join(question_tokens)

        for doc in se.search(query, color=False):
            result = doc.title
            res_tokens = tokenize(result.lower())

            for t1, t2 in product(res_tokens, question_tokens):
                if scaled_editdist(t1, t2) <= 0.5:
                    break
            else:
                paren_index = result.find("(")
                if paren_index != -1:
                    result = result[:paren_index]
                return result

        # if answer not found, remove first token of query
        del question_tokens[0]
    return "nie mam pojęcia, sorry"


if __name__ == "__main__":
    with open("data/poleval/odpowiedzi_solution.txt", "wt") as o:
        with open("data/poleval/pytania.txt", "rt") as i:
            for a in tqdm(map(answer, i.readlines())):
            # with mp.Pool(6) as p:
            #     for a in tqdm(p.imap(answer, i.readlines())):
                o.write(a)
                o.write("\n")
