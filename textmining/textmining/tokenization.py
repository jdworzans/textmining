import unicodedata


def lstrip_punctuation(word: str) -> str:
    """
    Return a copy of the string with
    the leading punctuation removed.
    """
    for idx, c in enumerate(word):
        if not unicodedata.category(c).startswith("P"):
            return word[idx:]
    return ""


def rstrip_punctuation(word: str) -> str:
    """
    Return a copy of the string with
    the trailing punctuation removed.
    """
    return lstrip_punctuation(word[::-1])[::-1]


def strip_punctuation(word: str) -> str:
    """
    Return a copy of the string with
    the leading and trailing punctuation removed.
    """
    return rstrip_punctuation(lstrip_punctuation(word))


def lstrip_notalnum(word: str):
    """
    Return a copy of the string with
    the leading not alphanumeric characters removed.
    """
    for idx, c in enumerate(word):
        if c.isalnum():
            return word[idx:]
    return ""


def rstrip_notalnum(word: str) -> str:
    """
    Return a copy of the string with
    the trailing not alphanumeric characters removed.
    """
    return lstrip_notalnum(word[::-1])[::-1]


def strip_notalnum(word: str) -> str:
    """
    Return a copy of the string with the leading
    and trailing not alphanumeric characters removed.
    """
    return rstrip_notalnum(lstrip_notalnum(word))


def tokenize(text: str):
    """
    Return tokenized text.

    Tokenization is applied by splitting on whitespaces
    and removing every leading and trailing not alphanumeric character.
    """
    return [
        tokenized_word
        for tokenized_word in map(strip_notalnum, text.split())
        if tokenized_word
    ]
