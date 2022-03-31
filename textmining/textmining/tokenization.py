import unicodedata

def lstrip_punctuation(word: str):
    """
    Return a copy of the string with
    the leading punctuation removed.
    """
    for idx, c in enumerate(word):
        if not unicodedata.category(c).startswith("P"):
            return word[idx:]
    return ""

def rstrip_punctuation(word: str):
    """
    Return a copy of the string with
    the trailing punctuation removed.
    """
    return lstrip_punctuation(word[::-1])[::-1]

def strip_punctuation(word: str):
    """
    Return a copy of the string with
    the leading and trailing punctuation removed.
    """
    return rstrip_punctuation(lstrip_punctuation(word))

def tokenize(text: str):
    return text.split()
