import pytest

from textmining.tokenization import (
    lstrip_punctuation,
    rstrip_punctuation,
    strip_notalnum,
    strip_punctuation,
)


@pytest.mark.parametrize(
    "word, target",
    [
        ("abc", "abc"),
        ("abc!", "abc!"),
        ("a!bc!", "a!bc!"),
        ("?a!bc!", "a!bc!"),
        ("....?a!bc!", "a!bc!"),
        ("....?a!bc", "a!bc"),
    ],
)
def test_lstrip_punctuation(word, target):
    assert lstrip_punctuation(word) == target


@pytest.mark.parametrize(
    "word, target",
    [
        ("abc", "abc"),
        ("abc!", "abc"),
        ("a!bc!", "a!bc"),
        ("?a!bc!", "?a!bc"),
        ("....?a!bc!", "....?a!bc"),
        ("....?a!bc", "....?a!bc"),
    ],
)
def test_rstrip_punctuation(word, target):
    assert rstrip_punctuation(word) == target


@pytest.mark.parametrize(
    "word, target",
    [
        ("abc", "abc"),
        ("abc!", "abc"),
        ("a!bc!", "a!bc"),
        ("?a!bc!", "a!bc"),
        ("....?a!bc!", "a!bc"),
        ("....?a!bc", "a!bc"),
    ],
)
def test_strip_punctuation(word, target):
    assert strip_punctuation(word) == target


@pytest.mark.parametrize(
    "word, target",
    [
        ("``awk''", "awk"),
        ("abc!", "abc"),
        ("a!bc!", "a!bc"),
        ("?a!bc!", "a!bc"),
        ("....?a!bc!", "a!bc"),
        ("....?a!bc", "a!bc"),
    ],
)
def test_strip_notalnum(word, target):
    assert strip_notalnum(word) == target
