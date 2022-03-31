import pytest

from textmining.tokenization import (
    lstrip_punctuation,
    strip_punctuation,
    rstrip_punctuation,
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
