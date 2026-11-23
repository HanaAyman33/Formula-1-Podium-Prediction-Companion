"""Text helpers shared by the classifiers and the entity extractor."""
import re
import unicodedata

_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def fold(text: str) -> str:
    """Lowercase and strip accents: 'Pérez' -> 'perez', 'São Paulo' -> 'sao paulo'."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


def normalize(text: str) -> str:
    """Folded text with possessives removed and punctuation turned into spaces."""
    t = fold(text)
    t = t.replace("’", "'").replace("&amp;", "&")
    t = re.sub(r"'s\b", "", t)            # Hamilton's -> hamilton
    t = re.sub(r"s'\b", "s", t)           # drivers' -> drivers
    t = re.sub(r"[^a-z0-9\-' ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def tokens(text: str) -> list[str]:
    return _WORD.findall(normalize(text))


def ngrams(toks: list[str], n_max: int = 5):
    """(start, end, phrase) for every n-gram, longest first, so longer names win."""
    for n in range(min(n_max, len(toks)), 0, -1):
        for i in range(len(toks) - n + 1):
            yield i, i + n, " ".join(toks[i:i + n])
