"""Dynamic-programming Edit Distance spelling suggestions."""

from collections import Counter

from engine import STOPWORDS, tokens


def edit_distance(left, right, maximum=None):
    """Levenshtein distance using two DP rows and an optional early cutoff."""
    left, right = left.casefold(), right.casefold()
    if len(left) > len(right):
        left, right = right, left
    if maximum is not None and len(right) - len(left) > maximum:
        return maximum + 1
    previous = list(range(len(left) + 1))
    for row_index, right_character in enumerate(right, 1):
        current = [row_index]
        for column_index, left_character in enumerate(left, 1):
            current.append(min(
                current[-1] + 1,
                previous[column_index] + 1,
                previous[column_index - 1] + (left_character != right_character),
            ))
        if maximum is not None and min(current) > maximum:
            return maximum + 1
        previous = current
    return previous[-1]


def vocabulary(engine):
    cached = getattr(engine, '_spelling_vocabulary', None)
    if cached is None:
        frequencies = Counter()
        for document in engine.docs.values():
            frequencies.update(term for term in tokens(document.text)
                               if term not in STOPWORDS and term.isalpha())
        cached = frequencies
        engine._spelling_vocabulary = cached
    return cached


def suggest_word(word, frequencies):
    word = word.casefold()
    if word in frequencies or not word.isalpha() or len(word) < 3:
        return word
    maximum = 1 if len(word) <= 4 else 2
    candidates = []
    for candidate, frequency in frequencies.items():
        if abs(len(candidate) - len(word)) > maximum:
            continue
        distance = edit_distance(word, candidate, maximum)
        if distance <= maximum:
            candidates.append((distance, -frequency, candidate))
    return min(candidates)[2] if candidates else word


def correct_query(query, engine):
    frequencies = vocabulary(engine)
    original = tokens(query)
    corrected = [suggest_word(word, frequencies) for word in original]
    if not original or corrected == original:
        return '', []
    return ' '.join(corrected), [
        (before, after) for before, after in zip(original, corrected) if before != after
    ]
