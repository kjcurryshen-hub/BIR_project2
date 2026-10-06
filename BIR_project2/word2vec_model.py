"""Small dependency-free CBOW and Skip-gram models for coursework demos."""

from bisect import bisect_left
from collections import Counter
import math
import random

from engine import STOPWORDS, tokens


MODEL_NAMES = {'skipgram': 'Skip-gram', 'cbow': 'CBOW'}


def _sigmoid(value):
    value = max(-8.0, min(8.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def _validate(model, dimension, window, epochs):
    if model not in MODEL_NAMES:
        raise ValueError('Model must be CBOW or Skip-gram')
    if not 8 <= dimension <= 64:
        raise ValueError('Vector dimension must be between 8 and 64')
    if not 1 <= window <= 5:
        raise ValueError('Window size must be between 1 and 5')
    if not 1 <= epochs <= 5:
        raise ValueError('Epochs must be between 1 and 5')


def train_word2vec(engine, model='skipgram', dimension=24, window=2, epochs=2,
                   max_vocabulary=800, max_examples=60000, negative_samples=3):
    """Train CBOW or Skip-gram with negative sampling on current abstracts."""
    _validate(model, dimension, window, epochs)
    cache = getattr(engine, '_word2vec_models', None)
    if cache is None:
        cache = {}
        engine._word2vec_models = cache
    key = (model, dimension, window, epochs, max_vocabulary,
           max_examples, negative_samples)
    if key in cache:
        return cache[key]

    documents = [[word for word in tokens(document.abstract)
                  if word not in STOPWORDS and word.isalpha()]
                 for document in engine.docs.values()]
    frequencies = Counter(word for document in documents for word in document)
    vocabulary = [word for word, count in frequencies.most_common(max_vocabulary)
                  if count >= 2]
    if len(vocabulary) < 2:
        result = {'model': model, 'model_name': MODEL_NAMES[model],
                  'vectors': {}, 'frequencies': frequencies, 'examples': 0,
                  'pairs': 0, 'epochs': epochs, 'dimension': dimension,
                  'window': window, 'vocabulary_size': len(vocabulary)}
        cache[key] = result
        return result

    index = {word: position for position, word in enumerate(vocabulary)}
    sequences = [[index[word] for word in document if word in index]
                 for document in documents]
    examples = []
    for sequence in sequences:
        for position, center in enumerate(sequence):
            start = max(0, position-window)
            end = min(len(sequence), position+window+1)
            context = tuple(sequence[context_position]
                            for context_position in range(start, end)
                            if context_position != position)
            if not context:
                continue
            if model == 'cbow':
                # Several surrounding words predict the middle word.
                examples.append((context, center))
            else:
                # The middle word predicts each surrounding word.
                examples.extend(((center,), context_word) for context_word in context)
            if len(examples) >= max_examples:
                del examples[max_examples:]
                break
        if len(examples) >= max_examples:
            break

    seed = 42 + dimension * 3 + window * 11 + epochs * 17 + (1 if model == 'cbow' else 0)
    rng = random.Random(seed)
    scale = 0.5 / dimension
    input_vectors = [[rng.uniform(-scale, scale) for _ in range(dimension)]
                     for _ in vocabulary]
    output_vectors = [[0.0] * dimension for _ in vocabulary]
    cumulative, total = [], 0.0
    for word in vocabulary:
        total += frequencies[word] ** 0.75
        cumulative.append(total)

    def update(source_indices, target, label, rate):
        count = len(source_indices)
        hidden = [sum(input_vectors[source][component]
                      for source in source_indices) / count
                  for component in range(dimension)]
        destination = output_vectors[target]
        old_destination = destination[:]
        score = sum(a*b for a, b in zip(hidden, destination))
        gradient = rate * (label - _sigmoid(score))
        for component in range(dimension):
            destination[component] += gradient * hidden[component]
            source_gradient = gradient * old_destination[component] / count
            for source in source_indices:
                input_vectors[source][component] += source_gradient

    for epoch in range(epochs):
        rng.shuffle(examples)
        rate = 0.035 * (1.0 - 0.35 * epoch / max(1, epochs-1))
        for source_indices, target in examples:
            update(source_indices, target, 1, rate)
            for _ in range(negative_samples):
                negative = bisect_left(cumulative, rng.random() * total)
                if negative == target:
                    negative = (negative + 1) % len(vocabulary)
                update(source_indices, negative, 0, rate)

    normalized = {}
    for word, vector in zip(vocabulary, input_vectors):
        length = math.sqrt(sum(value*value for value in vector)) or 1.0
        normalized[word] = [value/length for value in vector]
    result = {'model': model, 'model_name': MODEL_NAMES[model],
              'vectors': normalized, 'frequencies': frequencies,
              'examples': len(examples), 'pairs': len(examples),
              'epochs': epochs, 'dimension': dimension, 'window': window,
              'vocabulary_size': len(vocabulary)}
    cache[key] = result
    return result


def train_skipgram(engine, dimension=24, window=2, epochs=2,
                   max_vocabulary=800, max_pairs=60000, negative_samples=3):
    """Backward-compatible wrapper for earlier callers."""
    return train_word2vec(engine, 'skipgram', dimension, window, epochs,
                          max_vocabulary, max_pairs, negative_samples)


def train_cbow(engine, dimension=24, window=2, epochs=2,
               max_vocabulary=800, max_examples=60000, negative_samples=3):
    return train_word2vec(engine, 'cbow', dimension, window, epochs,
                          max_vocabulary, max_examples, negative_samples)


def nearest_words(model, word, limit=10):
    word = word.casefold().strip()
    vector = model['vectors'].get(word)
    if vector is None:
        return []
    similarities = []
    for candidate, other in model['vectors'].items():
        if candidate == word:
            continue
        similarities.append((sum(a*b for a, b in zip(vector, other)), candidate))
    return sorted(similarities, key=lambda pair: (-pair[0], pair[1]))[:limit]


def pca_projection(model, query_word='', limit=90):
    """Project a useful subset of learned vectors onto two PCA components.

    The implementation intentionally uses only Python's standard library so the
    coursework remains runnable without NumPy.  When a query is available, its
    closest words are selected first and frequent vocabulary terms fill the
    remaining places.
    """
    vectors = model.get('vectors', {})
    if len(vectors) < 3:
        return {'points': [], 'explained_variance': (0.0, 0.0)}

    query = query_word.casefold().strip()
    selected = []
    if query in vectors:
        selected.append(query)
        selected.extend(word for _, word in nearest_words(model, query, min(50, limit-1)))

    frequencies = model.get('frequencies', {})
    candidates = sorted(vectors, key=lambda word: (-frequencies.get(word, 0), word))
    for word in candidates:
        if word not in selected:
            selected.append(word)
        if len(selected) >= limit:
            break
    selected = selected[:limit]

    rows = [vectors[word] for word in selected]
    dimensions = len(rows[0])
    means = [sum(row[column] for row in rows) / len(rows)
             for column in range(dimensions)]
    centered = [[value-means[column] for column, value in enumerate(row)]
                for row in rows]
    denominator = max(1, len(centered)-1)
    covariance = [[sum(row[a]*row[b] for row in centered) / denominator
                   for b in range(dimensions)] for a in range(dimensions)]

    def dot(left, right):
        return sum(a*b for a, b in zip(left, right))

    def matrix_vector(vector):
        return [dot(row, vector) for row in covariance]

    def principal_component(previous=None):
        vector = [1.0/(index+1) for index in range(dimensions)]
        if previous:
            overlap = dot(vector, previous)
            vector = [value-overlap*axis for value, axis in zip(vector, previous)]
        length = math.sqrt(dot(vector, vector)) or 1.0
        vector = [value/length for value in vector]
        for _ in range(100):
            updated = matrix_vector(vector)
            if previous:
                overlap = dot(updated, previous)
                updated = [value-overlap*axis for value, axis in zip(updated, previous)]
            length = math.sqrt(dot(updated, updated))
            if length < 1e-12:
                return [0.0]*dimensions, 0.0
            updated = [value/length for value in updated]
            if dot(updated, vector) < 0:
                updated = [-value for value in updated]
            difference = sum((a-b)**2 for a, b in zip(updated, vector))
            vector = updated
            if difference < 1e-16:
                break
        eigenvalue = max(0.0, dot(vector, matrix_vector(vector)))
        return vector, eigenvalue

    first_axis, first_value = principal_component()
    second_axis, second_value = principal_component(first_axis)
    total_variance = max(0.0, sum(covariance[index][index]
                                  for index in range(dimensions)))
    query_vector = vectors.get(query)
    points = []
    for word, row in zip(selected, centered):
        similarity = (dot(query_vector, vectors[word])
                      if query_vector is not None else None)
        points.append({
            'word': word,
            'x': dot(row, first_axis),
            'y': dot(row, second_axis),
            'similarity': similarity,
            'frequency': frequencies.get(word, 0),
            'is_query': word == query,
        })
    explained = ((first_value/total_variance if total_variance else 0.0),
                 (second_value/total_variance if total_variance else 0.0))
    return {'points': points, 'explained_variance': explained}
