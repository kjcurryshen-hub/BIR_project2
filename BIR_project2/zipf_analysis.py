"""Zipf, CF, DF and TF-IDF analysis for the abstract collection.

The four conditions intentionally form a cumulative preprocessing pipeline:
A keeps punctuation attached to whitespace-delimited terms; B applies the
project tokenizer; C removes stopwords; D additionally applies Porter stemming.
"""

from collections import Counter
import math
import re

from engine import STOPWORDS, tokens
from porter import stem


CONDITIONS = {
    'A': 'A · Basic (whitespace + lowercase)',
    'B': 'B · Remove punctuation',
    'C': 'C · Remove stopwords',
    'D': 'D · Porter stemming',
}


def _basic_tokens(text):
    return [part for part in text.casefold().split()
            if any(character.isalnum() for character in part)]


def preprocess(text, condition):
    if condition == 'A':
        return _basic_tokens(text)
    result = tokens(text)
    if condition in {'C', 'D'}:
        result = [term for term in result if term not in STOPWORDS]
    if condition == 'D':
        result = [stem(term) for term in result]
    return result


def regression(frequencies):
    """OLS fit of log10(CF) = intercept + slope * log10(rank)."""
    if len(frequencies) < 2:
        return {'slope': 0.0, 'intercept': 0.0, 'exponent': 0.0,
                'r_squared': 0.0, 'rmse': 0.0}
    xs = [math.log10(rank) for rank in range(1, len(frequencies) + 1)]
    ys = [math.log10(value) for value in frequencies]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    slope = (sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) /
             denominator if denominator else 0.0)
    intercept = mean_y - slope * mean_x
    predictions = [intercept + slope * x for x in xs]
    residual_sum = sum((y - prediction) ** 2
                       for y, prediction in zip(ys, predictions))
    total_sum = sum((y - mean_y) ** 2 for y in ys)
    r_squared = 1 - residual_sum / total_sum if total_sum else 0.0
    return {'slope': slope, 'intercept': intercept, 'exponent': -slope,
            'r_squared': r_squared,
            'rmse': math.sqrt(residual_sum / len(xs))}


def analyze_documents(documents, condition):
    cf = Counter()
    df = Counter()
    document_tf = {}
    for document in documents:
        doc_terms = preprocess(document.abstract, condition)
        frequencies = Counter(doc_terms)
        document_tf[document.id] = frequencies
        cf.update(frequencies)
        df.update(frequencies.keys())
    ranked = sorted(cf.items(), key=lambda pair: (-pair[1], pair[0]))
    values = [frequency for _, frequency in ranked]
    document_count = len(documents)
    total_tokens = sum(values)
    result = {
        'condition': condition,
        'documents': document_count,
        'total_tokens': total_tokens,
        'unique_terms': len(ranked),
        'average_tokens': total_tokens / document_count if document_count else 0.0,
        'cf': cf,
        'df': df,
        'ranked': ranked,
        'document_tf': document_tf,
        'regression': regression(values),
    }
    # Compare high, middle and low thirds using the same log-log regression.
    regions = {}
    if values:
        boundaries = (0, max(1, len(values) // 3), max(2, 2 * len(values) // 3), len(values))
        labels = ('高頻區', '中頻區', '低頻區')
        for label, start, end in zip(labels, boundaries, boundaries[1:]):
            sliced = values[start:end]
            # Preserve the original ranks when fitting a region.
            ranks = list(range(start + 1, end + 1))
            regions[label] = regression_at_ranks(ranks, sliced)
    result['regions'] = regions
    return result


def regression_at_ranks(ranks, frequencies):
    if len(frequencies) < 2:
        return {'slope': 0.0, 'intercept': 0.0, 'exponent': 0.0,
                'r_squared': 0.0, 'rmse': 0.0}
    xs = [math.log10(rank) for rank in ranks]
    ys = [math.log10(value) for value in frequencies]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator if denominator else 0.0
    intercept = mean_y - slope * mean_x
    predictions = [intercept + slope * x for x in xs]
    residual_sum = sum((y - prediction) ** 2 for y, prediction in zip(ys, predictions))
    total_sum = sum((y - mean_y) ** 2 for y in ys)
    return {'slope': slope, 'intercept': intercept, 'exponent': -slope,
            'r_squared': 1 - residual_sum / total_sum if total_sum else 0.0,
            'rmse': math.sqrt(residual_sum / len(xs))}


def analyze_all(engine):
    cached = getattr(engine, '_zipf_analysis_cache', None)
    if cached is None:
        documents = list(engine.docs.values())
        cached = {key: analyze_documents(documents, key) for key in CONDITIONS}
        engine._zipf_analysis_cache = cached
    return cached


def query_term(value, condition):
    processed = preprocess(value.strip(), condition)
    if len(processed) != 1:
        if not processed:
            return None, '此輸入在所選前處理條件下沒有可分析的詞。'
        return None, '請只輸入一個英文單詞。'
    return processed[0], ''


def idf(document_count, document_frequency):
    return math.log(document_count / document_frequency) if document_count and document_frequency else 0.0
