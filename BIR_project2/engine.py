"""PMC JATS parsing, rule-based sentences, inverted index and BM25. Standard library only."""
from collections import Counter, defaultdict
from porter import stem
from dataclasses import dataclass, field
from pathlib import Path
import math
import json
import os
import re
import time
import xml.etree.ElementTree as ET

# A compact, explicit English list; counts always use the original tokens.
STOPWORDS = set('a an the and or but if then of in on at to for from by with as is are was were be been being this that these those it its we our you your they their he she his her not no do does did have has had can could may might will would should than which who whom what when where how into about also such using used use'.split())
# Decimal numbers stay as one token (0.65), while sentence punctuation is
# excluded. An en dash still separates a range: 0.44–0.97 -> two tokens.
TOKEN = re.compile(r"\d+(?:\.\d+)+|[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
ABBREV = re.compile(r'\b(?:e\.g\.|i\.e\.|et\s+al\.|Dr\.|Mr\.|Mrs\.|Ms\.|Prof\.|Fig\.|Figs\.|Eq\.|Eqs\.|vs\.|approx\.|No\.|vol\.)', re.I)


def tokens(text):
    return [m.group().casefold() for m in TOKEN.finditer(text)]


def terms(text):
    return [stem(t) for t in tokens(text) if t not in STOPWORDS]


def split_sentences(text):
    """Preserve original characters; protect abbreviation/decimal/initial periods.
    Paragraph boundaries are boundaries; unpunctuated residual text counts as one unit.
    Heuristic English segmenter, not a gold-standard biomedical NLP model.
    """
    result = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        protected = set()
        for m in ABBREV.finditer(line):
            protected.update(i for i in range(m.start(), m.end()) if line[i] == '.')
        for m in re.finditer(r'(?<=\d)\.(?=\d)', line):
            protected.add(m.start())
        # Initials before a capitalized name, and dotted acronyms followed by lowercase.
        for m in re.finditer(r'\b[A-Z]\.(?=\s+[A-Z][a-z])|\b(?:[A-Za-z]\.){2,}(?=\s+[a-z])', line):
            protected.update(i for i in range(m.start(), m.end()) if line[i] == '.')
        start = 0
        for m in re.finditer(r'[.!?。！？]+["\'”’\)\]]*(?=\s|$)', line):
            if all(line[i] == '.' and i in protected for i in range(m.start(), m.start()+len(m.group().rstrip('\"\'”’)]')))):
                continue
            segment = line[start:m.end()].strip()
            if segment:
                result.append(segment)
            start = m.end()
        tail = line[start:].strip()
        if tail:
            result.append(tail)
    return result


BLOCKS = {'title', 'p', 'sec', 'abstract', 'body', 'list', 'list-item', 'table', 'tr', 'td', 'th', 'caption', 'disp-quote', 'boxed-text', 'def-item', 'statement'}
SKIP = {'ref-list', 'floats-group', 'supplementary-material', 'sub-article', 'response'}


def extract_text(node):
    """No artificial spaces inside inline markup (e.g. inter<italic>leukin</italic>).
    Separate block elements; preserve child tails even when a child is excluded.
    """
    if node is None:
        return ''
    def walk(n):
        if n.tag in SKIP:
            return ''
        out = n.text or ''
        for child in n:
            block = child.tag in BLOCKS
            # PMC 的引用標籤常直接夾在單字後面，例如：
            # testing<xref>10</xref>。若不補空格會變成 testing10，
            # 造成斷詞與 Porter stemming 漏算。只替 xref 建立文字邊界，
            # 其他行內標籤（italic、bold 等）仍保持原本的連續文字。
            separator = '\n' if block else (' ' if child.tag == 'xref' else '')
            out += separator + walk(child) + separator
            out += child.tail or ''
        return out
    return '\n'.join(re.sub(r'\s+', ' ', s).strip() for s in walk(node).splitlines() if s.strip())


@dataclass
class Document:
    id: str
    title: str
    abstract: str
    body: str
    filename: str
    doi: str = ''
    journal: str = ''
    search_extra: str = ''
    text: str = field(init=False)
    search_text: str = field(init=False)
    search_length: int = field(init=False)
    counts: dict = field(init=False)
    tf: Counter = field(init=False)
    raw_tf: Counter = field(init=False)
    abstract_stem_tf: Counter = field(init=False)
    body_stem_tf: Counter = field(init=False)
    full_stem_tf: Counter = field(init=False)
    stem_tf: Counter = field(init=False)
    sentences: list = field(init=False)

    def __post_init__(self):
        # Search abstract + body. The title is metadata only: it is displayed,
        # but does not participate in matching, BM25, counts, or highlighting.
        self.text = '\n'.join(x for x in (self.abstract, self.body) if x)
        self.search_text = self.text
        self.raw_tf = Counter(tokens(self.search_text))
        self.tf = Counter(terms(self.text))
        # Keep query-term counts for each searchable region. Search still uses
        # abstract + body, while the UI can explain where every hit occurred.
        self.abstract_stem_tf = Counter(stem(t) for t in tokens(self.abstract))
        self.body_stem_tf = Counter(stem(t) for t in tokens(self.body))
        self.full_stem_tf = self.abstract_stem_tf + self.body_stem_tf
        # Compatibility for older callers: stem_tf means abstract-only counts.
        self.stem_tf = self.abstract_stem_tf
        self.search_length = sum(self.tf.values())
        # Displayed statistics use the abstract only. The body remains searchable,
        # but it does not affect Words / Sentences / Characters.
        statistics_text = self.abstract
        stats_raw_tf = Counter(tokens(statistics_text))
        stats_tf = Counter(terms(statistics_text))
        self.sentences = split_sentences(statistics_text)
        self.counts = {'characters': len(statistics_text),
                       'characters_no_space': sum(not c.isspace() for c in statistics_text),
                       'words': sum(stats_raw_tf.values()), 'sentences': len(self.sentences),
                       'indexed_terms': sum(stats_tf.values()), 'unique_terms': len(stats_tf)}


def parse_xml(path):
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) > 20 * 1024 * 1024:
        raise ValueError('File exceeds the 20 MB per-file limit')
    # Also accept the pubmed.json collection produced by the Project #2
    # downloader. This lets an existing collection move into the web app
    # without converting it back to XML.
    if path.suffix.lower() == '.json':
        try:
            collection = json.loads(raw.decode('utf-8-sig'))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f'Invalid UTF-8 PubMed JSON: {exc}') from exc
        records = collection.get('documents') if isinstance(collection, dict) else None
        if not isinstance(records, list):
            raise ValueError('PubMed JSON requires a documents array')
        docs = []
        for record in records:
            if not isinstance(record, dict):
                continue
            pmid = str(record.get('pmid', '')).strip()
            title = str(record.get('title', '')).strip()
            abstract = str(record.get('abstract', '')).strip()
            if not pmid or not abstract:
                continue
            normalized_id = pmid if pmid.upper().startswith('PMID') else 'PMID' + pmid
            docs.append(Document(normalized_id, title or '(Untitled)', abstract, '', path.name))
        if not docs:
            raise ValueError('PubMed JSON contains no records with both PMID and abstract')
        return docs
    if b'<!ENTITY' in raw.upper() or b'\x00' in raw:
        raise ValueError('Custom XML entities / non-UTF8-compatible encoding are not supported')
    root = ET.fromstring(raw)
    if root.tag == '{http://www.w3.org/2005/Atom}feed':
        from arxiv_bulk import entries
        docs = [Document(record['id'], record['title'], record['abstract'], '',
                         path.name, record['doi'], record['journal'])
                for record in entries(raw)]
        if not docs:
            raise ValueError('arXiv feed contains no records with abstracts')
        return docs
    for n in root.iter():
        if isinstance(n.tag, str):
            n.tag = n.tag.split('}')[-1]
    # PubMed EFetch XML can contain hundreds or thousands of abstract records
    # in one file. These records intentionally have no full-text <body>.
    pubmed_articles = ([root] if root.tag == 'PubmedArticle'
                       else root.findall('.//PubmedArticle'))
    if pubmed_articles:
        docs = []
        for article in pubmed_articles:
            citation = article.find('./MedlineCitation')
            article_meta = citation.find('./Article') if citation is not None else None
            if citation is None or article_meta is None:
                continue
            pmid = (citation.findtext('./PMID') or '').strip()
            title = extract_text(article_meta.find('./ArticleTitle'))
            abstract_parts = []
            for part in article_meta.findall('./Abstract/AbstractText'):
                text = extract_text(part)
                if not text:
                    continue
                label = (part.get('Label') or part.get('NlmCategory') or '').strip()
                abstract_parts.append(f'{label}: {text}' if label and label != 'UNASSIGNED' else text)
            abstract = '\n'.join(abstract_parts)
            if not pmid or not abstract:
                continue
            doi = ''
            for identifier in article_meta.findall('./ELocationID'):
                if (identifier.get('EIdType') or '').lower() == 'doi':
                    doi = ''.join(identifier.itertext()).strip()
                    break
            if not doi:
                for identifier in article.findall('./PubmedData/ArticleIdList/ArticleId'):
                    if (identifier.get('IdType') or '').lower() == 'doi':
                        doi = ''.join(identifier.itertext()).strip()
                        break
            journal = extract_text(article_meta.find('./Journal/Title'))
            docs.append(Document(f'PMID{pmid}', title or '(Untitled)', abstract, '',
                                 path.name, doi, journal))
        if not docs:
            raise ValueError('PubMed XML contains no records with both PMID and abstract')
        return docs
    articles = [root] if root.tag == 'article' else root.findall('.//article')
    if not articles:
        raise ValueError('No PMC JATS <article> or PubMed <PubmedArticle> record found')
    docs = []
    for number, article in enumerate(articles, 1):
        meta = article.find('./front/article-meta')
        if meta is None:
            raise ValueError('Missing front/article-meta')
        title = extract_text(meta.find('./title-group/article-title'))
        body_node = article.find('./body')
        # Drop section headings from body/abstract to make sentence units prose-like.
        # Preserve all prose, figure/table captions and table values.
        for container in [*meta.findall('./abstract'), body_node]:
            if container is not None:
                for parent in container.iter():
                    for child in list(parent):
                        if child.tag == 'title':
                            parent.remove(child)
        abstract = '\n'.join(extract_text(a) for a in meta.findall('./abstract'))
        body = extract_text(body_node)
        search_extra = ''  # No back matter or detached captions.
        if not body:
            raise ValueError('Missing non-empty body; a full-text article is required')
        ids = {a.get('pub-id-type'): ''.join(a.itertext()).strip() for a in meta.findall('./article-id')}
        pmcid = ids.get('pmcid') or ids.get('pmc')
        if pmcid and pmcid.isdigit():
            pmcid = 'PMC' + pmcid
        doc_id = pmcid or f'{path.name}#{number}'
        docs.append(Document(doc_id, title or '(Untitled)', abstract, body, path.name,
                             ids.get('doi', ''), extract_text(article.find('./front/journal-meta/journal-title-group/journal-title')),
                             search_extra))
    return docs


class SearchEngine:
    def __init__(self, folder):
        started = time.perf_counter()
        self.folder = Path(folder).resolve()
        self.docs = {}
        self.errors = []
        self.index = defaultdict(dict)
        deleted_path = self.folder / 'deleted_documents.json'
        self.deleted_ids = set()
        if deleted_path.exists():
            try:
                deleted_value = json.loads(deleted_path.read_text(encoding='utf-8'))
                if not isinstance(deleted_value, list) or not all(isinstance(item, str) for item in deleted_value):
                    raise ValueError('expected a JSON array of document IDs')
                self.deleted_ids = set(deleted_value)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                self.errors.append(f'deleted_documents.json: {exc}')
        paths = sorted(
            p for p in Path(folder).iterdir()
            if (p.suffix.lower() in {'.xml', '.nxml'} or
                (p.suffix.lower() == '.json' and
                 (p.name.casefold().startswith('pubmed') or p.name.startswith('upload_'))))
        )
        for path in paths:
            try:
                for doc in parse_xml(path):
                    if doc.id in self.deleted_ids:
                        continue
                    if doc.id in self.docs:
                        self.errors.append(f'{path.name}: duplicate {doc.id}; kept the first file')
                        continue
                    self.docs[doc.id] = doc
                    for term, frequency in doc.tf.items():
                        self.index[term][doc.id] = frequency
            except (ValueError, ET.ParseError, OSError) as exc:
                self.errors.append(f'{path.name}: {exc}')
        self.avg_len = sum(d.search_length for d in self.docs.values()) / max(1, len(self.docs))
        # Preserve the existing order, but always compact active document
        # numbers to 1..N. This removes gaps left by earlier deletions.
        number_path = self.folder / 'document_numbers.json'
        self.numbers = json.loads(number_path.read_text(encoding='utf-8')) if number_path.exists() else {}
        if (not isinstance(self.numbers, dict) or
            any(type(v) is not int or v < 1 for v in self.numbers.values()) or
            len(set(self.numbers.values())) != len(self.numbers)):
            raise ValueError('Invalid document_numbers.json; restore its backup before restarting')
        next_number = max(self.numbers.values(), default=0) + 1
        changed = not number_path.exists()
        for doc_id in self.docs:
            if doc_id not in self.numbers:
                self.numbers[doc_id] = next_number
                next_number += 1
                changed = True
        ordered_ids = sorted(
            self.docs,
            key=lambda doc_id: (self.numbers[doc_id], doc_id)
        )
        compact_numbers = {
            doc_id: number for number, doc_id in enumerate(ordered_ids, 1)
        }
        if compact_numbers != self.numbers:
            self.numbers = compact_numbers
            changed = True
        if changed:
            temp = number_path.with_suffix('.json.tmp')
            temp.write_text(json.dumps(self.numbers, ensure_ascii=False, indent=2), encoding='utf-8')
            os.replace(temp, number_path)
        self.build_ms = (time.perf_counter() - started) * 1000

    def search(self, query, mode='AND', doc_id=None):
        if mode not in {'AND', 'OR', 'PHRASE'}:
            raise ValueError('Unknown search mode')
        if doc_id is not None and doc_id not in self.docs:
            return [], []
        query_tokens = list(dict.fromkeys(terms(query)))
        if mode == 'PHRASE':
            phrase = [stem(t) for t in tokens(query)]
            if not phrase:
                return [], []
            # Phrase matching retains stopwords; exact contiguous token sequence.
            candidates = set()
            for doc in self.docs.values():
                for paragraph in doc.text.splitlines():
                    dt = [stem(t) for t in tokens(paragraph)]
                    if any(dt[i:i+len(phrase)] == phrase for i in range(len(dt)-len(phrase)+1)):
                        candidates.add(doc.id)
                        break
        else:
            if not query_tokens:
                return [], []
            postings = [set(self.index.get(t, {})) for t in query_tokens]
            candidates = set.intersection(*postings) if mode == 'AND' else set.union(*postings)
        if doc_id is not None:
            candidates &= {doc_id}
        n = len(self.docs)
        results = []
        for doc_id in candidates:
            doc = self.docs[doc_id]
            score = 0.0
            for term in query_tokens:
                tf = doc.tf[term]
                if not tf:
                    continue
                df = len(self.index[term])
                idf = math.log(1 + (n-df+0.5)/(df+0.5))
                norm = 1.2 * (1-0.75 + 0.75 * doc.search_length/max(self.avg_len, 1))
                score += idf * tf * 2.2 / (tf + norm)
            results.append((doc, score))
        results.sort(key=lambda pair: (-pair[1], pair[0].id))
        return results, query_tokens
