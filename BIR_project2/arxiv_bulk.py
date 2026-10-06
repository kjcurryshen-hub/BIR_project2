"""arXiv abstract imports using the public Atom API; no third-party packages.

API reference: https://info.arxiv.org/help/api/user-manual.html
"""

import re
import threading
import time
from urllib.parse import urlencode, urlsplit, unquote
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import xml.etree.ElementTree as ET

BASE = 'https://export.arxiv.org/api/query'
ATOM = '{http://www.w3.org/2005/Atom}'
OPEN = '{http://a9.com/-/spec/opensearch/1.1/}'
ARXIV = '{http://arxiv.org/schemas/atom}'
MAX_RESPONSE_BYTES = 20 * 1024 * 1024
_lock = threading.Lock()
_last_request = None


def normalize_id(value):
    value = unquote(str(value).strip())
    if value.startswith(('http://', 'https://')):
        url = urlsplit(value)
        if url.hostname not in {'arxiv.org', 'www.arxiv.org', 'export.arxiv.org'}:
            raise ValueError('Please use an arXiv ID or an arxiv.org URL.')
        value = re.sub(r'^/(?:abs|pdf)/', '', url.path).removesuffix('.pdf')
    value = re.sub(r'^arxiv\s*:\s*', '', value, flags=re.I)
    value = re.sub(r'v[1-9]\d*$', '', value, flags=re.I)
    if not re.fullmatch(r'(?:\d{4}\.\d{4,5}|[a-z][a-z0-9.-]*/\d{7})', value, re.I):
        raise ValueError(f'Invalid arXiv ID: {value}')
    return value.lower()


def parse_id_batch(value):
    cleaned = re.sub(r'arxiv\s*:\s*(?=[a-z0-9])', '', value, flags=re.I)
    pieces = [p for p in re.split(r'[\s,;，；]+', cleaned.strip()) if p]
    if not pieces:
        raise ValueError('請至少輸入 1 個 arXiv ID。')
    ids, seen, duplicates = [], set(), 0
    for piece in pieces:
        identifier = normalize_id(piece)
        if identifier in seen:
            duplicates += 1
        else:
            seen.add(identifier)
            ids.append(identifier)
    if len(ids) > 200:
        raise ValueError('一次最多可輸入 200 個不同的 arXiv ID。')
    return ids, duplicates


def default_query(title):
    phrase = re.sub(r'["\\]', ' ', title).strip()
    return f'(ti:"{phrase}" OR abs:"{phrase}")' if phrase else ''


def _request(parameters):
    # Serialize calls (including retries) and leave at least three seconds
    # between requests, as requested by the official arXiv API documentation.
    global _last_request
    with _lock:
        for attempt in range(3):
            if _last_request is not None:
                time.sleep(max(0, 3.0 - (time.monotonic() - _last_request)))
            _last_request = time.monotonic()
            try:
                req = Request(BASE + '?' + urlencode(parameters), headers={
                    'User-Agent': 'Biomedical-IR-Coursework/3.0',
                    'Accept': 'application/atom+xml',
                })
                with urlopen(req, timeout=60) as response:
                    raw = response.read(MAX_RESPONSE_BYTES + 1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise ValueError('arXiv response exceeds 20 MB.')
                return raw
            except HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504}:
                    raise OSError(f'arXiv HTTP {exc.code}; check your query or IDs.') from exc
                last_error = exc
            except OSError as exc:
                last_error = exc
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
        raise OSError(f'arXiv request failed after retries: {last_error}')


def _read_feed(raw):
    if b'<!ENTITY' in raw.upper() or b'\x00' in raw:
        raise ValueError('Unsupported arXiv XML encoding or entities.')
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ValueError(f'Invalid arXiv Atom response: {exc}') from exc
    if root.tag != ATOM + 'feed':
        raise ValueError('arXiv did not return an Atom feed.')
    for entry in root.findall(ATOM + 'entry'):
        if '/api/errors' in (entry.findtext(ATOM + 'id') or ''):
            raise ValueError('arXiv: ' + (entry.findtext(ATOM + 'summary') or 'Invalid query or ID.'))
    return root


def entries(raw):
    """Return stable base IDs and abstracts, excluding versions and empty text."""
    root = _read_feed(raw)
    records, seen = [], set()
    for entry in root.findall(ATOM + 'entry'):
        identifier = normalize_id(entry.findtext(ATOM + 'id') or '')
        abstract = ' '.join((entry.findtext(ATOM + 'summary') or '').split())
        if not abstract or identifier in seen:
            continue
        seen.add(identifier)
        records.append({
            'id': 'ARXIV:' + identifier,
            'title': ' '.join((entry.findtext(ATOM + 'title') or '(Untitled)').split()),
            'abstract': abstract,
            'doi': entry.findtext(ARXIV + 'doi') or '',
            'journal': entry.findtext(ARXIV + 'journal_ref') or 'arXiv preprint',
        })
    return records


def fetch_abstracts(query, count=500, email='', retstart=0):
    query = query.strip()
    if not 1 <= len(query) <= 300:
        raise ValueError('arXiv query must contain 1–300 characters.')
    if not 10 <= count <= 1000 or not 0 <= retstart <= 10000:
        raise ValueError('Use 10–1,000 abstracts and an offset between 0 and 10,000.')
    raw = _request({'search_query': query, 'start': retstart, 'max_results': count,
                    'sortBy': 'relevance', 'sortOrder': 'descending'})
    root = _read_feed(raw)
    downloaded = len(root.findall(ATOM + 'entry'))
    if not entries(raw):
        raise ValueError('arXiv found no usable abstracts for this query or next batch.')
    return raw, {'query': query, 'requested': count,
                 'available': int(root.findtext(OPEN + 'totalResults') or downloaded),
                 'downloaded': downloaded, 'retstart': retstart}


def fetch_ids(ids, email=''):
    ids = list(dict.fromkeys(normalize_id(i) for i in ids))
    if not 1 <= len(ids) <= 200:
        raise ValueError('Use 1–200 unique arXiv IDs.')
    raw = _request({'id_list': ','.join(ids), 'max_results': len(ids)})
    root = _read_feed(raw)
    returned = {normalize_id(e.findtext(ATOM + 'id') or '') for e in root.findall(ATOM + 'entry')}
    usable = len(entries(raw))
    if not usable:
        raise ValueError('arXiv returned no abstracts for these IDs.')
    return raw, {'requested': len(ids), 'returned': len(returned), 'usable': usable,
                 'not_found': len(set(ids) - returned), 'without_abstract': len(returned) - usable}


def remove_known_records(raw, known_ids):
    root = _read_feed(raw)
    seen, removed = set(known_ids), 0
    for entry in list(root.findall(ATOM + 'entry')):
        identifier = 'ARXIV:' + normalize_id(entry.findtext(ATOM + 'id') or '')
        abstract = (entry.findtext(ATOM + 'summary') or '').strip()
        if identifier in seen or not abstract:
            root.remove(entry)
            removed += 1
        else:
            seen.add(identifier)
    return ET.tostring(root, encoding='utf-8', xml_declaration=True), len(root.findall(ATOM + 'entry')), removed
