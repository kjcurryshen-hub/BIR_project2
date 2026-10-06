"""Bulk PubMed abstract download through NCBI E-utilities (standard library only)."""

import json
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


BASE = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
MAX_RESPONSE_BYTES = 25 * 1024 * 1024


def _request(endpoint, parameters):
    url = BASE + endpoint + '?' + urlencode(parameters)
    last_error = None
    for attempt in range(4):
        try:
            request = Request(url, headers={
                'User-Agent': 'NCKU-Biomedical-IR-Coursework/2.0',
                'Accept': 'application/json, application/xml;q=0.9',
            })
            with urlopen(request, timeout=45) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ValueError('PubMed response exceeds 25 MB')
            return raw
        except (OSError, ValueError) as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(1.5 * (attempt + 1))
    raise OSError(f'PubMed request failed after retries: {last_error}')


def fetch_abstracts(query='GLP-1[Title/Abstract]', count=500, email='', retstart=0):
    query = query.strip()
    if not query or len(query) > 300:
        raise ValueError('PubMed query must contain 1-300 characters')
    if not 10 <= count <= 1000:
        raise ValueError('Document count must be between 10 and 1000')
    if not 0 <= retstart <= 10000:
        raise ValueError('PubMed result offset must be between 0 and 10000')
    common = {'db': 'pubmed', 'tool': 'ncku_biomedical_ir_coursework'}
    if email:
        common['email'] = email
    search_parameters = {
        **common,
        'term': f'({query}) AND hasabstract[text]',
        'retmode': 'json',
        'retmax': count,
        'retstart': retstart,
        'sort': 'relevance',
    }
    try:
        search = json.loads(_request('esearch.fcgi', search_parameters))
        ids = search['esearchresult']['idlist']
        available = int(search['esearchresult']['count'])
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'Invalid PubMed search response: {exc}') from exc
    if not ids:
        raise ValueError('PubMed found no records with abstracts for this query')

    combined = ET.Element('PubmedArticleSet')
    for start in range(0, len(ids), 200):
        batch = ids[start:start + 200]
        fetch_parameters = {**common, 'id': ','.join(batch), 'retmode': 'xml'}
        try:
            root = ET.fromstring(_request('efetch.fcgi', fetch_parameters))
        except ET.ParseError as exc:
            raise ValueError(f'Invalid PubMed XML response: {exc}') from exc
        for article in root.findall('./PubmedArticle'):
            combined.append(article)
        if start + 200 < len(ids):
            # Without an API key NCBI asks clients to stay below 3 requests/sec.
            time.sleep(0.4)
    raw = ET.tostring(combined, encoding='utf-8', xml_declaration=True)
    returned = len(combined.findall('./PubmedArticle'))
    if not returned:
        raise ValueError('PubMed returned no usable abstract records')
    return raw, {'query': query, 'requested': count, 'available': available,
                 'downloaded': returned, 'retstart': retstart}


def fetch_pmids(pmids, email=''):
    """Fetch PubMed abstract records for an explicit ordered list of PMIDs."""
    normalized = []
    seen = set()
    for value in pmids:
        pmid = str(value).strip()
        if not pmid.isdigit() or pmid.startswith('0'):
            raise ValueError(f'Invalid PMID: {value}')
        if pmid not in seen:
            normalized.append(pmid)
            seen.add(pmid)
    if not 1 <= len(normalized) <= 200:
        raise ValueError('PMID batch must contain between 1 and 200 unique IDs')

    common = {'db': 'pubmed', 'tool': 'ncku_biomedical_ir_coursework'}
    if email:
        common['email'] = email
    combined = ET.Element('PubmedArticleSet')
    returned_ids = set()
    without_abstract = 0
    for start in range(0, len(normalized), 200):
        batch = normalized[start:start + 200]
        parameters = {**common, 'id': ','.join(batch), 'retmode': 'xml'}
        try:
            root = ET.fromstring(_request('efetch.fcgi', parameters))
        except ET.ParseError as exc:
            raise ValueError(f'Invalid PubMed XML response: {exc}') from exc
        for article in root.findall('./PubmedArticle'):
            pmid = (article.findtext('./MedlineCitation/PMID') or '').strip()
            if pmid:
                returned_ids.add(pmid)
            abstract_parts = article.findall('./MedlineCitation/Article/Abstract/AbstractText')
            has_abstract = any(''.join(part.itertext()).strip() for part in abstract_parts)
            if not has_abstract:
                without_abstract += 1
                continue
            combined.append(article)
        if start + 200 < len(normalized):
            time.sleep(0.4)

    usable = len(combined.findall('./PubmedArticle'))
    if not usable:
        raise ValueError('PubMed returned no records with abstracts for these PMIDs')
    raw = ET.tostring(combined, encoding='utf-8', xml_declaration=True)
    return raw, {
        'requested': len(normalized),
        'returned': len(returned_ids),
        'usable': usable,
        'not_found': len(set(normalized) - returned_ids),
        'without_abstract': without_abstract,
    }


def remove_known_records(raw, known_ids):
    """Remove PMIDs already loaded in the local collection before saving."""
    root = ET.fromstring(raw)
    removed = 0
    for article in list(root.findall('./PubmedArticle')):
        pmid = (article.findtext('./MedlineCitation/PMID') or '').strip()
        if pmid and f'PMID{pmid}' in known_ids:
            root.remove(article)
            removed += 1
    filtered = ET.tostring(root, encoding='utf-8', xml_declaration=True)
    return filtered, len(root.findall('./PubmedArticle')), removed
