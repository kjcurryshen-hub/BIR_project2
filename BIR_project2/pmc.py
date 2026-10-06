"""Resolve PMCIDs via the official PMC AWS dataset. No user-supplied URLs fetched."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time
import threading
from urllib.parse import urlencode
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from engine import parse_xml

BASE = 'https://pmc-oa-opendata.s3.amazonaws.com/'
IDCONV = 'https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/'
_cache = {}
_lock = threading.Lock()


def normalize_pmcid(value):
    value = value.strip()
    if not re.fullmatch(r'PMC[1-9]\d{0,11}', value, re.I):
        raise ValueError('請輸入 PMCID，例如 PMC12503546；不是 PMID 或 DOI。')
    return value.upper()


def normalize_pmid(value):
    """Accept a plain PMID or an optional PMID prefix, never a PMCID."""
    value = value.strip()
    match = re.fullmatch(r'(?:PMID\s*[:#]?\s*)?([1-9]\d{0,11})', value, re.I)
    if not match:
        raise ValueError('請輸入 PMID，例如 42724776；不是 PMCID、DOI 或網址。')
    return match[1]


def resolve_pmid(value):
    """Resolve one PMID to its PMCID through NCBI's official ID converter."""
    pmid = normalize_pmid(value)
    url = IDCONV + '?' + urlencode({
        'ids': pmid, 'format': 'json', 'tool': 'BiomedicalIR-StudentProject'
    })
    request = urllib.request.Request(url, headers={'User-Agent':'BiomedicalIR-StudentProject/4.0'})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read(1024*1024+1)
    except urllib.error.HTTPError as exc:
        raise ValueError(f'NCBI PMID 轉換服務回應 HTTP {exc.code}，請稍後重試。') from exc
    except (OSError, urllib.error.URLError) as exc:
        raise ValueError('連線 NCBI 失敗或逾時，請確認網路後重試。') from exc
    if len(raw) > 1024*1024:
        raise ValueError('NCBI PMID 轉換回應異常。')
    try:
        result = json.loads(raw)
        records = result.get('records', [])
        record = records[0] if len(records) == 1 else {}
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError('無法解析 NCBI PMID 轉換結果。') from exc
    if str(record.get('pmid', '')) != pmid or not record.get('pmcid'):
        raise ValueError('此 PMID 沒有對應的 PMCID，因此無法取得 PMC 全文 XML。')
    return pmid, normalize_pmcid(str(record['pmcid']))


def fetch(url):
    if not url.startswith(BASE):
        raise ValueError('來源不是 PMC 官方資料服務。')
    request = urllib.request.Request(url, headers={'User-Agent':'BiomedicalIR-StudentProject/3.0'})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read(20*1024*1024+1)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise ValueError('官方資料服務目前沒有這份檔案。') from exc
        raise ValueError(f'PMC 資料服務回應 HTTP {exc.code}，請稍後重試。') from exc
    except (OSError, urllib.error.URLError) as exc:
        raise ValueError('連線 PMC 失敗或逾時，請確認網路後重試。') from exc
    if len(raw)>20*1024*1024:
        raise ValueError('檔案超過本工具的 20 MB 限制。')
    return raw


def metadata(pmcid, version):
    pmcid = normalize_pmcid(pmcid)
    if not re.fullmatch(r'[1-9]\d{0,5}', str(version)):
        raise ValueError('文章版本號無效。')
    key = f'{pmcid}.{version}'
    with _lock:
        cached = _cache.get(key)
        if cached and time.monotonic()-cached[0]<300:
            return dict(cached[1])
    meta = json.loads(fetch(BASE+f'{key}/{key}.json'))
    if not isinstance(meta,dict) or meta.get('pmcid')!=pmcid or str(meta.get('version'))!=str(version):
        raise ValueError('官方資料的文章識別碼不一致。')
    expected = f's3://pmc-oa-opendata/{key}/{key}.xml'
    source = meta.get('xml_url') or ''
    if source != expected and not re.fullmatch(re.escape(expected)+r'\?md5=[a-fA-F0-9]{32}', source):
        raise ValueError('此版本沒有可取得的全文 XML。')
    meta['source_metadata_url'] = BASE+f'{key}/{key}.json'
    meta['source_xml_https'] = BASE+source.removeprefix('s3://pmc-oa-opendata/')
    with _lock:
        if len(_cache)>=100:
            _cache.pop(next(iter(_cache)))
        _cache[key] = (time.monotonic(), dict(meta))
    return meta


def lookup(pmcid):
    pmcid = normalize_pmcid(pmcid)
    listing = ET.fromstring(fetch(BASE+'?'+urlencode({'list-type':'2','prefix':pmcid+'.','delimiter':'/','max-keys':'100'})))
    versions = []
    for node in listing.findall('.//{*}CommonPrefixes/{*}Prefix'):
        match = re.fullmatch(re.escape(pmcid)+r'\.([1-9]\d*)/',node.text or '')
        if match:
            versions.append(int(match[1]))
    if not versions:
        raise ValueError('此 PMCID 沒有可取得的全文 XML。請核對編號，或點 PMC 原文查看；這不一定代表文章不存在。')
    if len(versions)>20 or listing.findtext('{*}IsTruncated')=='true':
        raise ValueError('版本過多，請至 PMC 確認文章版本。')
    found, errors = [], []
    for version in sorted(set(versions)):
        try:
            found.append(metadata(pmcid,version))
        except (ValueError, ET.ParseError) as exc:
            errors.append(f'版本 {version}: {exc}')
    if not found:
        raise ValueError('；'.join(errors))
    return found, errors


def article_xml(pmcid, version):
    meta = metadata(pmcid,version)
    raw = fetch(meta['source_xml_https'])
    expected = re.search(r'[?&]md5=([a-fA-F0-9]{32})',meta['source_xml_https'])
    if expected and hashlib.md5(raw).hexdigest()!=expected[1].lower():
        raise ValueError('XML 校驗失敗，請稍後重試。')
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/f'{meta["pmcid"]}.{meta["version"]}.xml'
        path.write_bytes(raw)
        docs = parse_xml(path)
    if len(docs)!=1 or docs[0].id!=meta['pmcid']:
        raise ValueError('全文 XML 與 PMCID 不一致。')
    meta['sha256'] = hashlib.sha256(raw).hexdigest()
    meta['downloaded_at_utc'] = datetime.now(timezone.utc).isoformat()
    return raw, meta
