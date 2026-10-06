"""Run: python app.py. Open http://127.0.0.1:8765 . No pip install required."""
import argparse
import csv
import html
import io
import json
import math
import re

# Use the Windows certificate store when truststore is installed. This fixes
# school/VPN/antivirus HTTPS inspection without disabling certificate checks.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import pmc
import pubmed_bulk
import arxiv_bulk
from collection_store import CollectionStore
from spelling import correct_query
from word2vec_model import MODEL_NAMES, train_word2vec, nearest_words, pca_projection
import hashlib
import secrets
import tempfile
import threading
import socket
import xml.etree.ElementTree as ET
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import time
from urllib.parse import parse_qs, urlencode, urlsplit
from porter import stem
from engine import SearchEngine, TOKEN, tokens, terms, parse_xml
from zipf_analysis import CONDITIONS, analyze_all, idf, query_term

ROOT = Path(__file__).resolve().parent

def esc(value):
    return html.escape(str(value), quote=True)


def safe_csv(value):
    """Prevent spreadsheet formula evaluation for externally supplied text."""
    return "'" + value if isinstance(value, str) and value.startswith(('=', '+', '-', '@', '\t', '\r')) else value


def parse_pmid_batch(value):
    """Parse a pasted PMID list separated by whitespace, commas, or semicolons."""
    cleaned = re.sub(r'PMID\s*[:#]?\s*(?=\d)', '', value, flags=re.I)
    pieces = [piece for piece in re.split(r'[\s,;，；]+', cleaned.strip()) if piece]
    if not pieces:
        raise ValueError('請至少輸入 1 個 PMID。')
    pmids = []
    seen = set()
    duplicate_count = 0
    for piece in pieces:
        pmid = piece
        if not re.fullmatch(r'[1-9]\d*', pmid):
            raise ValueError(f'PMID「{piece}」格式不正確，請只輸入數字。')
        if pmid in seen:
            duplicate_count += 1
            continue
        seen.add(pmid)
        pmids.append(pmid)
    if len(pmids) > 200:
        raise ValueError('一次最多可輸入 200 個不同的 PMID。')
    return pmids, duplicate_count


def source_options(selected=''):
    return ('<option value="">Select a source…</option>' + ''.join(
        '<option value="'+key+'"'+(' selected' if key == selected else '')+'>'+label+'</option>'
        for key, label in [('pubmed', 'PubMed'), ('arxiv', 'arXiv')]))


def bulk_import_form(token, item=None):
    """One source-aware form for both new topics and additional batches."""
    source = item.get('source', 'pubmed') if item else ''
    queries = item.get('source_queries', {}) if item else {}
    query = item.get('query', '') if item else ''
    hidden = '<input type="hidden" name="token" value="'+esc(token)+'">'
    if item:
        hidden += '<input type="hidden" name="collection_id" value="'+esc(item['id'])+'">'
    return ('<form class="limit-form source-form" action="/abstract-import" method="post" '
            'data-import-source data-mode="bulk" data-queries="'+esc(json.dumps(queries))+'" '
            'data-topic="'+esc(item['title'] if item else '')+'" '
            'data-wait="正在下載摘要並建立索引；500 篇可能需要一至數分鐘…">'+hidden+
            '<label class="source-choice">Source<select name="source" required>'+source_options(source)+'</select></label>'
            '<fieldset class="import-fields"'+('' if source else ' hidden disabled')+'>'+
            ('' if item else '<div class="query"><label>主題名稱（顯示於網站）<input name="title" '
             'placeholder="例如 GLP-1 或 Machine learning" maxlength="80" required></label></div>')+
            '<div class="query"><label><span data-query-label>搜尋條件</span><input name="query" value="'+esc(query)+'" '
            'maxlength="300" required></label></div>'
            '<div><label>'+('追加篇數' if item else '篇數')+'<input name="count" type="number" value="500" '
            'min="10" max="1000" required></label><small class="field-limit">Maximum: 1,000 abstracts per import.</small></div>'
            '<div class="query" data-ncbi-email><label>NCBI 聯絡信箱（選填）<input name="email" type="email" '
            'maxlength="200" placeholder="你的學校信箱"></label></div>'
            '<button type="submit">'+('Add Next Batch' if item else 'Create Topic &amp; Import')+'</button>'
            '<p class="muted" data-source-help></p></fieldset><noscript>請啟用 JavaScript 以選擇來源並顯示匯入欄位。</noscript></form>')

def highlight(text, query):
    wanted = set(stem(t) for t in tokens(query))
    out, end = [], 0
    for m in TOKEN.finditer(text):
        if stem(m.group()) in wanted:
            out += [esc(text[end:m.start()]), '<mark>' + esc(m.group()) + '</mark>']
            end = m.end()
    return ''.join(out) + esc(text[end:])

def snippet(doc, query):
    wanted = set(stem(t) for t in tokens(query))
    match = next((m for m in TOKEN.finditer(doc.text) if stem(m.group()) in wanted), None)
    start = max(0, match.start()-100) if match else 0
    end = min(len(doc.text), start+420)
    return ('…' if start else '') + highlight(doc.text[start:end], query) + ('…' if end<len(doc.text) else '')


def document_term_counts(doc, query):
    """Return each distinct query stem and its abstract/body/full counts."""
    query_terms = list(dict.fromkeys(stem(t) for t in tokens(query)))
    return [
        (term,
         doc.abstract_stem_tf[term],
         doc.body_stem_tf[term],
         doc.full_stem_tf[term])
        for term in query_terms
    ]


def term_counts_html(doc, query):
    counts = document_term_counts(doc, query)
    if not counts:
        return ''
    abstract_total = sum(abstract for _, abstract, _, _ in counts)
    body_total = sum(body for _, _, body, _ in counts)
    full_total = sum(full for _, _, _, full in counts)
    rows = ''.join(
        f'<tr><td>{esc(term)}</td><td>{abstract:,} 次</td>'
        f'<td>{body:,} 次</td><td>{full:,} 次</td></tr>'
        for term, abstract, body, full in counts
    )
    return (
        '<details class="term-counts" open>'
        '<summary>這篇文章的搜尋詞分布：'
        f'摘要 {abstract_total:,} 次 · 正文 {body_total:,} 次 · 全文 {full_total:,} 次</summary>'
        '<div class="scroll"><table>'
        '<tr><th>搜尋詞</th><th>摘要次數</th><th>正文次數</th><th>全文總次數</th></tr>'
        f'{rows}</table></div>'
        '<p class="muted">AND／OR／PHRASE 依摘要＋正文判斷；下方 Words、Sentences、Characters 只統計摘要。</p>'
        '</details>'
    )

STYLE = '''
:root{color-scheme:light;--ink:#122d3d;--muted:#526a79;--teal:#00796f}*{box-sizing:border-box}
body{margin:0;background:#f2f6f8;color:var(--ink);font:16px/1.65 system-ui,"Microsoft JhengHei",sans-serif}
header{background:#102e3f;color:white;padding:30px max(5vw,20px)}.header-layout{max-width:1500px;margin:auto;display:grid;grid-template-columns:minmax(310px,1fr) minmax(620px,1.7fr);gap:36px;align-items:center}header h1{margin:6px 0;font-size:40px}
header p{margin:6px 0;color:#c0d9e2;font-size:20px}.eyebrow{letter-spacing:2px;font-size:16px;color:#8cddd0}.home-link{color:white;text-decoration:none}.home-link:hover,.home-link:focus{text-decoration:underline}
main{max-width:1500px;margin:auto;padding:26px 20px 60px}.panel,.card{background:white;border:1px solid #dbe5e9;border-radius:14px;padding:22px;margin:0 0 18px}
.header-metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.header-metrics .metric{min-width:0;background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.2);padding:15px 16px;border-radius:12px}.header-metrics small{display:block;color:#bcd6df;white-space:nowrap}.header-metrics b{display:block;color:white;font-size:27px;line-height:1.25;margin-top:4px}.muted,small{color:var(--muted)}
form{display:flex;gap:12px;align-items:end;flex-wrap:wrap}label{display:block;font-size:14px}input,select,button,textarea{font:inherit;padding:11px;border:1px solid #b9cbd4;border-radius:7px}input,textarea{width:100%}textarea{min-height:100px;resize:vertical}.query{flex:1;min-width:220px}button{background:#176b67;color:white;cursor:pointer;border-color:#125c58;border-radius:8px;font-weight:650;box-shadow:0 2px 5px rgba(16,74,76,.14);transition:background .16s ease,border-color .16s ease,box-shadow .16s ease,transform .16s ease}button:not(.dialog-close):hover,button:not(.dialog-close):focus{background:#105b57;border-color:#0d4f4c;transform:translateY(-1px);box-shadow:0 4px 9px rgba(16,74,76,.18)}button:disabled{cursor:not-allowed;opacity:.55;transform:none!important;box-shadow:none}a{color:#00736c}h2{font-size:22px;margin:0 0 12px}h3{font-size:19px;margin:5px 0 8px}.row{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}.chip{font-size:12px;background:#e6f4f2;color:#126f67;padding:3px 9px;border-radius:16px;display:inline-block}.stats{display:flex;flex-wrap:wrap;gap:16px;color:var(--muted);font-size:14px}mark{background:#ffe29a;padding:0 2px}pre{white-space:pre-wrap;word-break:break-word;font:inherit;max-height:560px;overflow:auto}summary{cursor:pointer;font-weight:600}table{width:100%;border-collapse:collapse}th,td{text-align:left;border-bottom:1px solid #dbe5e9;padding:9px}th{background:#edf4f6}.bar{height:8px;background:#e6f0f2;border-radius:4px;margin-top:10px}.bar span{display:block;height:100%;background:var(--teal);border-radius:4px}.warning{background:#fff3d8;padding:14px;border-radius:8px}code{background:#edf2f4;padding:2px 5px}footer{margin-top:30px;font-size:13px;color:var(--muted)}.scroll{overflow-x:auto}@media(max-width:650px){.metrics{grid-template-columns:repeat(2,1fr)}header h1{font-size:25px}}
.field-limit{display:block;margin-top:6px;color:#647786;font-size:12px;letter-spacing:.025em}.limit-form{align-items:flex-start}.limit-form>div{display:flex;flex-direction:column}.limit-form input,.limit-form select{min-height:56px}.limit-form>button{min-height:56px;margin-top:25px}.analysis-actions{display:flex;align-items:center;justify-content:flex-end;gap:12px;flex-wrap:wrap}.analysis-actions>a:not(.compare-launch){white-space:nowrap}.compare-launch{display:inline-flex;align-items:center;justify-content:center;min-width:178px;padding:12px 20px;color:white;text-decoration:none;font-weight:700;background:#173f50;border:1px solid #123543;border-radius:9px;box-shadow:0 3px 8px rgba(16,46,63,.18);transition:background .16s ease,transform .16s ease,box-shadow .16s ease}.compare-launch:hover,.compare-launch:focus{color:white;background:#102f3d;transform:translateY(-1px);box-shadow:0 5px 11px rgba(16,46,63,.22)}.comparison-form{align-items:end}.comparison-form>div{flex:1;min-width:210px}.comparison-hero{overflow:hidden;position:relative;background:#173f50;color:white;border:0;box-shadow:0 6px 18px rgba(16,46,63,.14)}.comparison-hero::after{content:"";position:absolute;width:260px;height:260px;border-radius:50%;right:-90px;top:-150px;background:rgba(140,221,208,.08)}.comparison-hero p,.comparison-hero .muted{color:#d2e4e9}.comparison-hero .chip{background:rgba(255,255,255,.12);color:white}.domain-head-a{color:#096e92}.domain-head-b{color:#8b4ea8}.top-terms-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.comparison-note{padding:14px 16px;border-left:4px solid #00796f;background:#eef8f6;border-radius:9px}
.workspace{display:grid;grid-template-columns:340px minmax(0,1fr);gap:24px;align-items:start}
.sidebar{min-width:0}.results-pane{min-width:0}.sidebar .panel{padding:18px}.sidebar h2{font-size:19px}
.sidebar form{align-items:stretch}.sidebar .query{flex-basis:100%;min-width:0}.sidebar button{width:100%}
.sidebar .card{padding:14px}.sidebar h3{font-size:16px;overflow-wrap:anywhere}.sidebar .row{display:block}
.library-item{border-top:1px solid #dbe5e9;padding:14px 0;overflow-wrap:anywhere}.library-item .stats{font-size:12px;gap:5px}
.library-item>a{font-weight:600;font-size:14px}.sidebar .muted{font-size:13px}.results-pane .card h3{overflow-wrap:anywhere}
.library-filter{margin:8px 0 4px}.delete-form{margin-top:9px}.sidebar .delete-form button{width:auto;padding:6px 11px;font-size:13px}.library-item[hidden]{display:none}input[type=file]::file-selector-button{margin-right:10px;padding:8px 12px;color:white;font:inherit;font-weight:650;background:#176b67;border:1px solid #125c58;border-radius:7px;cursor:pointer;box-shadow:0 2px 5px rgba(16,74,76,.13);transition:background .16s ease,box-shadow .16s ease}input[type=file]::file-selector-button:hover{background:#105b57;box-shadow:0 4px 8px rgba(16,74,76,.17)}
.term-counts{margin:14px 0;padding:12px 14px;background:#f5faf9;border:1px solid #cfe4df;border-radius:9px}.term-counts table{margin-top:8px}.term-counts th,.term-counts td{padding:6px 8px}
.analysis-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:14px 0}.model-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:14px 0}.analysis-grid .metric-box,.model-grid .metric-box{background:#f5faf9;border:1px solid #cfe4df;border-radius:10px;padding:14px}.metric-box small{display:block}.metric-box b{font-size:23px}.training-metrics{display:grid;grid-template-columns:repeat(8,minmax(0,1fr));gap:14px;margin:20px 0}.training-metrics .metric-box{grid-column:span 2;position:relative;overflow:hidden;min-height:112px;padding:20px 18px;background:#f8fbfb;border:1px solid #cedee1;border-radius:11px;box-shadow:0 2px 6px rgba(16,46,63,.06)}.training-metrics .metric-box::before{content:"";position:absolute;inset:0 auto 0 0;width:4px;background:#427d7a}.training-metrics .metric-box:nth-child(5){grid-column:2/span 2}.training-metrics .metric-box:nth-child(6){grid-column:4/span 2}.training-metrics .metric-box:nth-child(7){grid-column:6/span 2}.training-metrics small{color:#58717e;font-size:13px;letter-spacing:.04em}.training-metrics b{display:block;margin-top:7px;font-size:29px;line-height:1.2;color:#102e3f}.chart{width:100%;height:auto;display:block;border:1px solid #dbe5e9;border-radius:10px;background:white}.chart-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.distribution-bar{min-width:90px;height:8px;background:#e5edef;border-radius:5px}.distribution-bar span{display:block;height:100%;background:#0b8f83;border-radius:5px}.condition-tabs{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}.condition-tabs a{text-decoration:none;border:1px solid #a9c5c5;border-radius:8px;padding:7px 14px;background:white;box-shadow:0 1px 3px rgba(16,74,76,.07);transition:transform .16s ease,box-shadow .16s ease,background .16s ease}.condition-tabs a:hover,.condition-tabs a:focus{transform:translateY(-1px);background:#f3f8f7;box-shadow:0 3px 7px rgba(16,74,76,.11)}.condition-tabs a.active{background:#176b67;color:white;border-color:#125c58;box-shadow:0 2px 5px rgba(16,74,76,.14)}.formula{font-family:Cambria,"Times New Roman",serif;font-size:18px}.nowrap{white-space:nowrap}
.page-nav{max-width:1500px;margin:24px auto 0;display:flex;gap:10px;flex-wrap:wrap;align-items:center}.page-nav a,.page-nav .upload-nav-button{color:#d7eeef;text-decoration:none;border:1px solid #52717e;border-radius:8px;padding:9px 20px;font-weight:650;background:rgba(255,255,255,.035);line-height:1.65;box-shadow:none;transition:transform .16s ease,background .16s ease}.page-nav a[aria-current="page"]{background:#e6f4f2;color:#125f59;border-color:#e6f4f2}.page-nav a:hover,.page-nav .upload-nav-button:hover{background:#244b5b;color:white;transform:translateY(-1px);box-shadow:none}.page-nav a:focus-visible,.page-nav .upload-nav-button:focus-visible{outline:3px solid #8cddd0;outline-offset:3px}.page-nav .upload-nav-button{margin-left:auto}.management .library-item{padding:20px 0}.management .library-item>a{font-size:19px}.management .delete-form button{padding:6px 12px;font-size:14px}.management input[type=file]{min-width:0;max-width:100%}.search-workspace{grid-template-columns:260px minmax(0,1fr)}
.collection-nav{max-width:1500px;margin:14px auto 0;display:flex;align-items:center;gap:8px;flex-wrap:wrap}.collection-nav>span{color:#bcd6df}.collection-nav a{color:#d7eeef;text-decoration:none;border:1px solid #52717e;border-radius:20px;padding:5px 12px;transition:transform .16s ease,background .16s ease}.collection-nav a:hover,.collection-nav a:focus{transform:translateY(-1px);background:#244b5b}.collection-nav a.active{background:#8cddd0;color:#102e3f;border-color:#8cddd0}.topic-card{border:1px solid #cfe0e5;border-radius:10px;padding:14px;margin:12px 0}.topic-card.active{border:2px solid var(--teal);background:#f5faf9}.topic-title-line{display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin:0 0 11px}.topic-title-line h3{margin:0;font-size:20px;line-height:1.25}.topic-count{display:inline-flex;align-items:center;padding:5px 10px;border-radius:18px;background:#dff4ef;color:#086b63;font-size:15px;font-weight:750;line-height:1.15;box-shadow:inset 0 0 0 1px rgba(0,121,111,.08)}.topic-status{font-size:12px;background:#e6f4f2;color:#126f67;padding:3px 9px;border-radius:16px}.danger{background:#b42318;border-color:#941d14;box-shadow:0 2px 5px rgba(180,35,24,.14)}.danger:hover,.danger:focus{background:#941d14;box-shadow:0 4px 9px rgba(180,35,24,.2)!important}.inline-actions{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.switch-topic{display:inline-flex;align-items:center;justify-content:center;padding:6px 11px;color:white;text-decoration:none;font-size:14px;font-weight:650;background:#176b67;border:1px solid #125c58;border-radius:8px;box-shadow:0 2px 5px rgba(16,74,76,.13);transition:background .16s ease,transform .16s ease,box-shadow .16s ease}.switch-topic:hover,.switch-topic:focus{color:white;background:#105b57;transform:translateY(-1px);box-shadow:0 4px 8px rgba(16,74,76,.17)}.topic-card .inline-actions form{display:inline-flex;margin:0;gap:0}.topic-card .inline-actions .danger{padding:6px 10px;font-size:14px;line-height:1.35}.document-delete-form{display:inline-flex;margin:8px 0 0}.document-delete-form .danger{padding:5px 9px;font-size:13px;line-height:1.3}.search-heading{align-items:center;flex-wrap:nowrap}.search-heading h2{margin:0}.search-mode-chip{display:inline-flex;align-items:center;justify-content:center;flex:0 0 auto;line-height:1.25;padding:6px 11px}.back-link{display:inline-flex;align-items:center;gap:6px;text-decoration:none;border:1px solid #8fbab7;border-radius:8px;padding:7px 12px;margin-bottom:14px;background:#f8fbfb;font-weight:650;box-shadow:0 1px 3px rgba(16,74,76,.08);transition:background .16s ease,transform .16s ease,box-shadow .16s ease}.back-link:hover,.back-link:focus{background:#eef6f5;transform:translateY(-1px);box-shadow:0 3px 7px rgba(16,74,76,.12)}.result-heading{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}.upload-dialog{width:min(650px,calc(100% - 24px));max-height:92vh;border:0;border-radius:13px;padding:0;color:var(--ink);box-shadow:0 18px 48px rgba(4,30,44,.27)}.upload-dialog::backdrop{background:rgba(9,35,48,.58)}.upload-dialog-body{padding:24px;overflow:auto;max-height:92vh}.dialog-heading{display:flex;justify-content:space-between;align-items:start;gap:14px}.dialog-heading h2{margin:0}.dialog-close-form{display:block}.dialog-close{width:36px;height:36px;padding:0;border-radius:50%;background:#edf4f6;color:var(--ink);border-color:#d3e0e5;font-size:22px;line-height:1;box-shadow:none}.dialog-close:hover,.dialog-close:focus{background:#dfeaec}.upload-option{margin-top:18px;padding:18px;background:#f7fbfb;border:1px solid #d4e5e5;border-radius:10px}.upload-option h3{margin:0 0 6px}.upload-form{margin-top:12px;align-items:stretch}.upload-form .query{flex-basis:100%}.upload-form button[type=submit]{width:100%}.suggestion{background:#eaf6ff;border-left:4px solid #1976a3;padding:12px 14px;border-radius:7px}.word-neighbor{display:grid;grid-template-columns:minmax(100px,1fr) 4fr 90px;gap:10px;align-items:center;margin:7px 0}.pca-section{margin-top:30px;padding-top:24px;border-top:1px solid #d7e2e5}.pca-heading{display:flex;align-items:flex-start;justify-content:space-between;gap:18px}.pca-heading h3{margin:0 0 4px}.pca-heading p{margin:0;max-width:850px}.pca-count{flex:0 0 auto;padding:5px 10px;border:1px solid #b8cdcf;border-radius:6px;color:#3f626b;background:#f7faf9;font-size:13px;font-weight:650}.pca-chart-wrap{position:relative;margin-top:16px;overflow:hidden;border:1px solid #d4e0e3;border-radius:9px;background:#fff}.pca-chart{display:block;width:100%;height:auto}.pca-plot-background{fill:#fbfdfd}.pca-gridline{stroke:#dce6e8;stroke-width:1}.pca-tick{fill:#667b85;font-size:12px}.pca-axis-label{fill:#344f5b;font-size:14px;font-weight:650}.pca-point{fill:#287d78;stroke:#fff;stroke-width:1.5;opacity:.72;cursor:crosshair;transition:opacity .12s ease,fill .12s ease}.pca-point:hover,.pca-point:focus,.pca-point.is-active{fill:#0d514e;opacity:1;outline:none}.pca-point-query{fill:#b26532;opacity:1;stroke:#713c1d;stroke-width:2}.pca-point-query:hover,.pca-point-query:focus,.pca-point-query.is-active{fill:#934b21}.pca-query-label{fill:#713c1d;font-size:14px;font-weight:750;paint-order:stroke;stroke:#fff;stroke-width:4px;stroke-linejoin:round}.pca-tooltip{position:absolute;z-index:3;pointer-events:none;max-width:240px;padding:9px 11px;border-radius:6px;background:#102e3f;color:#fff;font-size:13px;line-height:1.5;box-shadow:0 5px 14px rgba(10,38,51,.2);white-space:pre-line}.pca-caption{margin:9px 0 0;color:#607580;font-size:13px}
.zipf-chart-wrap{position:relative}.zipf-point{stroke:#fff;stroke-width:1;opacity:.18;cursor:crosshair;transition:opacity .1s ease}.zipf-point:hover,.zipf-point.is-active{opacity:1;stroke:#102e3f;stroke-width:1.5}.zipf-tooltip{position:absolute;z-index:3;pointer-events:none;max-width:250px;padding:8px 10px;border-radius:6px;background:#102e3f;color:#fff;font-size:13px;line-height:1.5;box-shadow:0 4px 12px rgba(10,38,51,.2);white-space:pre-line}
@media(max-width:1100px){.header-layout{grid-template-columns:1fr}.header-metrics{grid-template-columns:repeat(4,1fr)}}
@media(max-width:950px){.workspace{grid-template-columns:290px minmax(0,1fr)}}
@media(max-width:720px){.workspace{grid-template-columns:1fr}.sidebar .library-list{max-height:350px;overflow:auto}main{padding:18px 12px}header h1{font-size:32px}.eyebrow{font-size:14px}header p{font-size:17px}.header-metrics{grid-template-columns:repeat(2,1fr)}.header-metrics b{font-size:24px}.page-nav .upload-nav-button{margin-left:0}}
@media(max-width:900px){.chart-grid,.model-grid{grid-template-columns:1fr}.analysis-grid{grid-template-columns:repeat(2,1fr)}.training-metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.training-metrics .metric-box{grid-column:auto!important}}
@media(max-width:720px){.top-terms-grid{grid-template-columns:1fr}.analysis-actions{justify-content:flex-start}.compare-launch{width:100%}}
@media(max-width:520px){.training-metrics{grid-template-columns:1fr}}
'''


STYLE += '''
.source-form{align-items:flex-start}.source-choice{flex:0 0 220px;max-width:100%}.source-choice select{display:block;width:100%;margin-top:5px}.import-fields{border:0;padding:0;margin:0;min-width:0;flex:1 1 100%;display:flex;align-items:flex-start;gap:12px;flex-wrap:wrap}.import-fields[hidden],[data-ncbi-email][hidden]{display:none!important}.import-fields>p{flex-basis:100%;margin:0}.import-fields button{align-self:flex-start;margin-top:23px}.upload-form .import-fields button{margin-top:0}.import-fields label input{display:block;margin-top:5px}.source-choice:focus-within{color:#125f59}
'''

def site_shell(engine, content, active, title, upload_token=''):
    total_words = sum(document.counts['words'] for document in engine.docs.values())
    total_sentences = sum(document.counts['sentences'] for document in engine.docs.values())
    metrics = ''.join(
        f'<div class="metric"><small>{label}</small><b>{value:,}</b></div>'
        for label, value in [('文件數', len(engine.docs)),
                             ('摘要單字總數', total_words),
                             ('摘要句子總數', total_sentences),
                             ('全文索引詞種類', len(engine.index))]
    )
    upload_button = ('<button class="upload-nav-button" type="button" data-open-upload>'
                     'Add Articles</button>') if upload_token else ''
    navigation = '<nav class="page-nav" aria-label="主要導覽">' + ''.join(
        '<a href="'+url+'"'+(' aria-current="page"' if key == active else '')+'>'+label+'</a>'
        for url, label, key in [('/', 'Search', 'search'),
                                ('/manage', 'Manage Articles', 'manage'),
                                ('/word2vec', 'Word2Vec Models', 'word2vec'),
                                ('/analysis', 'Zipf Analysis', 'analysis')]
    ) + upload_button + '</nav>'
    next_path = {'search': '/', 'manage': '/manage', 'word2vec': '/word2vec',
                 'analysis': '/analysis'}.get(active, '/')
    summaries = getattr(engine, 'collection_summaries', [])
    if summaries:
        collection_navigation = ('<nav class="collection-nav" aria-label="主題資料集"><span>目前主題：</span>'+
            ''.join('<a class="'+('active' if item['id'] == getattr(engine, 'collection_id', '') else '')+
                    '" href="/collection-select?'+esc(urlencode({'id': item['id'], 'next': next_path}))+'">'+
                    esc(item['title'])+' · '+str(item['count'])+' docs</a>' for item in summaries)+
            '</nav>')
    else:
        collection_navigation = '<div class="collection-nav"><span>尚未建立主題資料集，請到「PMID查詢/上傳」建立。</span></div>'
    header = ('<header><div class="header-layout"><div class="header-copy">'
              '<div class="eyebrow">BIOMEDICAL INFORMATION RETRIEVAL</div>'
              '<h1><a class="home-link" href="/" title="回到主畫面">生醫文獻全文檢索</a></h1>'
              '</div><div class="header-metrics" aria-label="文件集統計">'+metrics+
              '</div></div>'+navigation+collection_navigation+'</header>')
    has_collection = bool(getattr(engine, 'collection_id', ''))
    upload_modal = ''
    if upload_token:
        upload_modal = ('<dialog class="upload-dialog" id="single-upload-dialog"><div class="upload-dialog-body">'
                        '<div class="dialog-heading"><div><h2>加入文章</h2>'
                        '<span class="chip">加入主題 · '+esc(getattr(engine, 'collection_title', '尚未建立主題'))+'</span></div>'
                        '<form class="dialog-close-form" method="dialog"><button class="dialog-close" value="cancel" '
                        'aria-label="關閉上傳視窗">×</button></form></div>'
                        '<p>先選擇 PubMed 或 arXiv，再貼上多筆對應 ID 下載摘要；也可上傳一篇 PMC 全文檔。</p>'+
                        ('' if has_collection else '<p class="warning">請先建立或選擇一個主題資料集。</p>')+
                        '<section class="upload-option"><h3>使用 Document IDs 批次加入</h3>'
                        '<p class="muted">支援空白、逗號或換行分隔，每次最多 200 篇；沒有摘要或已存在的文章會自動略過。</p>'
                        '<form class="upload-form source-form" action="/id-batch-import" method="post" '
                        'data-import-source data-mode="ids" data-wait="正在下載摘要並建立索引…">'
                        '<input type="hidden" name="token" value="'+esc(upload_token)+'">'
                        '<input type="hidden" name="collection_id" value="'+esc(getattr(engine, 'collection_id', ''))+'">'
                        '<label class="source-choice">Source<select name="source" required>'+source_options()+'</select></label>'
                        '<fieldset class="import-fields" hidden disabled><div class="query"><label for="batch-ids" data-id-label>Document IDs</label>'
                        '<textarea id="batch-ids" name="ids" maxlength="15000" required></textarea>'
                        '<small class="field-limit" data-id-limit>Maximum: 200 IDs per batch.</small></div>'
                        '<div class="query" data-ncbi-email>'
                        '<label for="batch-pmid-email">NCBI 聯絡信箱（選填）</label>'
                        '<input id="batch-pmid-email" name="email" type="email" maxlength="200" placeholder="你的學校信箱"></div>'
                        '<button type="submit" data-id-submit'+('' if has_collection else ' disabled')+'>Add Abstracts</button>'
                        '<p class="muted" data-source-help></p></fieldset></form></section>'
                        '<section class="upload-option"><h3>上傳單篇全文檔</h3>'
                        '<p class="muted">支援一個 PMC JATS XML 或 NXML，單檔最大 20 MB。</p>'
                        '<form class="upload-form" action="/upload" method="post" enctype="multipart/form-data" '
                        'data-wait="正在上傳並建立索引…"><input type="hidden" name="token" value="'+esc(upload_token)+'">'
                        '<input type="hidden" name="upload_mode" value="single">'
                        '<div class="query"><label for="single-upload-file">選擇單篇文章檔案</label>'
                        '<input id="single-upload-file" name="files" type="file" accept=".xml,.nxml" required></div>'
                        '<button type="submit"'+('' if has_collection else ' disabled')+'>Upload Article</button></form>'
                        '</section></div></dialog>')
    return ('<!doctype html><html lang="zh-Hant"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>'+esc(title)+' · Biomedical IR</title><style>'+STYLE+'</style>'+
            header+'<main>'+content+'</main>'+upload_modal+WAIT_SCRIPT+'</html>')


def zipf_svg(datasets, log_x=False, log_y=False, fitted=None):
    """Render a dependency-free interactive SVG for rank-frequency series.

    A dataset may contain either frequencies or ``(term, frequency)`` pairs.
    Supplying the pairs enables word-level hover details.
    """
    width, height = 900, 430
    left, right, top, bottom = 72, 24, 24, 58
    plot_width, plot_height = width-left-right, height-top-bottom
    usable = []
    for label, series, color in datasets:
        if not series:
            continue
        if isinstance(series[0], (tuple, list)) and len(series[0]) == 2:
            terms_for_series = [str(term) for term, _ in series]
            values = [frequency for _, frequency in series]
        else:
            values = list(series)
            terms_for_series = [''] * len(values)
        usable.append((label, values, color, terms_for_series))
    if not usable:
        return '<p class="muted">目前沒有可繪製的詞頻資料。</p>'
    maximum_rank = max(len(values) for _, values, _, _ in usable)
    maximum_frequency = max(values[0] for _, values, _, _ in usable)
    tx = (lambda value: math.log10(value)) if log_x else (lambda value: value)
    ty = (lambda value: math.log10(value)) if log_y else (lambda value: value)
    min_x, max_x = tx(1), tx(maximum_rank)
    min_y, max_y = ty(1), ty(maximum_frequency)
    if max_x == min_x:
        max_x += 1
    if max_y == min_y:
        max_y += 1
    xpixel = lambda value: left + (tx(value)-min_x)/(max_x-min_x)*plot_width
    ypixel = lambda value: top + plot_height - (ty(value)-min_y)/(max_y-min_y)*plot_height
    pieces = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Rank frequency chart">',
              f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#6d7f88"/>',
              f'<line x1="{left}" y1="{top+plot_height}" x2="{left+plot_width}" y2="{top+plot_height}" stroke="#6d7f88"/>']
    for index in range(6):
        fraction = index / 5
        x_transformed = min_x + fraction * (max_x-min_x)
        y_transformed = min_y + fraction * (max_y-min_y)
        x_value = 10 ** x_transformed if log_x else x_transformed
        y_value = 10 ** y_transformed if log_y else y_transformed
        x = left + fraction * plot_width
        y = top + plot_height - fraction * plot_height
        pieces += [f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_height}" stroke="#edf1f3"/>',
                   f'<text x="{x:.1f}" y="{height-29}" text-anchor="middle" font-size="12" fill="#526a79">{x_value:.0f}</text>',
                   f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_width}" y2="{y:.1f}" stroke="#edf1f3"/>',
                   f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="#526a79">{y_value:.0f}</text>']
    for label, values, color, terms_for_series in usable:
        maximum_points = 550
        sample_count = min(maximum_points, len(values))
        indices = ([0] if sample_count == 1 else
                   sorted(set(round(i*(len(values)-1)/(sample_count-1))
                              for i in range(sample_count))))
        points = ' '.join(f'{xpixel(i+1):.2f},{ypixel(values[i]):.2f}' for i in indices)
        pieces.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" vector-effect="non-scaling-stroke"/>')
        for i in indices:
            rank, frequency = i+1, values[i]
            term = terms_for_series[i]
            pieces.append(
                f'<circle class="zipf-point" cx="{xpixel(rank):.2f}" cy="{ypixel(frequency):.2f}" r="3" '
                f'fill="{color}" data-word="{esc(term)}" data-series="{esc(label)}" '
                f'data-rank="{rank}" data-frequency="{frequency}" '
                f'data-log-rank="{math.log10(rank):.4f}" data-log-frequency="{math.log10(frequency):.4f}" '
                f'data-log-chart="{str(bool(log_x or log_y)).lower()}">'
                f'<title>{esc(term)} · rank {rank} · frequency {frequency}</title></circle>')
    if fitted and len(usable) == 1:
        reg = fitted
        fit_points = []
        for rank in (1, maximum_rank):
            predicted_log = reg['intercept'] + reg['slope'] * math.log10(rank)
            predicted = 10 ** predicted_log
            fit_points.append(f'{xpixel(rank):.2f},{ypixel(max(1, predicted)):.2f}')
        pieces.append('<polyline points="'+' '.join(fit_points)+'" fill="none" stroke="#c24332" stroke-width="2" stroke-dasharray="8 6" vector-effect="non-scaling-stroke"/>')
    pieces += [f'<text x="{left+plot_width/2:.1f}" y="{height-7}" text-anchor="middle" font-size="14" fill="#122d3d">Rank'+('（log）' if log_x else '')+'</text>',
               f'<text x="18" y="{top+plot_height/2:.1f}" text-anchor="middle" font-size="14" fill="#122d3d" transform="rotate(-90 18 {top+plot_height/2:.1f})">Collection Frequency'+('（log）' if log_y else '')+'</text>']
    # Keep the legend away from the steep high-frequency head of the curve.
    legend_x = left + plot_width - 270
    for index, (label, _, color, _) in enumerate(usable):
        y = top + 18 + index*22
        pieces += [f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+26}" y2="{y}" stroke="{color}" stroke-width="3"/>',
                   f'<text x="{legend_x+34}" y="{y+4}" font-size="13" fill="#122d3d">{esc(label)}</text>']
    if fitted:
        y = top + 18 + len(usable)*22
        pieces += [f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+26}" y2="{y}" stroke="#c24332" stroke-width="2" stroke-dasharray="8 6"/>',
                   f'<text x="{legend_x+34}" y="{y+4}" font-size="13" fill="#122d3d">Linear regression</text>']
    pieces.append('</svg>')
    return ('<div class="zipf-chart-wrap">'+''.join(pieces)+
            '<div class="zipf-tooltip" role="status" hidden></div></div>')


def analysis_page(engine, params, upload_token=''):
    condition = params.get('condition', ['A'])[0]
    if condition not in CONDITIONS:
        condition = 'A'
    analyses = analyze_all(engine)
    selected = analyses[condition]
    regression = selected['regression']
    term_value = params.get('term', [''])[0][:100].strip()
    content = ('<section class="panel"><div class="row"><div><h2>Collection Vocabulary</h2>'
               '<span class="chip">主題 · '+esc(getattr(engine, 'collection_title', '目前資料集'))+'</span></div>'
               '<div class="analysis-actions"><a class="compare-launch" href="/compare?'+
               esc(urlencode({'a': getattr(engine, 'collection_id', ''), 'condition': condition}))+'">Compare Domains</a>'
               '<a href="/analysis.csv?condition='+condition+'">下載完整詞彙統計 CSV</a></div></div>'
               '<div class="condition-tabs">')
    for key, label in CONDITIONS.items():
        content += ('<a class="'+('active' if key == condition else '')+'" href="/analysis?'+
                    esc(urlencode({'condition': key, **({'term': term_value} if term_value else {})}))+'">'+esc(label)+'</a>')
    content += ('</div><div class="scroll"><table><tr><th>Condition</th><th>Tokens</th><th>Vocabulary</th>'
                '<th>Avg./doc</th><th>Slope</th><th>Exponent b</th><th>R²</th><th>RMSE</th></tr>')
    for key, label in CONDITIONS.items():
        result = analyses[key]
        reg = result['regression']
        content += (f'<tr><td>{esc(label)}</td><td>{result["total_tokens"]:,}</td>'
                    f'<td>{result["unique_terms"]:,}</td><td>{result["average_tokens"]:,.2f}</td>'
                    f'<td>{reg["slope"]:.4f}</td><td>{reg["exponent"]:.4f}</td>'
                    f'<td>{reg["r_squared"]:.4f}</td><td>{reg["rmse"]:.4f}</td></tr>')
    content += '</table></div></section>'

    colors = {'A': '#006d77', 'B': '#3a86a8', 'C': '#f28e2b', 'D': '#7b2cbf'}
    content += ('<section class="panel"><h2>Zipf Distribution · '+esc(CONDITIONS[condition])+'</h2>'
                '<div class="chart-grid"><div><h3>Experiment 1 · Rank vs Frequency</h3>'+
                zipf_svg([(CONDITIONS[condition], selected['ranked'], colors[condition])])+
                '<p class="muted">兩軸皆為線性刻度，呈現完整 vocabulary 的 rank-frequency curve。</p></div>'
                '<div><h3>Experiment 2 · Log-Log Plot</h3>'+
                zipf_svg([(CONDITIONS[condition], selected['ranked'], colors[condition])], True, True, regression)+
                '<p class="muted">虛線為 OLS：log₁₀(CF) = intercept + slope × log₁₀(rank)。</p></div></div>'
                '<div class="analysis-grid">'
                f'<div class="metric-box"><small>Slope</small><b>{regression["slope"]:.4f}</b></div>'
                f'<div class="metric-box"><small>Intercept</small><b>{regression["intercept"]:.4f}</b></div>'
                f'<div class="metric-box"><small>Zipf exponent b</small><b>{regression["exponent"]:.4f}</b></div>'
                f'<div class="metric-box"><small>R² / RMSE</small><b>{regression["r_squared"]:.4f} / {regression["rmse"]:.4f}</b></div>'
                '</div></section>')

    comparison_datasets = [(CONDITIONS[key], analyses[key]['ranked'], colors[key])
                           for key in CONDITIONS]
    content += ('<section class="panel"><h2>四種 Preprocessing 的 Log-Log 比較</h2>'+
                zipf_svg(comparison_datasets, True, True)+
                '<p class="muted">比較 vocabulary size、高頻詞、Zipf exponent 與曲線形狀時，請搭配上方數值表。</p></section>')

    content += ('<section class="panel"><div class="row"><div><h2>單一詞分布搜尋</h2>'
                '<p>查詢一個詞在各篇摘要中的 TF，以及整個 collection 的 CF、DF、IDF。送出後會進入獨立結果頁。</p></div></div>'
                '<form action="/term-result" method="get"><input type="hidden" name="condition" value="'+condition+'">'
                '<div class="query"><label for="term">輸入一個英文單詞</label><input id="term" name="term" value="'+esc(term_value)+'" '
                'placeholder="例如 cancer" maxlength="100" required></div><button>Analyze Distribution</button></form></section>')

    content += ('<section class="panel"><h2>Top 50 Collection Frequency 與 Document Frequency</h2>'
                '<p class="formula">CF(t) = collection 內總出現次數；DF(t) = 包含該詞的文件數；IDF(t) = ln(N / DF(t))。</p>'
                '<div class="scroll"><table><tr><th>Rank</th><th>Term</th><th>CF</th><th>DF</th><th>IDF</th></tr>')
    for rank, (term, frequency) in enumerate(selected['ranked'][:50], 1):
        document_frequency = selected['df'][term]
        content += (f'<tr><td>{rank}</td><td><a href="/term-result?{esc(urlencode({"condition": condition, "term": term}))}">{esc(term)}</a></td>'
                    f'<td>{frequency:,}</td><td>{document_frequency:,}</td><td>{idf(selected["documents"], document_frequency):.4f}</td></tr>')
    content += ('</table></div><p>同一個詞若集中在少數文章中重複很多次，會出現高 CF、較低 DF；DF 更能表示詞是否廣泛分布。'
                '非常常見且遍布文件的詞具有較低 IDF，因此在 TF-IDF 中辨識力較低。</p></section>')

    content += ('<footer>Project #2 分析頁 · 所有數值均由目前載入的 abstract 即時計算。'
                '<br>最終報告仍需依你的實驗結果回答 RQ1–RQ5，並另外撰寫 300–500 words IR implications 與一頁 Executive Summary。</footer>')
    return site_shell(engine, content, 'analysis', 'Zipf／詞彙分布', upload_token)


def compare_domains_page(store, active_engine, params, upload_token=''):
    """Compare Zipf statistics for any two independently stored topics."""
    summaries = store.summaries()
    condition = params.get('condition', ['A'])[0]
    if condition not in CONDITIONS:
        condition = 'A'
    available_ids = [item['id'] for item in summaries]
    active_id = getattr(active_engine, 'collection_id', '')
    domain_a = params.get('a', [active_id])[0]
    if domain_a not in available_ids:
        domain_a = available_ids[0] if available_ids else ''
    other_ids = [collection_id for collection_id in available_ids if collection_id != domain_a]
    domain_b = params.get('b', [other_ids[0] if other_ids else ''])[0]
    if domain_b not in available_ids:
        domain_b = other_ids[0] if other_ids else ''

    def topic_options(selected_id):
        return ''.join(
            '<option value="'+esc(item['id'])+'"'+(' selected' if item['id'] == selected_id else '')+'>'+
            esc(item['title'])+' · '+f'{item["count"]:,}'+' documents</option>'
            for item in summaries
        )

    condition_options = ''.join(
        '<option value="'+key+'"'+(' selected' if key == condition else '')+'>'+esc(label)+'</option>'
        for key, label in CONDITIONS.items()
    )
    content = ('<a class="back-link" href="/analysis?'+esc(urlencode({'condition': condition}))+'">← Back to Zipf Analysis</a>'
               '<section class="panel comparison-hero"><span class="chip">OPTIONAL CHALLENGE</span>'
               '<h2>Compare Two Domains</h2><p>Choose any two saved topics and compare their abstract collections under the same preprocessing condition.</p>'
               '<form class="comparison-form" action="/compare" method="get">'
               '<div><label for="domain-a">Domain A</label><select id="domain-a" name="a" required>'+topic_options(domain_a)+'</select></div>'
               '<div><label for="domain-b">Domain B</label><select id="domain-b" name="b" required>'+topic_options(domain_b)+'</select></div>'
               '<div><label for="compare-condition">Preprocessing</label><select id="compare-condition" name="condition">'+condition_options+'</select></div>'
               '<button type="submit">Run Domain Comparison</button></form></section>')

    if len(summaries) < 2:
        content += ('<section class="panel"><p class="warning">至少需要兩個不同主題才能比較。'
                    '請先到「PMID查詢／上傳」建立第二個主題。</p></section>')
        return site_shell(active_engine, content, 'analysis', 'Compare Domains', upload_token)
    if domain_a == domain_b:
        content += '<section class="panel"><p class="warning">Domain A 與 Domain B 必須選擇不同主題。</p></section>'
        return site_shell(active_engine, content, 'analysis', 'Compare Domains', upload_token)

    engine_a = store.engine(domain_a)
    engine_b = store.engine(domain_b)
    result_a = analyze_all(engine_a)[condition]
    result_b = analyze_all(engine_b)[condition]
    regression_a = result_a['regression']
    regression_b = result_b['regression']
    content += ('<section class="panel"><div class="row"><div><h2>Comparison Results</h2>'
                '<span class="chip">'+esc(CONDITIONS[condition])+'</span></div>'
                '<span class="muted">Same preprocessing applied to both domains</span></div>'
                '<div class="scroll"><table><tr><th>Measure</th>'
                '<th class="domain-head-a">Domain A · '+esc(engine_a.collection_title)+'</th>'
                '<th class="domain-head-b">Domain B · '+esc(engine_b.collection_title)+'</th></tr>'
                f'<tr><td>Documents</td><td>{result_a["documents"]:,}</td><td>{result_b["documents"]:,}</td></tr>'
                f'<tr><td>Total tokens</td><td>{result_a["total_tokens"]:,}</td><td>{result_b["total_tokens"]:,}</td></tr>'
                f'<tr><td>Vocabulary</td><td>{result_a["unique_terms"]:,}</td><td>{result_b["unique_terms"]:,}</td></tr>'
                f'<tr><td>Avg. tokens / document</td><td>{result_a["average_tokens"]:,.2f}</td><td>{result_b["average_tokens"]:,.2f}</td></tr>'
                f'<tr><td>Zipf exponent b</td><td>{regression_a["exponent"]:.4f}</td><td>{regression_b["exponent"]:.4f}</td></tr>'
                f'<tr><td>R²</td><td>{regression_a["r_squared"]:.4f}</td><td>{regression_b["r_squared"]:.4f}</td></tr>'
                f'<tr><td>RMSE</td><td>{regression_a["rmse"]:.4f}</td><td>{regression_b["rmse"]:.4f}</td></tr>'
                '</table></div>')
    if result_a['documents'] != result_b['documents']:
        content += ('<p class="warning"><b>Sample-size note:</b> The two domains contain different numbers of documents. '
                    'Interpret raw token and vocabulary counts with care.</p>')
    content += '</section>'

    content += ('<section class="panel"><h2>Zipf Log-Log Overlay</h2>'+
                zipf_svg([(engine_a.collection_title, result_a['ranked'], '#087fa4'),
                          (engine_b.collection_title, result_b['ranked'], '#9654ae')], True, True)+
                '<p class="muted">Both curves use the same axes and preprocessing condition, making their shapes directly comparable.</p></section>')

    def top_terms_table(result, title, heading_class):
        rows = ''.join(
            f'<tr><td>{rank}</td><td>{esc(term)}</td><td>{frequency:,}</td></tr>'
            for rank, (term, frequency) in enumerate(result['ranked'][:10], 1)
        )
        return ('<div><h3 class="'+heading_class+'">'+esc(title)+'</h3>'
                '<div class="scroll"><table><tr><th>Rank</th><th>Term</th><th>CF</th></tr>'+rows+'</table></div></div>')

    content += ('<section class="panel"><h2>Top 10 Terms</h2><div class="top-terms-grid">'+
                top_terms_table(result_a, 'Domain A · '+engine_a.collection_title, 'domain-head-a')+
                top_terms_table(result_b, 'Domain B · '+engine_b.collection_title, 'domain-head-b')+
                '</div></section>'
                '<section class="panel"><h2>Interpretation Prompt</h2>'
                '<p class="comparison-note"><b>Does domain specificity affect the Zipf distribution?</b> '
                'Use the exponent, R², vocabulary size, Top 10 terms, and curve shape above to support your answer. '
                'Differences in corpus size should be reported as an experimental limitation.</p></section>'
                '<footer>Domain comparison uses abstract text only. Each topic remains independently stored and indexed.</footer>')
    return site_shell(active_engine, content, 'analysis', 'Compare Domains', upload_token)


def term_result_page(engine, params, upload_token=''):
    """Show one term's distribution without repeating the full Zipf dashboard."""
    condition = params.get('condition', ['A'])[0]
    if condition not in CONDITIONS:
        condition = 'A'
    term_value = params.get('term', [''])[0][:100].strip()
    selected = analyze_all(engine)[condition]
    back_url = '/analysis?' + urlencode({'condition': condition})
    content = ('<a class="back-link" href="'+esc(back_url)+'">← Back to Zipf Analysis</a>'
               '<section class="panel"><div class="result-heading"><div><h2>單一詞分布結果</h2>'
               '<span class="chip">主題 · '+esc(getattr(engine, 'collection_title', '目前資料集'))+'</span> '
               '<span class="chip">'+esc(CONDITIONS[condition])+'</span></div></div>'
               '<p>可直接輸入另一個英文單詞，同一頁會重新載入新的分布結果。</p>'
               '<form action="/term-result" method="get"><input type="hidden" name="condition" value="'+condition+'">'
               '<div class="query"><label for="term-result-query">輸入一個英文單詞</label>'
               '<input id="term-result-query" name="term" value="'+esc(term_value)+'" '
               'placeholder="例如 receptor" maxlength="100" required autofocus></div>'
               '<button>Analyze Distribution</button></form>')

    processed_term, error = query_term(term_value, condition)
    if error:
        content += '<p class="warning">'+esc(error)+'</p></section>'
        return site_shell(engine, content, 'analysis', '單一詞分布結果', upload_token)

    collection_frequency = selected['cf'][processed_term]
    document_frequency = selected['df'][processed_term]
    inverse_document_frequency = idf(selected['documents'], document_frequency)
    rows = []
    for document in engine.docs.values():
        frequency = selected['document_tf'][document.id][processed_term]
        if frequency:
            rows.append((frequency, engine.numbers[document.id], document))
    rows.sort(key=lambda row: (-row[0], row[1]))
    maximum_tf = rows[0][0] if rows else 0
    content += ('<div class="analysis-grid">'
                f'<div class="metric-box"><small>處理後 term</small><b>{esc(processed_term)}</b></div>'
                f'<div class="metric-box"><small>CF</small><b>{collection_frequency:,}</b></div>'
                f'<div class="metric-box"><small>DF</small><b>{document_frequency:,} / {selected["documents"]:,}</b></div>'
                f'<div class="metric-box"><small>IDF = ln(N/DF)</small><b>{inverse_document_frequency:.4f}</b></div></div>')
    if rows:
        content += ('<p><a href="/term-distribution.csv?'+esc(urlencode({'condition': condition, 'term': term_value}))+'">'
                    '下載此詞的文章分布 CSV</a></p>'
                    '<div class="scroll"><table><tr><th>文章</th><th>Document ID</th><th>標題</th><th>TF</th><th>相對頻率</th></tr>')
        for frequency, number, document in rows[:200]:
            link = '/?'+urlencode({'q': term_value, 'mode': 'AND', 'doc': document.id})
            width = 100 * frequency / maximum_tf if maximum_tf else 0
            content += (f'<tr><td>{number:03d}</td><td>{esc(document.id)}</td><td><a href="{esc(link)}">{esc(document.title)}</a></td>'
                        f'<td>{frequency:,}</td><td><div class="distribution-bar"><span style="width:{width:.2f}%"></span></div></td></tr>')
        content += '</table></div>'
        if len(rows) > 200:
            content += '<p class="muted">畫面顯示前 200 篇；CSV 包含全部命中文章。</p>'
    else:
        content += '<p class="warning">此詞沒有出現在目前的摘要 collection。可在上方直接改查其他單詞。</p>'
    content += '</section><footer>單一詞分布只使用目前主題的英文摘要計算。</footer>'
    return site_shell(engine, content, 'analysis', '單一詞分布結果', upload_token)


def pca_chart_html(model, query_word):
    """Render a dependency-free, interactive SVG for a Word2Vec PCA view."""
    projection = pca_projection(model, query_word)
    points = projection['points']
    if not points:
        return ''

    width, height = 960, 520
    left, right, top, bottom = 72, 28, 28, 58
    plot_width, plot_height = width-left-right, height-top-bottom
    xs = [point['x'] for point in points]
    ys = [point['y'] for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_padding = max((x_max-x_min)*0.08, 0.01)
    y_padding = max((y_max-y_min)*0.08, 0.01)
    x_min, x_max = x_min-x_padding, x_max+x_padding
    y_min, y_max = y_min-y_padding, y_max+y_padding

    def screen_x(value):
        return left+(value-x_min)/(x_max-x_min)*plot_width

    def screen_y(value):
        return top+(y_max-value)/(y_max-y_min)*plot_height

    svg = [f'<svg class="pca-chart" viewBox="0 0 {width} {height}" role="img" '
           'aria-labelledby="pca-title pca-description">',
           '<title id="pca-title">PCA projection of trained word embeddings</title>',
           '<desc id="pca-description">Hover or focus a point to inspect its word, principal component coordinates, frequency, and cosine similarity.</desc>',
           f'<rect class="pca-plot-background" x="{left}" y="{top}" width="{plot_width}" height="{plot_height}"/>']
    for step in range(6):
        x_value = x_min+(x_max-x_min)*step/5
        x = screen_x(x_value)
        y_value = y_min+(y_max-y_min)*step/5
        y = screen_y(y_value)
        svg.append(f'<line class="pca-gridline" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+plot_height}"/>')
        svg.append(f'<text class="pca-tick" x="{x:.2f}" y="{height-30}" text-anchor="middle">{x_value:.2f}</text>')
        svg.append(f'<line class="pca-gridline" x1="{left}" y1="{y:.2f}" x2="{left+plot_width}" y2="{y:.2f}"/>')
        svg.append(f'<text class="pca-tick" x="{left-12}" y="{y+4:.2f}" text-anchor="end">{y_value:.2f}</text>')
    svg.extend([
        f'<text class="pca-axis-label" x="{left+plot_width/2:.2f}" y="{height-8}" text-anchor="middle">PC1</text>',
        f'<text class="pca-axis-label" transform="translate(18 {top+plot_height/2:.2f}) rotate(-90)" text-anchor="middle">PC2</text>',
    ])
    for point in points:
        x, y = screen_x(point['x']), screen_y(point['y'])
        frequency = point['frequency']
        radius = 7.0 if point['is_query'] else min(6.0, 3.3+math.log1p(frequency)*0.35)
        similarity = '' if point['similarity'] is None else f'{point["similarity"]:.4f}'
        classes = 'pca-point pca-point-query' if point['is_query'] else 'pca-point'
        svg.append(
            f'<circle class="{classes}" cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" '
            f'data-radius="{radius:.2f}" data-word="{esc(point["word"])}" '
            f'data-pc1="{point["x"]:.4f}" data-pc2="{point["y"]:.4f}" '
            f'data-similarity="{similarity}" data-frequency="{frequency}" tabindex="0">'
            f'<title>{esc(point["word"])}</title></circle>')
        if point['is_query']:
            svg.append(f'<text class="pca-query-label" x="{x+11:.2f}" y="{y-11:.2f}">{esc(point["word"])}</text>')
    svg.append('</svg>')
    variance_1, variance_2 = projection['explained_variance']
    return ('<section class="pca-section"><div class="pca-heading"><div><h3>PCA Projection of Word Embeddings</h3>'
            '<p class="muted">將高維詞向量降成二維。靠得較近表示模型學到的使用情境較相似；PCA 位置不等同因果或分類結果。</p></div>'
            f'<span class="pca-count">{len(points)} terms</span></div>'
            '<div class="pca-chart-wrap">'+''.join(svg)+'<div class="pca-tooltip" role="status" hidden></div></div>'
            f'<p class="pca-caption">Explained variance: PC1 {variance_1*100:.1f}% · PC2 {variance_2*100:.1f}% · '
            'Point size reflects corpus frequency. Hover or focus any point for details.</p></section>')


def word2vec_page(engine, params, upload_token=''):
    model_key = params.get('model', ['skipgram'])[0]
    if model_key not in MODEL_NAMES:
        model_key = 'skipgram'
    term = params.get('term', [''])[0][:80].strip().casefold()
    errors = []

    def integer_parameter(name, default, minimum, maximum, label):
        try:
            value = int(params.get(name, [str(default)])[0])
        except ValueError:
            errors.append(f'{label} 必須是整數。')
            return default
        if not minimum <= value <= maximum:
            errors.append(f'{label} 必須介於 {minimum}～{maximum}。')
            return default
        return value

    window = integer_parameter('window', 2, 1, 5, 'Window size')
    dimension = integer_parameter('dimension', 24, 8, 64, 'Vector dimension')
    epochs = integer_parameter('epochs', 2, 1, 5, 'Epochs')
    model_options = ''.join(
        '<option value="'+key+'"'+(' selected' if key == model_key else '')+'>'+label+'</option>'
        for key, label in MODEL_NAMES.items()
    )
    content = ('<section class="panel"><div class="row"><div><h2>Word2Vec 模型訓練</h2>'
               '<span class="chip">主題 · '+esc(getattr(engine, 'collection_title', '目前資料集'))+'</span></div></div>'
               '<p>兩個模型都只使用目前主題的英文摘要，小寫斷詞、移除 stopwords，並使用 Negative Sampling。'
               '更換主題或參數後會建立另一組模型結果。</p>'
               '<div class="model-grid"><div class="metric-box"><small>CBOW</small><b>Context → Center</b>'
               '<p>使用周圍詞預測中間詞，速度較快，常見詞通常較穩定。</p></div>'
               '<div class="metric-box"><small>Skip-gram</small><b>Center → Context</b>'
               '<p>使用中間詞預測周圍詞，較能觀察較少見詞，但訓練較慢。</p></div></div>'
               '<form class="limit-form" action="/word2vec" method="get" data-wait="正在訓練目前主題的 Word2Vec 模型…">'
               '<input type="hidden" name="train" value="1">'
               '<div><label for="model">訓練模型</label><select id="model" name="model">'+model_options+'</select></div>'
               '<div><label for="window">Window size</label><input id="window" name="window" type="number" value="'+str(window)+'" min="1" max="5" required><small class="field-limit">Maximum: 5</small></div>'
               '<div><label for="dimension">Vector dimension</label><input id="dimension" name="dimension" type="number" value="'+str(dimension)+'" min="8" max="64" required><small class="field-limit">Maximum: 64</small></div>'
               '<div><label for="epochs">Epochs</label><input id="epochs" name="epochs" type="number" value="'+str(epochs)+'" min="1" max="5" required><small class="field-limit">Maximum: 5</small></div>'
               '<div class="query"><label for="word2vec-term">查詢相似詞</label><input id="word2vec-term" name="term" value="'+esc(term)+'" placeholder="例如 receptor" maxlength="80" required></div>'
               '<button>Train &amp; Find Similar Words</button></form>'
               '<p class="muted">Recommended defaults: window 2, dimension 24, epochs 2. Higher values increase training time. Training cap: 60,000 examples per run.</p></section>')
    if errors:
        content += '<p class="warning">'+' '.join(esc(error) for error in errors)+'</p>'
    if params.get('train', [''])[0] == '1' and not errors:
        if not engine.docs:
            content += '<p class="warning">目前主題沒有文章，請先匯入 PubMed 摘要。</p>'
        else:
            started = time.perf_counter()
            model = train_word2vec(engine, model_key, dimension, window, epochs)
            elapsed = time.perf_counter() - started
            content += (f'<section class="panel"><h2>訓練結果 · {esc(model["model_name"])}</h2>'
                        f'<div class="training-metrics"><div class="metric-box"><small>Model</small><b>{esc(model["model_name"])}</b></div>'
                        f'<div class="metric-box"><small>Vocabulary</small><b>{model["vocabulary_size"]:,}</b></div>'
                        f'<div class="metric-box"><small>Training examples</small><b>{model["examples"]:,}</b></div>'
                        f'<div class="metric-box"><small>Time</small><b>{elapsed:.2f} s</b></div>'
                        f'<div class="metric-box"><small>Window</small><b>{model["window"]}</b></div>'
                        f'<div class="metric-box"><small>Dimension</small><b>{model["dimension"]}</b></div>'
                        f'<div class="metric-box"><small>Epochs</small><b>{model["epochs"]}</b></div></div>')
            neighbors = nearest_words(model, term)
            if neighbors:
                content += '<h3>「'+esc(term)+'」的相似詞（Cosine similarity）</h3>'
                for similarity, word in neighbors:
                    width = max(0.0, min(100.0, (similarity + 1.0) * 50.0))
                    content += (f'<div class="word-neighbor"><b>{esc(word)}</b><div class="bar">'
                                f'<span style="width:{width:.2f}%"></span></div><span>{similarity:.4f}</span></div>')
                content += pca_chart_html(model, term)
            else:
                correction, _ = correct_query(term, engine)
                extra = (' 建議改查「'+esc(correction)+'」。' if correction else '')
                content += '<p class="warning">這個詞不在目前模型的 vocabulary 中。'+extra+'</p>'
            content += '</section>'
    content += ('<footer>Word2Vec 分析只使用目前主題的摘要。CBOW 與 Skip-gram 使用相同 preprocessing，'
                '因此可在相同參數下比較兩種模型的相似詞結果。</footer>')
    return site_shell(engine, content, 'word2vec', 'Word2Vec模型', upload_token)

def page(engine, params, upload_token="", notice="", pmc_html="", management=False):
    number = lambda doc: f"{engine.numbers[doc.id]:03d}"
    query = params.get('q', [''])[0][:500]
    selected = params.get('doc', [''])[0]
    selected_doc = engine.docs.get(selected)
    mode = params.get('mode', ['AND'])[0]
    if mode not in {'AND','OR','PHRASE'}:
        mode = 'AND'
    started = time.perf_counter()
    results, active_terms = engine.search(query, mode, selected if selected_doc else None)
    elapsed = (time.perf_counter()-started)*1000
    options = ''.join(f'<option value="{v}" {"selected" if mode==v else ""}>{label}</option>' for v,label in [('AND','AND · 所有關鍵字'),('OR','OR · 任一關鍵字'),('PHRASE','PHRASE · 連續詞組')])
    scope_input = f'<input type="hidden" name="doc" value="{esc(selected)}">' if selected_doc else ''
    scope_title = f'只搜尋文章 {number(selected_doc)}' if selected_doc else '搜尋文章'
    scope_note = (f'目前只比對「{esc(selected_doc.title)}」。<a href="/">回到全部文章搜尋</a>'
                  if selected_doc else '目前會搜尋文件集中的全部文章。可到「PMID查詢/上傳」點選已有文章，改為單篇搜尋。')
    content = f'''<section class="panel search-panel"><div class="row search-heading"><h2>{scope_title}</h2>{'<span class="chip search-mode-chip">單篇模式</span>' if selected_doc else '<span class="chip search-mode-chip">全部文章模式</span>'}</div>
<form action="/" method="get">{scope_input}<div class="query"><label for="q">輸入英文關鍵字</label><input id="q" name="q" value="{esc(query)}" placeholder="例如 cancer 或 gene expression" maxlength="500"></div><div><label for="mode">比對方式</label><select id="mode" name="mode">{options}</select></div><button>Search</button></form>
<p class="muted">{scope_note}<br>檢索摘要與正文，不檢索標題。AND / OR 忽略常見停用詞；PHRASE 保留停用詞，依連續詞幹順序比對。所有模式皆使用 Porter stemming。大小寫不影響結果。</p></section>'''
    corrected_query, corrections = correct_query(query, engine) if query.strip() else ('', [])
    if corrected_query:
        suggestion_params = {'q': corrected_query, 'mode': mode}
        if selected_doc:
            suggestion_params['doc'] = selected_doc.id
        changes = '、'.join(f'{before} → {after}' for before, after in corrections)
        content += ('<p class="suggestion"><b>拼字修正（Dynamic Programming Edit Distance）：</b>'
                    +esc(changes)+'。你是不是要找 <a href="/?'+esc(urlencode(suggestion_params))+'">'
                    +esc(corrected_query)+'</a>？</p>')
    if notice:
        content += '<section class="panel" role="status"><h2>操作結果</h2><pre>'+esc(notice)+'</pre></section>'
    main_content = content
    content = ''
    summaries = getattr(engine, 'collection_summaries', [])
    active_id = getattr(engine, 'collection_id', '')
    content += '<section class="panel"><h2>主題資料集</h2><p>每個主題獨立保存文章與分析結果，不會混合計算 Zipf、CF／DF／IDF 或 Word2Vec。</p>'
    if summaries:
        for item in summaries:
            is_active = item['id'] == active_id
            content += ('<div class="topic-card '+('active' if is_active else '')+'"><div class="row"><div>'
                        '<div class="topic-title-line"><h3>'+esc(item['title'])+'</h3>'
                        f'<span class="topic-count">{item["count"]:,} 篇</span>'+
                        ('<span class="topic-status">目前使用中</span>' if is_active else '')+'</div>'
                        '<div class="inline-actions"><a class="switch-topic" href="/collection-select?'+esc(urlencode({'id': item['id'], 'next': '/manage'}))+'">Switch Topic</a>'
                        '<form action="/collection-delete" method="post" data-confirm="確定刪除主題「'+esc(item['title'])+'」及其中全部 '+str(item['count'])+' 篇文章嗎？此操作無法復原。">'
                        '<input type="hidden" name="token" value="'+esc(upload_token)+'"><input type="hidden" name="collection_id" value="'+esc(item['id'])+'">'
                        '<button class="danger" type="submit">Delete Topic</button></form></div></div>'
                        '<details><summary>新增文章到「'+esc(item['title'])+'」</summary>'
                        +bulk_import_form(upload_token, item)+'</details></div></div>')
    else:
        content += '<p class="warning">目前沒有主題，請使用下方表單建立第一個主題。</p>'
    content += '</section>'
    content += ('<section class="panel"><h2>建立新主題並匯入摘要</h2>'
                '<p class="muted">先選擇來源，再填寫主題、搜尋條件與篇數。大量匯入不需要自行輸入 ID。</p>'
                +bulk_import_form(upload_token)+'</section>')
    content += '<section class="panel"><h2>上傳 PubMed collection</h2><form action="/upload" method="post" enctype="multipart/form-data"><input type="hidden" name="token" value="'+esc(upload_token)+'"><input type="hidden" name="upload_mode" value="collection"><div class="query"><label for="files">選擇含多篇摘要的 PubMed XML 或 pubmed.json（可多選）</label><input id="files" name="files" type="file" accept=".xml,.json" multiple required></div><button>Upload Collection</button></form><p class="muted">單篇 PMC JATS／NXML 文章或批次 PMID 請使用頁面上方的「Add Articles」按鈕。此處支援含多篇摘要的 PubMed XML，以及 Project #2 程式產生的 pubmed.json。每檔最多 20 MB，每次最多 10 個檔案、合計最多 40 MB。</p></section>'
    import_panels = content
    content = ''
    content += '<details class="panel"><summary>AND 和 OR 差在哪？</summary><p>搜尋 cancer treatment：AND 要在同篇文章的摘要＋正文同時包含兩詞，不必相鄰；OR 包含任一詞即可。要找連續詞組請選 PHRASE。</p><p>文章刪除後會重新連續編號；搜尋排名依 BM25 分數排列。</p></details>'
    sidebar = content
    content = ''
    if engine.errors:
        content += '<details class="panel"><summary>讀取提示（'+str(len(engine.errors))+'）</summary><ul>'+''.join('<li>'+esc(e)+'</li>' for e in engine.errors)+'</ul></details>'
    if not engine.docs:
        content += '<p class="warning">尚無可用 XML。請前往 <a href="/manage">PMID查詢/上傳</a> 加入文章。</p>'
    diagnostics = content
    content = main_content + diagnostics
    if selected:
        doc = selected_doc
        if doc:
            content += f'<article class="panel"><span class="chip">文件詳情 · {esc(doc.id)}</span><h2>{number(doc)}. {esc(doc.title)}</h2><p>{esc(doc.journal)}</p>'
            content += '<div class="scroll"><table><tr><th>摘要統計</th><th>數量</th></tr>'
            for k,label in [('words','Words'),('sentences','Sentences'),('characters','Characters（含空白）')]:
                content += f'<tr><td>{label}</td><td>{doc.counts[k]:,}</td></tr>'
            content += '</table></div><h3>摘要</h3><pre>'+highlight(doc.abstract or '此文章無摘要。',query)+'</pre>'
            if re.fullmatch(r'PMC[1-9]\d*', doc.id):
                content += '<p><a href="https://pmc.ncbi.nlm.nih.gov/articles/'+esc(doc.id)+'/" target="_blank" rel="noopener">前往 PMC 查看原文 ↗</a></p>'
            elif re.fullmatch(r'PMID[1-9]\d*', doc.id):
                content += '<p><a href="https://pubmed.ncbi.nlm.nih.gov/'+esc(doc.id[4:])+'/" target="_blank" rel="noopener">前往 PubMed 查看文獻 ↗</a></p>'
            elif doc.id.startswith('ARXIV:'):
                content += '<p><a href="https://arxiv.org/abs/'+esc(doc.id[6:])+'" target="_blank" rel="noopener">前往 arXiv 查看文獻 ↗</a></p>'
            content += document_delete_form(doc, upload_token)+'</article>'
        else:
            content += '<p class="warning">找不到指定文件。</p>'
    if query.strip():
        result_scope = f'文章 {number(selected_doc)}' if selected_doc else '全部文章'
        content += f'<div class="row"><h2>搜尋結果 · {len(results)} 篇</h2><span class="muted">範圍：{result_scope} · {elapsed:.3f} ms · BM25 排序</span></div>'
        raw_terms = tokens(query)
        unique_terms = list(dict.fromkeys(stem(t) for t in raw_terms))
        content += '<section class="panel"><h3>查詢分析與總數</h3><p>斷詞結果（'+str(len(raw_terms))+' 個詞）：'+(' '.join('<span class="chip">'+esc(t)+'</span>' for t in raw_terms) or '無')+'</p>'
        searched_total = 1 if selected_doc else len(engine.docs)
        content += f'<p><b>命中文章總數：{len(results)} 篇</b> ／ 本次搜尋範圍：{searched_total} 篇 ／ 本地文件總數：{len(engine.docs)} 篇</p>'
        content += '<p class="stats">'+ ' · '.join(f'{label}：{sum(d.counts[key] for d,_ in results):,}' for key,label in [('characters','命中文件摘要字元總數'),('words','摘要單字總數'),('sentences','摘要句子總數')]) + '</p>'
        content += '<div class="scroll"><table><tr><th>搜尋詞</th><th>比對處理</th><th>全文含此詞篇數</th><th>摘要次數</th><th>正文次數</th><th>全文總次數</th></tr>'
        for term in unique_terms:
            frequencies = {d.id: d.full_stem_tf[term] for d in engine.docs.values()}
            status = '參與詞組比對' if mode == 'PHRASE' else ('有效搜尋詞' if term in active_terms else '停用詞，忽略')
            abstract_total = sum(d.abstract_stem_tf[term] for d,_ in results)
            body_total = sum(d.body_stem_tf[term] for d,_ in results)
            full_total = sum(d.full_stem_tf[term] for d,_ in results)
            content += f'<tr><td>{esc(term)}</td><td>{status}</td><td>{sum(v>0 for v in frequencies.values())}</td><td>{abstract_total:,}</td><td>{body_total:,}</td><td>{full_total:,}</td></tr>'
        content += '</table></div>'
        if len(raw_terms) == 1:
            corpus = analyze_all(engine)['D']
            corpus_term = stem(raw_terms[0])
            corpus_df = corpus['df'][corpus_term]
            content += (f'<p><b>此詞在摘要 collection 的分布：</b>CF {corpus["cf"][corpus_term]:,} · '
                        f'DF {corpus_df:,}/{corpus["documents"]:,} · IDF {idf(corpus["documents"], corpus_df):.4f} · '
                        '<a href="/term-result?'+esc(urlencode({'condition': 'D', 'term': query}))+'">查看各文章 TF 與 Zipf 分析</a></p>')
        content += '<p class="muted">搜尋與全文總次數涵蓋摘要＋正文，不包含標題。以空白與一般標點斷開英文詞，保留 HIV-1 這類連字號詞；出現次數按 Porter 詞幹合併（testing、tested → test），不是詞組次數。Words、Sentences、Characters 仍只統計摘要。</p></section>'
        content += '<p class="muted">有效索引詞：'+esc(', '.join(active_terms) or '無（PHRASE 仍可比對停用詞詞組）')+'</p>'
        if results:
            export_params = {'q':query,'mode':mode}
            if selected_doc:
                export_params['doc'] = selected
            content += f'<p><a href="/export.csv?{esc(urlencode(export_params))}">下載這次搜尋結果 CSV</a></p>'
        else:
            content += '<p class="panel">沒有符合的文件。可嘗試較少關鍵字、改用 OR，或查看下方文件的用詞。僅含停用詞的 AND / OR 查詢不會回傳結果。</p>'
        max_score = results[0][1] if results else 0
        for rank,(doc,score) in enumerate(results[:100],1):
            link = '/?'+urlencode({'q':query,'mode':mode,'doc':doc.id})
            content += f'<article class="card"><div class="row"><span class="chip">搜尋排名 {rank} · 文章 {number(doc)} · {esc(doc.id)}</span><small>BM25 {score:.4f}</small></div><h3><a href="{esc(link)}">{number(doc)}. {esc(doc.title)}</a></h3>{document_links(doc)}<p>{snippet(doc,query)}</p>{term_counts_html(doc,query)}<div class="stats"><span>摘要 {doc.counts["words"]:,} words</span><span>{doc.counts["sentences"]:,} sentences</span><span>{doc.counts["characters"]:,} characters</span></div><div class="bar" aria-label="相對最高分"><span style="width:{100*score/max_score if max_score else 0:.2f}%"></span></div></article>'
        if len(results)>100:
            content += '<p>畫面顯示前 100 篇；CSV 含全部結果。</p>'
    results_content = content
    content = '<section class="panel"><h2>已有文章 · '+str(len(engine.docs))+' 篇</h2><label for="library-filter">搜尋已載入文章</label><input class="library-filter" id="library-filter" type="search" placeholder="輸入標題、PMCID 或編號" autocomplete="off"><p id="library-filter-status" class="muted" aria-live="polite">顯示 '+str(min(len(engine.docs),200))+' 篇</p><p><a href="/export.csv">下載全部文件統計 CSV</a></p><div class="library-list">'
    for doc in sorted(engine.docs.values(), key=lambda d: engine.numbers[d.id])[:200]:
        link = '/?'+urlencode({'doc':doc.id})
        searchable = f'{number(doc)} {doc.id} {doc.title}'.casefold()
        content += f'<div class="library-item" data-search="{esc(searchable)}"><a href="{esc(link)}">{number(doc)}. {esc(doc.title)}</a><div class="muted">{esc(doc.id)}</div><div class="stats">摘要：{doc.counts["words"]:,} words · {doc.counts["sentences"]:,} sentences · {doc.counts["characters"]:,} characters</div>{document_links(doc)}{document_delete_form(doc, upload_token)}</div>'
    content += '</div><p class="muted">最多列出 200 篇，CSV 包含全部。刪除單篇文章後，文章編號、搜尋索引、Zipf 與 Word2Vec 分析會立即更新。</p></section>'
    if not query.strip() and not selected:
        results_content += '<section class="panel"><h2>開始探索你的文章</h2><p>在上方輸入英文關鍵字，這裡會列出符合的文章、斷詞與摘要統計。要挑選單篇文章，請前往 <a href="/manage">PMID查詢/上傳</a>，點選已有文章的標題。</p></section>'
    if management:
        notice_html = '<section class="panel" role="status"><h2>操作結果</h2><pre>'+esc(notice)+'</pre></section>' if notice else ''
        content = '<div class="workspace management"><aside class="sidebar" aria-label="查詢與上傳">'+import_panels+'</aside><section class="results-pane" aria-label="已有文章">'+notice_html+diagnostics+content+'</section></div>'
    else:
        content = '<div class="workspace search-workspace"><aside class="sidebar" aria-label="搜尋說明">'+sidebar+'</aside><section class="results-pane" aria-label="搜尋與文章結果">'+results_content+'</section></div>'
    content += f'<footer>Biomedical IR · 課程作業工具 · 建立索引 {engine.build_ms:.2f} ms<br>資料來源：NIH / NLM PubMed 與 PubMed Central。隨附資料為下載時快照，可能不是最新版本；本工具未受 NIH / NLM 背書。<br>Words、Sentences、Characters 只統計摘要；搜尋詞分布分別列出摘要、正文與全文。關鍵字檢索涵蓋摘要與正文，不包含標題。句數採規則估計。</footer>'
    return site_shell(engine, content, 'manage' if management else 'search', '文獻搜尋', upload_token)


WAIT_SCRIPT = """<script>
const uploadDialog = document.getElementById('single-upload-dialog');
for (const button of document.querySelectorAll('[data-open-upload]')) {
 button.addEventListener('click', () => {
  if (uploadDialog && typeof uploadDialog.showModal === 'function') uploadDialog.showModal();
 });
}
if (uploadDialog) {
 uploadDialog.addEventListener('click', event => {
  if (event.target === uploadDialog) uploadDialog.close();
 });
}
for (const form of document.querySelectorAll('form[data-wait]')) {
 form.addEventListener('submit', () => {
  const note = document.createElement('p'); note.setAttribute('role','status');
  note.textContent = form.dataset.wait; form.after(note);
  const button = form.querySelector('button'); if(button) button.disabled=true;
 });
}
for (const form of document.querySelectorAll('form[data-confirm]')) {
 form.addEventListener('submit', event => {
  if (!window.confirm(form.dataset.confirm)) event.preventDefault();
 });
}
const libraryFilter = document.querySelector('#library-filter');
const libraryStatus = document.querySelector('#library-filter-status');
if (libraryFilter) {
 const items = [...document.querySelectorAll('.library-item')];
 const applyLibraryFilter = () => {
  const words = libraryFilter.value.toLocaleLowerCase().trim().split(/\\s+/).filter(Boolean);
  let visible = 0;
  for (const item of items) {
   const haystack = item.dataset.search || '';
   const matched = words.every(word => haystack.includes(word));
   item.hidden = !matched;
   if (matched) visible += 1;
  }
  if (libraryStatus) libraryStatus.textContent = words.length ? `找到 ${visible} 篇` : `顯示 ${visible} 篇`;
 };
 libraryFilter.addEventListener('input', applyLibraryFilter);
}
for (const form of document.querySelectorAll('[data-import-source]')) {
 const source = form.querySelector('[name=source]');
 const fields = form.querySelector('.import-fields');
 const email = form.querySelector('[data-ncbi-email]');
 const query = form.querySelector('[name=query]');
 const title = form.querySelector('[name=title]');
 const ids = form.querySelector('[name=ids]');
 const help = form.querySelector('[data-source-help]');
 const savedQueries = JSON.parse(form.dataset.queries || '{}');
 const customized = new Set(Object.keys(savedQueries).filter(key => savedQueries[key]));
 const drafts = {};
 let previous = source.value;
 const automaticQuery = () => {
  const text = (title ? title.value : form.dataset.topic || '').trim();
  if (!text) return '';
  const phrase = text.split('"').join(' ').split(String.fromCharCode(92)).join(' ').trim();
  return source.value === 'arxiv' ? `(ti:"${phrase}" OR abs:"${phrase}")` : `${text}[Title/Abstract]`;
 };
 const update = (switching = false) => {
  if (switching && previous) {
   if (query && customized.has(previous)) savedQueries[previous] = query.value;
   if (ids) drafts[previous] = ids.value;
  }
  fields.hidden = !source.value;
  fields.disabled = !source.value;
  const isArxiv = source.value === 'arxiv';
  email.hidden = isArxiv || !source.value;
  email.querySelector('input').disabled = isArxiv || !source.value;
  if (query) {
   form.querySelector('[data-query-label]').textContent = isArxiv ? 'arXiv 搜尋條件' : 'PubMed 搜尋條件';
   query.placeholder = isArxiv ? '例如 (ti:"machine learning" OR abs:"machine learning") 或 cat:cs.AI' : '例如 GLP-1[Title/Abstract]';
   query.value = customized.has(source.value) ? savedQueries[source.value] || '' : automaticQuery();
   help.textContent = isArxiv ? '輸入關鍵字、title/abstract 條件，或分類（例如 cat:cs.AI）。匯入摘要，不下載 PDF；依 relevance 排序，追加時從下一批開始。' : '主題名稱只用於顯示；搜尋條件才是實際送出的查詢。依 relevance 排序，追加時從下一批開始。';
  }
  if (ids) {
   if (switching) ids.value = drafts[source.value] || '';
   form.querySelector('[data-id-label]').textContent = isArxiv ? 'arXiv IDs' : 'PMIDs';
   ids.placeholder = isArxiv ? '例如：2010.11929, 1706.03762（也可貼 arXiv /abs/ 或 /pdf/ 連結）' : '例如：40760693, 40378625, 31555683';
   form.querySelector('[data-id-limit]').textContent = `Maximum: 200 ${isArxiv ? 'arXiv IDs' : 'PMIDs'} per batch.`;
   form.querySelector('[data-id-submit]').textContent = isArxiv ? 'Add arXiv Abstracts' : 'Add PubMed Abstracts';
   help.textContent = isArxiv ? '不同版本視為同一篇文章；僅匯入摘要，不下載 PDF。' : '以 PMID 下載摘要，不要求有 PMC 全文。';
  }
  previous = source.value;
 };
 source.addEventListener('change', () => update(true));
 if (query) query.addEventListener('input', () => {
  customized.add(source.value);
  savedQueries[source.value] = query.value;
 });
 if (title) title.addEventListener('input', () => {
  if (!customized.has(source.value) || !query.value.trim()) {
   customized.delete(source.value);
   query.value = automaticQuery();
  }
 });
 update();
}
for (const chart of document.querySelectorAll('.pca-chart-wrap')) {
 const tooltip = chart.querySelector('.pca-tooltip');
 const points = chart.querySelectorAll('.pca-point');
 const hidePoint = point => {
  point.classList.remove('is-active');
  point.setAttribute('r', point.dataset.radius);
  tooltip.hidden = true;
 };
 const showPoint = (point, event) => {
  point.classList.add('is-active');
  point.setAttribute('r', (Number(point.dataset.radius) + 2.5).toFixed(2));
  const similarity = point.dataset.similarity
   ? `Cosine similarity: ${point.dataset.similarity}\n` : '';
  tooltip.textContent = `${point.dataset.word}\nPC1: ${point.dataset.pc1} · PC2: ${point.dataset.pc2}\n${similarity}Corpus frequency: ${point.dataset.frequency}`;
  tooltip.hidden = false;
  const chartBox = chart.getBoundingClientRect();
  const pointBox = point.getBoundingClientRect();
  const clientX = event && Number.isFinite(event.clientX) ? event.clientX : pointBox.left + pointBox.width/2;
  const clientY = event && Number.isFinite(event.clientY) ? event.clientY : pointBox.top + pointBox.height/2;
  let x = clientX-chartBox.left+14;
  let y = clientY-chartBox.top+14;
  x = Math.max(8, Math.min(x, chart.clientWidth-tooltip.offsetWidth-8));
  y = Math.max(8, Math.min(y, chart.clientHeight-tooltip.offsetHeight-8));
  tooltip.style.left = `${x}px`;
  tooltip.style.top = `${y}px`;
 };
 for (const point of points) {
  point.addEventListener('pointerenter', event => showPoint(point, event));
  point.addEventListener('pointermove', event => showPoint(point, event));
  point.addEventListener('pointerleave', () => hidePoint(point));
  point.addEventListener('focus', () => showPoint(point));
  point.addEventListener('blur', () => hidePoint(point));
 }
}
for (const chart of document.querySelectorAll('.zipf-chart-wrap')) {
 const tooltip = chart.querySelector('.zipf-tooltip');
 const points = chart.querySelectorAll('.zipf-point');
 const hidePoint = point => {
  point.classList.remove('is-active');
  point.setAttribute('r', '3');
  tooltip.hidden = true;
 };
 const showPoint = (point, event) => {
  point.classList.add('is-active');
  point.setAttribute('r', '6');
  let details = `${point.dataset.word}\nRank (x): ${point.dataset.rank}\nFrequency (y): ${point.dataset.frequency}`;
  if (point.dataset.logChart === 'true') {
   details += `\nlog₁₀(x): ${point.dataset.logRank} · log₁₀(y): ${point.dataset.logFrequency}`;
  }
  tooltip.textContent = details;
  tooltip.hidden = false;
  const chartBox = chart.getBoundingClientRect();
  let x = event.clientX-chartBox.left+13;
  let y = event.clientY-chartBox.top+13;
  x = Math.max(8, Math.min(x, chart.clientWidth-tooltip.offsetWidth-8));
  y = Math.max(8, Math.min(y, chart.clientHeight-tooltip.offsetHeight-8));
  tooltip.style.left = `${x}px`;
  tooltip.style.top = `${y}px`;
 };
 for (const point of points) {
  point.addEventListener('pointerenter', event => showPoint(point, event));
  point.addEventListener('pointermove', event => showPoint(point, event));
  point.addEventListener('pointerleave', () => hidePoint(point));
 }
}
window.addEventListener('pageshow',()=>document.querySelectorAll('form[data-wait] button').forEach(b=>b.disabled=false));
</script>"""


def document_links(doc):
    links = '<a href="/document.xml?'+esc(urlencode({'doc':doc.id}))+'">下載原始檔</a>'
    if re.fullmatch(r'PMC[1-9]\d*',doc.id):
        links += ' · <a href="https://pmc.ncbi.nlm.nih.gov/articles/'+esc(doc.id)+'/" target="_blank" rel="noopener">PMC 原文 ↗</a>'
    elif re.fullmatch(r'PMID[1-9]\d*',doc.id):
        links += ' · <a href="https://pubmed.ncbi.nlm.nih.gov/'+esc(doc.id[4:])+'/" target="_blank" rel="noopener">PubMed 文獻 ↗</a>'
    elif doc.id.startswith('ARXIV:'):
        links += ' · <a href="https://arxiv.org/abs/'+esc(doc.id[6:])+'" target="_blank" rel="noopener">arXiv 文獻 ↗</a>'
    return '<p class="stats">'+links+'</p>'


def document_delete_form(doc, token):
    return ('<form class="document-delete-form" action="/document-delete" method="post" '
            'data-confirm="確定刪除文章「'+esc(doc.title)+'」嗎？刪除後會從搜尋、Zipf 與 Word2Vec 分析中移除。">'
            '<input type="hidden" name="token" value="'+esc(token)+'">'
            '<input type="hidden" name="document_id" value="'+esc(doc.id)+'">'
            '<button class="danger" type="submit">Delete Article</button></form>')


def pmc_cards(pmid, pmcid, versions, errors, token, engine):
    out = '<h3>查詢總數：'+str(int(bool(versions)))+' 篇文章 · 可下載版本 '+str(len(versions))+' 個</h3>'
    out += '<p><b>PMID：'+esc(pmid)+'</b> → <b>PMCID：'+esc(pmcid)+'</b> · <a href="https://pubmed.ncbi.nlm.nih.gov/'+esc(pmid)+'/" target="_blank" rel="noopener">查看 PubMed ↗</a></p>'
    out += '<p class="muted">以下為官方資料服務中可取得的 XML 版本；版本號不保證代表最新正式出版稿。</p>'
    for meta in versions:
        version = str(meta['version'])
        query = esc(urlencode({'pmcid':pmcid,'version':version}))
        out += '<article class="card"><h3>'+esc(meta.get('title') or pmcid)+'</h3><p>'+esc(meta.get('citation',''))+'</p><p>'+esc(pmcid)+' · 版本 '+esc(version)+' · '+('作者稿' if meta.get('is_manuscript') in (True,'yes') else '非作者稿')+' · 授權 '+esc(meta.get('license_code','未標示'))+'</p>'
        if meta.get('is_retracted') in (True,'yes'):
            out += '<p class="warning">官方標示此文章已撤稿。</p>'
        out += '<p><a href="https://pmc.ncbi.nlm.nih.gov/articles/'+pmcid+'/" target="_blank" rel="noopener">前往 PMC 看文章 ↗</a> · <a href="/pmc.xml?'+query+'">下載官方 XML</a></p>'
        if pmcid in engine.docs:
            out += '<p>本站已有這篇文章。<a href="/?'+esc(urlencode({'doc':pmcid}))+'">查看本站文章與統計</a></p>'
        else:
            out += '<form action="/pmc-import" method="post" data-wait="正在下載 XML 並建立索引，請稍候…"><input type="hidden" name="token" value="'+esc(token)+'"><input type="hidden" name="pmcid" value="'+pmcid+'"><input type="hidden" name="version" value="'+esc(version)+'"><button>Add to Collection</button></form>'
        out += '</article>'
    for error in errors:
        out += '<p class="warning">'+esc(error)+'</p>'
    return out


def make_handler(store):
    state = {'active_id': store.first_id()}
    upload_lock = threading.Lock()
    upload_token = secrets.token_urlsafe(32)

    def active_engine():
        active_id = state.get('active_id')
        if active_id and not store.exists(active_id):
            state['active_id'] = store.first_id()
        return store.engine(state.get('active_id'))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            params = parse_qs(url.query)
            if url.path == '/collection-select':
                collection_id = params.get('id', [''])[0]
                destination = params.get('next', ['/'])[0]
                if not store.exists(collection_id):
                    self.send_error(404, 'Unknown topic collection'); return
                if destination not in {'/', '/manage', '/word2vec', '/analysis'}:
                    destination = '/'
                state['active_id'] = collection_id
                self.send_response(303)
                self.send_header('Location', destination)
                self.end_headers()
                return
            engine = active_engine()
            attachment = None
            if url.path in {'/', '/manage'}:
                data = page(engine,params,upload_token,management=url.path == '/manage').encode('utf-8')
                content_type = 'text/html; charset=utf-8'
            elif url.path == '/analysis':
                data = analysis_page(engine, params, upload_token).encode('utf-8')
                content_type = 'text/html; charset=utf-8'
            elif url.path == '/compare':
                data = compare_domains_page(store, engine, params, upload_token).encode('utf-8')
                content_type = 'text/html; charset=utf-8'
            elif url.path == '/term-result':
                data = term_result_page(engine, params, upload_token).encode('utf-8')
                content_type = 'text/html; charset=utf-8'
            elif url.path == '/word2vec':
                data = word2vec_page(engine, params, upload_token).encode('utf-8')
                content_type = 'text/html; charset=utf-8'
            elif url.path == '/pmc':
                value = params.get('pmid',[''])[0]
                try:
                    pmid, pmcid = pmc.resolve_pmid(value)
                    versions, errors = pmc.lookup(pmcid)
                    extra = pmc_cards(pmid,pmcid,versions,errors,upload_token,engine)
                except (ValueError, ET.ParseError, OSError) as exc:
                    extra = '<p class="warning">'+esc(exc)+'</p>'
                    if re.fullmatch(r'(?:PMID\s*[:#]?\s*)?[1-9]\d*',value.strip(),re.I):
                        pmid_link = re.sub(r'\D','',value)
                        extra += '<a href="https://pubmed.ncbi.nlm.nih.gov/'+esc(pmid_link)+'/" target="_blank" rel="noopener">前往 PubMed 核對文章 ↗</a>'
                data = page(engine,params,upload_token,pmc_html=extra,management=True).encode('utf-8')
                content_type = 'text/html; charset=utf-8'
            elif url.path == '/pmc.xml':
                try:
                    data, meta = pmc.article_xml(params.get('pmcid',[''])[0],params.get('version',[''])[0])
                    attachment = f'{meta["pmcid"]}.{meta["version"]}.xml'
                    content_type = 'application/xml; charset=utf-8'
                except (ValueError, ET.ParseError, OSError) as exc:
                    self.render_notice(str(exc),400); return
            elif url.path == '/document.xml':
                doc = engine.docs.get(params.get('doc',[''])[0])
                if not doc:
                    self.send_error(404); return
                try:
                    data = (engine.folder/doc.filename).read_bytes()
                except OSError:
                    self.render_notice('XML 檔案已被移動或刪除。',404); return
                suffix = Path(doc.filename).suffix.lower()
                attachment = f'article-{engine.numbers[doc.id]:03d}{suffix}'
                content_type = ('application/json; charset=utf-8' if suffix == '.json'
                                else 'application/xml; charset=utf-8')
            elif url.path == '/analysis.csv':
                condition = params.get('condition', ['A'])[0]
                if condition not in CONDITIONS:
                    self.send_error(400, 'Unknown preprocessing condition'); return
                result = analyze_all(engine)[condition]
                f = io.StringIO(newline='')
                writer = csv.writer(f)
                writer.writerow(['rank', 'term', 'CF', 'DF', 'IDF'])
                for rank, (term, frequency) in enumerate(result['ranked'], 1):
                    document_frequency = result['df'][term]
                    writer.writerow([rank, safe_csv(term), frequency, document_frequency,
                                     round(idf(result['documents'], document_frequency), 8)])
                data = ('\ufeff'+f.getvalue()).encode('utf-8')
                attachment = f'zipf-condition-{condition}.csv'
                content_type = 'text/csv; charset=utf-8'
            elif url.path == '/term-distribution.csv':
                condition = params.get('condition', ['A'])[0]
                value = params.get('term', [''])[0][:100]
                if condition not in CONDITIONS:
                    self.send_error(400, 'Unknown preprocessing condition'); return
                processed, error = query_term(value, condition)
                if error:
                    self.send_error(400, error); return
                result = analyze_all(engine)[condition]
                f = io.StringIO(newline='')
                writer = csv.writer(f)
                writer.writerow(['number', 'document_id', 'title', 'term', 'TF'])
                rows = []
                for document in engine.docs.values():
                    frequency = result['document_tf'][document.id][processed]
                    if frequency:
                        rows.append((frequency, engine.numbers[document.id], document))
                for frequency, number, document in sorted(rows, key=lambda row: (-row[0], row[1])):
                    writer.writerow([number, safe_csv(document.id), safe_csv(document.title),
                                     safe_csv(processed), frequency])
                data = ('\ufeff'+f.getvalue()).encode('utf-8')
                attachment = 'term-distribution.csv'
                content_type = 'text/csv; charset=utf-8'
            elif url.path == '/export.csv':
                q = params.get('q',[''])[0][:500]
                mode = params.get('mode',['AND'])[0]
                if mode not in {'AND','OR','PHRASE'}:
                    self.send_error(400,'Unknown mode'); return
                scoped_doc = params.get('doc',[''])[0] or None
                rows = engine.search(q,mode,scoped_doc)[0] if q.strip() else [(d,0) for d in engine.docs.values()]
                f = io.StringIO(newline='')
                writer = csv.writer(f)
                writer.writerow(['number','id','title','BM25','characters','characters_no_space','words','sentences','indexed_terms','unique_terms','filename'])
                for d,s in rows:
                    writer.writerow([engine.numbers[d.id],safe_csv(d.id),safe_csv(d.title),round(s,6),*d.counts.values(),safe_csv(d.filename)])
                data = ('\ufeff'+f.getvalue()).encode('utf-8')
                content_type = 'text/csv; charset=utf-8'
            else:
                self.send_error(404); return
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length',str(len(data)))
            self.send_header('X-Content-Type-Options','nosniff')
            if attachment:
                self.send_header('Content-Disposition',f'attachment; filename="{attachment}"')
            if url.path.endswith('.csv') and not attachment:
                self.send_header('Content-Disposition','attachment; filename="results.csv"')
            self.end_headers()
            self.wfile.write(data)
        def do_POST(self):
            post_path = urlsplit(self.path).path
            if post_path in {'/pubmed-import', '/abstract-import'}:
                self.import_pubmed_collection(); return
            if post_path in {'/pmid-batch-import', '/id-batch-import'}:
                self.import_pmid_batch(); return
            if post_path == '/collection-delete':
                self.delete_collection(); return
            if post_path == '/document-delete':
                self.delete_document(); return
            if post_path == '/pmc-import':
                self.import_pmc(); return
            if post_path != '/upload':
                self.send_error(404); return
            limit = 42 * 1024 * 1024  # 40 MB file payload plus multipart envelope.
            try:
                length = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                self.send_error(400, 'Invalid Content-Length'); return
            if length <= 0 or length > limit or self.headers.get('Transfer-Encoding'):
                self.send_error(413, 'Upload is too large or has no valid length'); return
            content_type = self.headers.get('Content-Type', '')
            if not content_type.lower().startswith('multipart/form-data'):
                self.send_error(400, 'Expected multipart/form-data'); return
            self.connection.settimeout(30)
            try:
                raw = self.rfile.read(length)
            except (socket.timeout, OSError):
                self.send_error(408, 'Upload timed out'); return
            if len(raw) != length:
                self.send_error(400, 'Incomplete upload'); return
            message = BytesParser(policy=policy.default).parsebytes(
                ('Content-Type: '+content_type+'\r\nMIME-Version: 1.0\r\n\r\n').encode('utf-8')+raw)
            if not message.is_multipart():
                self.send_error(400, 'Invalid multipart boundary'); return
            parts = list(message.iter_parts())
            supplied = next((p.get_payload(decode=True) for p in parts
                             if p.get_param('name', header='content-disposition') == 'token'), b'')
            if not secrets.compare_digest(supplied or b'', upload_token.encode()):
                self.send_error(403, 'Reload the page and try uploading again'); return
            upload_mode_raw = next((p.get_payload(decode=True) for p in parts
                                    if p.get_param('name', header='content-disposition') == 'upload_mode'), b'collection')
            try:
                upload_mode = (upload_mode_raw or b'collection').decode('utf-8')
            except UnicodeDecodeError:
                self.send_error(400, 'Invalid upload mode'); return
            if upload_mode not in {'single', 'collection'}:
                self.send_error(400, 'Invalid upload mode'); return
            files = [p for p in parts if p.get_param('name', header='content-disposition') == 'files'
                     and p.get_filename()]
            maximum_files = 1 if upload_mode == 'single' else 10
            if not 1 <= len(files) <= maximum_files:
                self.send_error(400, '單篇上傳每次只能選擇 1 個檔案。' if upload_mode == 'single'
                                else 'Select between 1 and 10 files'); return
            payloads = [(p.get_filename(), p.get_payload(decode=True) or b'') for p in files]
            if sum(len(raw) for _,raw in payloads) > 40*1024*1024:
                self.send_error(413, 'Total file size exceeds 40 MB'); return
            messages = []
            # Serialize writes; GET requests always read a complete immutable index snapshot.
            with upload_lock:
                collection_id = state.get('active_id')
                if not collection_id or not store.exists(collection_id):
                    self.render_notice('請先建立或選擇一個主題資料集，再上傳文章。', 400); return
                for filename, raw in payloads:
                    basename = filename.replace('\\', '/').split('/')[-1]
                    suffix = Path(basename).suffix.lower()
                    if suffix not in {'.xml', '.nxml', '.json'}:
                        messages.append(f'{basename}: 不支援此格式，請選 XML、NXML 或 pubmed.json。'); continue
                    if upload_mode == 'single' and suffix not in {'.xml', '.nxml'}:
                        messages.append(f'{basename}: 單篇上傳請選擇 PMC JATS XML 或 NXML。'); continue
                    if not raw or len(raw) > 20*1024*1024:
                        messages.append(f'{basename}: 檔案為空或超過 20 MB。'); continue
                    current = store.engine(collection_id)
                    # Ignore client path entirely. Content-derived names prevent overwrites.
                    digest = hashlib.sha256(raw).hexdigest()
                    target = current.folder / ('upload_'+digest+suffix)
                    if target.exists():
                        messages.append(f'{basename}: 相同內容已存在，未重複加入。'); continue
                    try:
                        with tempfile.TemporaryDirectory(dir=current.folder) as folder:
                            candidate = Path(folder)/target.name
                            candidate.write_bytes(raw)
                            docs = parse_xml(candidate)
                            if upload_mode == 'single' and len(docs) != 1:
                                messages.append(f'{basename}: 檔案內含 {len(docs)} 篇文章；單篇上傳只接受一篇，請改用 PubMed collection 上傳。')
                                continue
                            ids = [d.id for d in docs]
                            duplicate = set(ids) & set(current.docs)
                            if duplicate or len(set(ids)) != len(ids):
                                messages.append(f'{basename}: 文件 ID 重複，整個檔案略過。'); continue
                            candidate.replace(target)
                        try:
                            updated = store.reload(collection_id)
                            if not all(doc_id in updated.docs for doc_id in ids):
                                raise ValueError('New document was not indexed')
                        except Exception:
                            target.unlink(missing_ok=True)
                            raise
                        numbers = ', '.join(f'{updated.numbers[i]:03d}' for i in ids)
                        messages.append(f'{basename}: 已加入 {len(docs)} 篇文章（編號 {numbers}），可立即搜尋與查看統計。')
                    except (ValueError, ET.ParseError, OSError) as exc:
                        messages.append(f'{basename}: 上傳失敗，請確認是有效的 PMC JATS、含 PMID＋abstract 的 PubMed XML，或本專案格式的 pubmed.json。原因：{exc}')
                engine = store.engine(collection_id)
            data = page(engine, {}, upload_token, '\n'.join(messages),management=True).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type','text/html; charset=utf-8')
            self.send_header('Content-Length',str(len(data)))
            self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers()
            self.wfile.write(data)
        def render_notice(self, text, status=200, params=None):
            management = urlsplit(self.path).path in {'/upload', '/collection-delete', '/document-delete', '/pubmed-import', '/abstract-import', '/pmid-batch-import', '/id-batch-import', '/pmc-import', '/pmc.xml'}
            data = page(active_engine(),params or {},upload_token,text,management=management).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type','text/html; charset=utf-8')
            self.send_header('Content-Length',str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def delete_document(self):
            """Remove one document from the current topic and rebuild every analysis."""
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096 or self.headers.get('Transfer-Encoding'):
                    raise ValueError('文章刪除請求格式不正確。')
                values = parse_qs(self.rfile.read(length).decode('utf-8'))
                token = values.get('token', [''])[0]
                if not secrets.compare_digest(token.encode(), upload_token.encode()):
                    self.render_notice('頁面已過期，請重新整理後再刪除。', 403); return
                document_id = values.get('document_id', [''])[0]
                with upload_lock:
                    collection_id = state.get('active_id')
                    if not collection_id or not store.exists(collection_id):
                        self.render_notice('請先選擇一個主題資料集。', 400); return
                    current = store.engine(collection_id)
                    document = current.docs.get(document_id)
                    if document is None:
                        self.render_notice('找不到這篇文章，可能已經被刪除。', 404); return
                    title = document.title
                    number = current.numbers[document_id]
                    updated = store.delete_document(collection_id, document_id)
                self.render_notice(f'已刪除文章 {number:03d}「{title}」。現在主題內共有 {len(updated.docs):,} 篇文章。')
            except (ValueError, UnicodeDecodeError, OSError) as exc:
                self.render_notice(str(exc), 400)

        def delete_collection(self):
            """Permanently remove one named topic and all of its documents."""
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096 or self.headers.get('Transfer-Encoding'):
                    raise ValueError('主題刪除請求格式不正確。')
                values = parse_qs(self.rfile.read(length).decode('utf-8'))
                token = values.get('token', [''])[0]
                if not secrets.compare_digest(token.encode(), upload_token.encode()):
                    self.render_notice('頁面已過期，請重新整理後再刪除。', 403); return
                collection_id = values.get('collection_id', [''])[0]
                with upload_lock:
                    item = store.get(collection_id)
                    if item is None:
                        self.render_notice('找不到這個主題，可能已經被刪除。', 404); return
                    count = len(store.engine(collection_id).docs)
                    title = item['title']
                    store.delete(collection_id)
                    if state.get('active_id') == collection_id:
                        state['active_id'] = store.first_id()
                self.render_notice(f'已刪除主題「{title}」及其中 {count:,} 篇文章。')
            except (ValueError, UnicodeDecodeError, OSError) as exc:
                self.render_notice(str(exc), 400)

        def import_pubmed_collection(self):
            created_collection = None
            source_name = '摘要'
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096 or self.headers.get('Transfer-Encoding'):
                    raise ValueError('摘要下載請求格式不正確。')
                values = parse_qs(self.rfile.read(length).decode('utf-8'))
                token = values.get('token', [''])[0]
                if not secrets.compare_digest(token.encode(), upload_token.encode()):
                    self.render_notice('頁面已過期，請重新整理後再下載。', 403); return
                source = values.get('source', ['pubmed' if urlsplit(self.path).path == '/pubmed-import' else ''])[0]
                if source not in {'pubmed', 'arxiv'}:
                    raise ValueError('請先選擇 PubMed 或 arXiv。')
                source_name = 'arXiv' if source == 'arxiv' else 'PubMed'
                importer = arxiv_bulk if source == 'arxiv' else pubmed_bulk
                collection_id = values.get('collection_id', [''])[0].strip()
                title = values.get('title', [''])[0].strip()
                if collection_id and not store.exists(collection_id):
                    raise ValueError('找不到要追加文章的主題。')
                if not collection_id and not title:
                    raise ValueError('建立新主題時必須輸入主題顯示名稱。')
                query = values.get('query', [''])[0].strip()
                if not query and title:
                    query = arxiv_bulk.default_query(title) if source == 'arxiv' else f'{title}[Title/Abstract]'
                try:
                    count = int(values.get('count', ['500'])[0])
                except ValueError as exc:
                    raise ValueError('篇數必須是10～1,000的整數。') from exc
                email = values.get('email', [''])[0].strip() if source == 'pubmed' else ''
                if email and (len(email) > 200 or not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', email)):
                    raise ValueError('NCBI聯絡信箱格式不正確。')
                retstart = store.next_offset(collection_id, query, source) if collection_id else 0
                raw, metadata = importer.fetch_abstracts(query, count, email, retstart=retstart)
                with upload_lock:
                    if not collection_id:
                        collection_id = store.create(title, query, source)
                        created_collection = collection_id
                    current = store.engine(collection_id)
                    known_ids = set(current.docs) | set(current.deleted_ids)
                    raw, remaining, duplicate_count = importer.remove_known_records(raw, known_ids)
                    if not remaining:
                        store.record_import(collection_id, query, count, metadata['downloaded'], 0, retstart, source)
                        state['active_id'] = collection_id
                        self.render_notice(
                            f'主題「{store.get(collection_id)["title"]}」已檢查下一批 {metadata["downloaded"]} 篇，'
                            '但文章都已存在或先前已刪除，未重複加入。')
                        return
                    digest = hashlib.sha256(raw).hexdigest()[:20]
                    target = current.folder / f'{source}_{digest}.xml'
                    if target.exists():
                        raise ValueError('相同的摘要 collection 檔案已存在。')
                    ids = []
                    try:
                        with tempfile.TemporaryDirectory(dir=current.folder) as folder:
                            candidate = Path(folder) / target.name
                            candidate.write_bytes(raw)
                            documents = parse_xml(candidate)
                            ids = [document.id for document in documents]
                            if len(ids) != len(set(ids)) or set(ids) & known_ids:
                                raise ValueError('下載結果仍含重複 Document ID，未寫入資料集。')
                            candidate.replace(target)
                        updated = store.reload(collection_id)
                        if not all(document_id in updated.docs for document_id in ids):
                            raise ValueError('摘要已下載，但未能完整建立索引。')
                        store.record_import(collection_id, query, count, metadata['downloaded'], len(ids), retstart, source)
                    except Exception:
                        target.unlink(missing_ok=True)
                        if store.exists(collection_id):
                            store.reload(collection_id)
                        raise
                    state['active_id'] = collection_id
                message = (f'主題「{store.get(collection_id)["title"]}」：{source_name} 查詢「{metadata["query"]}」'
                           f'共找到 {metadata["available"]:,} 篇；本次從第 {retstart+1:,} 筆開始，'
                           f'要求 {metadata["requested"]:,} 篇、下載 {metadata["downloaded"]:,} 篇，'
                           f'成功加入 {len(ids):,} 篇摘要。')
                if duplicate_count:
                    message += f' 另有 {duplicate_count:,} 篇已存在或先前已刪除的文章已略過。'
                self.render_notice(message)
            except (ValueError, UnicodeDecodeError, ET.ParseError, OSError) as exc:
                if created_collection and store.exists(created_collection):
                    try:
                        store.delete(created_collection)
                    except OSError:
                        pass
                self.render_notice(f'{source_name} 批次下載失敗：{exc}', 400)

        def import_pmid_batch(self):
            """Fetch source-specific Document IDs into the explicitly selected topic."""
            source_name = 'Document ID'
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 65536 or self.headers.get('Transfer-Encoding'):
                    raise ValueError('Document ID 匯入請求格式不正確。')
                values = parse_qs(self.rfile.read(length).decode('utf-8'))
                token = values.get('token', [''])[0]
                if not secrets.compare_digest(token.encode(), upload_token.encode()):
                    self.render_notice('頁面已過期，請重新整理後再匯入。', 403); return
                source = values.get('source', ['pubmed' if urlsplit(self.path).path == '/pmid-batch-import' else ''])[0]
                if source not in {'pubmed', 'arxiv'}:
                    raise ValueError('請先選擇 PubMed 或 arXiv。')
                source_name = 'arXiv ID' if source == 'arxiv' else 'PMID'
                importer = arxiv_bulk if source == 'arxiv' else pubmed_bulk
                pasted = values.get('ids', values.get('pmids', ['']))[0]
                pmids, pasted_duplicates = (arxiv_bulk.parse_id_batch(pasted) if source == 'arxiv' else parse_pmid_batch(pasted))
                email = values.get('email', [''])[0].strip() if source == 'pubmed' else ''
                if email and (len(email) > 200 or not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', email)):
                    raise ValueError('NCBI 聯絡信箱格式不正確。')
                collection_id = values.get('collection_id', [state.get('active_id') if urlsplit(self.path).path == '/pmid-batch-import' else ''])[0]
                if not collection_id or not store.exists(collection_id):
                    raise ValueError('請先建立或選擇一個主題資料集。')

                raw, metadata = (arxiv_bulk.fetch_ids(pmids) if source == 'arxiv' else pubmed_bulk.fetch_pmids(pmids, email))
                with upload_lock:
                    current = store.engine(collection_id)
                    known_ids = set(current.docs) | set(current.deleted_ids)
                    raw, remaining, existing_count = importer.remove_known_records(raw, known_ids)
                    if not remaining:
                        message = (f'已查詢 {metadata["requested"]:,} 個 {source_name}，但可用摘要都已在此主題中'
                                   '（包含先前刪除的文章），因此沒有重複加入。')
                        if pasted_duplicates:
                            message += f' 輸入內容另有 {pasted_duplicates:,} 個重複 ID 已略過。'
                        self.render_notice(message)
                        return

                    digest = hashlib.sha256(raw).hexdigest()[:20]
                    target = current.folder / f'{source}_ids_{digest}.xml'
                    if target.exists():
                        raise ValueError('相同的 ID 摘要檔案已存在。')
                    ids = []
                    try:
                        with tempfile.TemporaryDirectory(dir=current.folder) as folder:
                            candidate = Path(folder) / target.name
                            candidate.write_bytes(raw)
                            documents = parse_xml(candidate)
                            ids = [document.id for document in documents]
                            if len(ids) != len(set(ids)) or set(ids) & known_ids:
                                raise ValueError('回傳結果仍含重複 Document ID，未寫入資料集。')
                            candidate.replace(target)
                        updated = store.reload(collection_id)
                        if not all(document_id in updated.docs for document_id in ids):
                            raise ValueError('摘要已下載，但未能完整建立索引。')
                    except Exception:
                        target.unlink(missing_ok=True)
                        store.reload(collection_id)
                        raise

                message = (f'已查詢 {metadata["requested"]:,} 個不同 {source_name}，成功加入 {len(ids):,} 篇摘要；'
                           f'目前主題共有 {len(updated.docs):,} 篇文章。')
                details = []
                if existing_count:
                    details.append(f'{existing_count:,} 篇已存在或先前已刪除')
                if metadata['without_abstract']:
                    details.append(f'{metadata["without_abstract"]:,} 篇沒有摘要')
                if metadata['not_found']:
                    details.append(f'{metadata["not_found"]:,} 個 ID 未找到')
                if pasted_duplicates:
                    details.append(f'輸入內 {pasted_duplicates:,} 個重複值')
                if details:
                    message += ' 另略過：' + '、'.join(details) + '。'
                self.render_notice(message)
            except (ValueError, UnicodeDecodeError, ET.ParseError, OSError) as exc:
                self.render_notice(f'{source_name} 批次匯入失敗：{exc}', 400)

        def import_pmc(self):
            try:
                length = int(self.headers.get('Content-Length','0'))
                if not 0<length<=4096 or self.headers.get('Transfer-Encoding'):
                    raise ValueError('匯入請求格式不正確。')
                self.connection.settimeout(30)
                values = parse_qs(self.rfile.read(length).decode('utf-8'))
                token = values.get('token',[''])[0]
                if not secrets.compare_digest(token.encode(),upload_token.encode()):
                    self.render_notice('頁面已過期，請重新整理後再匯入。',403); return
                collection_id = state.get('active_id')
                if not collection_id or not store.exists(collection_id):
                    raise ValueError('請先建立或選擇一個主題資料集。')
                pmcid = pmc.normalize_pmcid(values.get('pmcid',[''])[0])
                version = values.get('version',[''])[0]
                if pmcid in store.engine(collection_id).docs:
                    self.render_notice('本站已有這篇文章，未重複加入。',params={'doc':[pmcid]}); return
                raw, meta = pmc.article_xml(pmcid,version)
                with upload_lock:
                    current = store.engine(collection_id)
                    if pmcid in current.docs:
                        self.render_notice('本站已有這篇文章，未重複加入。',params={'doc':[pmcid]}); return
                    target = current.folder/f'{pmcid}.{version}.xml'
                    sidecar = current.folder/f'{pmcid}.{version}.metadata.json'
                    if target.exists() or sidecar.exists():
                        raise ValueError('已有同名檔案，請先檢查 data 資料夾，系統不會覆寫。')
                    try:
                        with tempfile.TemporaryDirectory(dir=current.folder) as tempdir:
                            temporary = Path(tempdir)/target.name
                            temporary.write_bytes(raw)
                            temporary.replace(target)
                        sidecar.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
                        updated = store.reload(collection_id)
                        if pmcid not in updated.docs:
                            raise ValueError('無法建立新文章索引。')
                    except Exception:
                        target.unlink(missing_ok=True)
                        sidecar.unlink(missing_ok=True)
                        raise
                self.render_notice(f'已下載並加入 {pmcid}，文章編號 {updated.numbers[pmcid]:03d}。可立即搜尋、查看統計及下載 XML。',params={'doc':[pmcid]})
            except (ValueError, ET.ParseError, OSError) as exc:
                self.render_notice(str(exc),400)
    return Handler


def main():
    parser = argparse.ArgumentParser(description='Local biomedical full-text search')
    parser.add_argument('--data',type=Path,default=ROOT/'data')
    parser.add_argument('--port',type=int,default=8765)
    args = parser.parse_args()
    args.data.mkdir(parents=True,exist_ok=True)
    store = CollectionStore(args.data)
    engine = store.engine(store.first_id())
    try:
        server = ThreadingHTTPServer(('127.0.0.1',args.port),make_handler(store))
    except OSError as exc:
        raise SystemExit(f'Cannot open port {args.port}: {exc}. Try --port 8766')
    print(f'Loaded {len(engine.docs)} documents; {len(engine.errors)} warnings.',flush=True)
    print(f'Open http://127.0.0.1:{args.port} in your browser. Ctrl+C to stop.',flush=True)
    for error in engine.errors:
        print('Warning:',error,flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == '__main__':
    main()
