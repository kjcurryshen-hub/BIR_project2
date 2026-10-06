"""Independent subject collections for the local Biomedical IR website."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import secrets
import shutil

from engine import SearchEngine


class CollectionStore:
    """Persist named topics and keep one SearchEngine per topic."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.collections_root = self.root / 'collections'
        self.collections_root.mkdir(exist_ok=True)
        self.metadata_path = self.root / 'collections.json'
        self._engines = {}
        self._metadata = self._load()

    def _load(self):
        if not self.metadata_path.exists():
            return {'version': 1, 'collections': {}}
        try:
            value = json.loads(self.metadata_path.read_text(encoding='utf-8'))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f'無法讀取 collections.json：{exc}') from exc
        if not isinstance(value, dict) or not isinstance(value.get('collections'), dict):
            raise ValueError('collections.json 格式錯誤。')
        return value

    def _save(self):
        temporary = self.metadata_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(self._metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temporary, self.metadata_path)

    @property
    def ids(self):
        return list(self._metadata['collections'])

    def exists(self, collection_id):
        return collection_id in self._metadata['collections']

    def get(self, collection_id):
        return self._metadata['collections'].get(collection_id)

    def first_id(self):
        return self.ids[0] if self.ids else None

    def folder(self, collection_id):
        if not self.exists(collection_id):
            raise ValueError('找不到指定的主題資料集。')
        folder = (self.collections_root / collection_id).resolve()
        if folder.parent != self.collections_root:
            raise ValueError('主題資料夾路徑不安全。')
        return folder

    def create(self, title, query='', source='pubmed'):
        title = re.sub(r'\s+', ' ', title).strip()
        if not 1 <= len(title) <= 80:
            raise ValueError('主題名稱必須是1～80個字元。')
        if any(item['title'].casefold() == title.casefold()
               for item in self._metadata['collections'].values()):
            raise ValueError('已有相同名稱的主題；請直接在該主題追加文章。')
        base = re.sub(r'[^a-z0-9]+', '-', title.casefold()).strip('-')[:36] or 'topic'
        collection_id = f'{base}-{secrets.token_hex(4)}'
        folder = self.collections_root / collection_id
        folder.mkdir()
        (folder / 'document_numbers.json').write_text('{}\n', encoding='utf-8')
        now = datetime.now(timezone.utc).isoformat(timespec='seconds')
        self._metadata['collections'][collection_id] = {
            'title': title,
            'default_query': query.strip(),
            'source': source,
            'source_queries': {source: query.strip()},
            'created_at': now,
            'document_count': 0,
            'imports': [],
        }
        try:
            self._save()
        except Exception:
            shutil.rmtree(folder)
            self._metadata['collections'].pop(collection_id, None)
            raise
        return collection_id

    def summaries(self):
        return [
            {'id': collection_id,
             'title': item['title'],
             'query': item.get('default_query', ''),
             'source': item.get('source', 'pubmed'),
             'source_queries': item.get('source_queries', {item.get('source', 'pubmed'): item.get('default_query', '')}),
             'count': int(item.get('document_count', 0))}
            for collection_id, item in self._metadata['collections'].items()
        ]

    def _decorate(self, engine, collection_id):
        item = self.get(collection_id) if collection_id else None
        engine.collection_id = collection_id or ''
        engine.collection_title = item['title'] if item else '尚未建立主題'
        engine.collection_query = item.get('default_query', '') if item else ''
        engine.collection_source = item.get('source', 'pubmed') if item else 'pubmed'
        engine.collection_summaries = self.summaries()
        return engine

    def engine(self, collection_id):
        if collection_id and self.exists(collection_id):
            if collection_id not in self._engines:
                self._engines[collection_id] = SearchEngine(self.folder(collection_id))
                self._metadata['collections'][collection_id]['document_count'] = len(self._engines[collection_id].docs)
                self._save()
            return self._decorate(self._engines[collection_id], collection_id)
        # The root has no active XML files; it is a convenient empty engine.
        return self._decorate(SearchEngine(self.root), None)

    def reload(self, collection_id):
        engine = SearchEngine(self.folder(collection_id))
        self._engines[collection_id] = engine
        self._metadata['collections'][collection_id]['document_count'] = len(engine.docs)
        self._save()
        return self._decorate(engine, collection_id)

    def next_offset(self, collection_id, query, source='pubmed'):
        imports = self.get(collection_id).get('imports', [])
        return max((int(item.get('retstart', 0)) + int(item.get('downloaded', 0))
                    for item in imports if item.get('source', 'pubmed') == source
                    and item.get('query', '').casefold() == query.casefold()),
                   default=0)

    def record_import(self, collection_id, query, requested, downloaded, added, retstart, source='pubmed'):
        item = self.get(collection_id)
        item.setdefault('source_queries', {item.get('source', 'pubmed'): item.get('default_query', '')})
        item['default_query'] = query
        item['source'] = source
        item.setdefault('source_queries', {})[source] = query
        item.setdefault('imports', []).append({
            'source': source,
            'query': query,
            'requested': requested,
            'downloaded': downloaded,
            'added': added,
            'retstart': retstart,
            'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        })
        item['document_count'] = len(self._engines[collection_id].docs)
        self._save()

    def delete_document(self, collection_id, document_id):
        """Remove one document from the active index, including multi-record source files."""
        engine = self.engine(collection_id)
        if document_id not in engine.docs:
            raise ValueError('找不到指定的文章。')
        deleted_path = self.folder(collection_id) / 'deleted_documents.json'
        deleted = set(engine.deleted_ids)
        deleted.add(document_id)
        temporary = deleted_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(sorted(deleted), ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temporary, deleted_path)
        return self.reload(collection_id)

    def delete(self, collection_id):
        item = self.get(collection_id)
        if item is None:
            raise ValueError('找不到指定的主題資料集。')
        folder = self.folder(collection_id)
        if folder.parent != self.collections_root or not folder.is_dir():
            raise ValueError('主題資料夾不存在或路徑不安全。')
        # Remove metadata only after the exact topic directory is gone.
        shutil.rmtree(folder)
        self._engines.pop(collection_id, None)
        self._metadata['collections'].pop(collection_id, None)
        self._save()
        return item
