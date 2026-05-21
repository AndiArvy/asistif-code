from __future__ import annotations

import os
import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional


class TTSCache:
    def __init__(self, cache_dir: str = "tts_cache") -> None:
        self.cache_dir: Path = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_index_file: Path = self.cache_dir / "cache_index.json"
        self.lock: threading.Lock = threading.Lock()
        self.cache_index: Dict[str, Any] = {}
        self._load_cache_index()

    def _load_cache_index(self) -> None:
        if self.cache_index_file.exists():
            try:
                with open(self.cache_index_file, 'r', encoding='utf-8') as f:
                    self.cache_index = json.load(f)
            except (json.JSONDecodeError, OSError):
                print("[TTS Cache] Warning: corrupted cache index, resetting.")
                self.cache_index = {}

    def _save_cache_index(self) -> None:
        try:
            with open(self.cache_index_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[TTS Cache] Error saving index: {e}")

    def _text_to_hash(self, text: str) -> str:
        text_normalized = text.strip().lower()
        return hashlib.md5(text_normalized.encode()).hexdigest()

    def get_cached_file(self, text: str) -> Optional[str]:
        with self.lock:
            text_hash = self._text_to_hash(text)
            if text_hash in self.cache_index:
                cached_entry = self.cache_index[text_hash]
                file_path = self.cache_dir / cached_entry["filename"]
                if file_path.exists():
                    return str(file_path)
                del self.cache_index[text_hash]
                self._save_cache_index()
            return None

    def add_to_cache(self, text: str, file_path: str) -> None:
        with self.lock:
            text_hash = self._text_to_hash(text)
            filename = Path(file_path).name
            self.cache_index[text_hash] = {
                "text": text,
                "filename": filename,
                "hash": text_hash,
            }
            self._save_cache_index()

    def get_cache_dir(self) -> str:
        return str(self.cache_dir)

    def hash_text(self, text: str) -> str:
        return self._text_to_hash(text)

    def get_cache_info(self) -> Dict[str, Any]:
        with self.lock:
            total_files = len([
                f for f in self.cache_dir.glob("*")
                if f.is_file() and f.name != "cache_index.json"
            ])
            total_size = sum(
                f.stat().st_size for f in self.cache_dir.glob("*") if f.is_file()
            )
            return {
                "cache_dir": str(self.cache_dir),
                "total_entries": len(self.cache_index),
                "total_files": total_files,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "entries": list(self.cache_index.values()),
            }


_tts_cache_instance: Optional[TTSCache] = None
_tts_cache_lock: threading.Lock = threading.Lock()


def get_tts_cache() -> TTSCache:
    global _tts_cache_instance
    if _tts_cache_instance is None:
        with _tts_cache_lock:
            if _tts_cache_instance is None:
                _tts_cache_instance = TTSCache("tts_cache")
    return _tts_cache_instance
