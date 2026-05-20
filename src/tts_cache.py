# tts_cache.py
"""
TTS Caching System - Stores generated speech files locally to reduce internet dependency
Automatically checks cache before making TTS HTTP requests
"""

import os
import hashlib
import json
import threading
from pathlib import Path

class TTSCache:
    def __init__(self, cache_dir="tts_cache"):
        """
        Initialize TTS Cache Manager
        
        Args:
            cache_dir: Directory to store cached audio files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_index_file = self.cache_dir / "cache_index.json"
        self.lock = threading.Lock()
        self._load_cache_index()
    
    def _load_cache_index(self):
        """Load the cache index from disk"""
        if self.cache_index_file.exists():
            try:
                with open(self.cache_index_file, 'r', encoding='utf-8') as f:
                    self.cache_index = json.load(f)
            except (json.JSONDecodeError, OSError):
                print(f"[TTS Cache] Warning: corrupted cache index, resetting.")
                self.cache_index = {}
        else:
            self.cache_index = {}
    
    def _save_cache_index(self):
        """Save the cache index to disk"""
        try:
            with open(self.cache_index_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[TTS Cache] Error saving index: {e}")
    
    def _text_to_hash(self, text):
        """Generate a unique hash for the text"""
        text_normalized = text.strip().lower()
        return hashlib.md5(text_normalized.encode()).hexdigest()
    
    def get_cached_file(self, text):
        """
        Check if text has been cached, return file path if exists
        
        Args:
            text: The text to check
            
        Returns:
            Path to cached audio file, or None if not in cache
        """
        with self.lock:
            text_hash = self._text_to_hash(text)
            
            if text_hash in self.cache_index:
                cached_entry = self.cache_index[text_hash]
                file_path = self.cache_dir / cached_entry["filename"]
                
                if file_path.exists():
                    return str(file_path)
                else:
                    # File was deleted, remove from index
                    del self.cache_index[text_hash]
                    self._save_cache_index()
            
            return None
    
    def add_to_cache(self, text, file_path):
        """
        Add a generated TTS audio file to cache
        
        Args:
            text: The original text
            file_path: Path to the generated audio file
        """
        with self.lock:
            text_hash = self._text_to_hash(text)
            filename = Path(file_path).name
            
            self.cache_index[text_hash] = {
                "text": text,
                "filename": filename,
                "hash": text_hash
            }
            self._save_cache_index()
    
    def clear_cache(self):
        """Clear all cached audio files"""
        with self.lock:
            try:
                for file in self.cache_dir.glob("*.mp3"):
                    file.unlink()
                for file in self.cache_dir.glob("*.wav"):
                    file.unlink()
                for file in self.cache_dir.glob("*.ogg"):
                    file.unlink()
                self.cache_index = {}
                self._save_cache_index()
                print("[TTS Cache] Cache cleared successfully")
            except Exception as e:
                print(f"[TTS Cache] Error clearing cache: {e}")
    
    def get_cache_dir(self):
        return str(self.cache_dir)

    def hash_text(self, text):
        return self._text_to_hash(text)

    def get_cache_info(self):
        """Get information about cached files"""
        with self.lock:
            total_files = len([f for f in self.cache_dir.glob("*") if f.is_file() and f.name != "cache_index.json"])
            total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*") if f.is_file())
            
            return {
                "cache_dir": str(self.cache_dir),
                "total_entries": len(self.cache_index),
                "total_files": total_files,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "entries": list(self.cache_index.values())
            }


# Global cache instance
_tts_cache_instance = None
_tts_cache_lock = threading.Lock()

def get_tts_cache():
    """Get or create the global TTS cache instance"""
    global _tts_cache_instance
    if _tts_cache_instance is None:
        with _tts_cache_lock:
            if _tts_cache_instance is None:
                _tts_cache_instance = TTSCache("tts_cache")
    return _tts_cache_instance
