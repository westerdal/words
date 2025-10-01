#!/usr/bin/env python3
"""
OpenAI Clue Cache System
Persistent caching system to store OpenAI API responses for clue generation.
"""

import json
import os
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

try:
    from .config import Config
    from .progress_tracker import quick_log
except ImportError:
    # Handle standalone execution
    import sys
    sys.path.append(str(Path(__file__).parent))
    from config import Config
    from progress_tracker import quick_log


class ClueCache:
    """Manages persistent caching of OpenAI clue generation results"""
    
    def __init__(self, secret_word: str):
        self.secret_word = secret_word.lower()
        self.cache_file = Config.SECRETWORD_DIR / Config.get_cache_filename(self.secret_word)
        self.generic_cache_file = Config.SECRETWORD_DIR / "cache-generic.json"
        
        # Performance tracking
        self.hits = 0
        self.misses = 0
        self.api_calls_saved = 0
        self.session_start = datetime.now(timezone.utc)
        
        # Create prompt hash for validation
        self.current_prompt_hash = self._generate_prompt_hash()
        
        # Load existing caches
        self.cache = self.load_cache()
        self.generic_cache = self.load_generic_cache()
        
    def load_cache(self) -> Dict:
        """Load existing cache or create new one"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    
                # Validate cache version and prompt compatibility
                if self._is_cache_valid(cache_data):
                    quick_log(self.secret_word, f"📋 Loaded cache with {len(cache_data.get('clues', {})):,} entries")
                    return cache_data
                else:
                    quick_log(self.secret_word, f"⚠️ Cache invalid/outdated - creating new cache")
                    return self.create_empty_cache()
                    
            except (json.JSONDecodeError, KeyError) as e:
                quick_log(self.secret_word, f"⚠️ Cache corrupted ({e}) - creating new cache")
                return self.create_empty_cache()
        else:
            return self.create_empty_cache()
    
    def load_generic_cache(self) -> Dict:
        """Load shared generic cache for universally weak words"""
        if self.generic_cache_file.exists():
            try:
                with open(self.generic_cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, KeyError):
                return {"clues": {}, "metadata": {"created_at": datetime.now(timezone.utc).isoformat()}}
        else:
            return {"clues": {}, "metadata": {"created_at": datetime.now(timezone.utc).isoformat()}}
    
    def create_empty_cache(self) -> Dict:
        """Create a new empty cache structure"""
        return {
            "metadata": {
                "secret_word": self.secret_word,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "total_cached_clues": 0,
                "api_calls_saved": 0,
                "cache_version": "1.0",
                "prompt_hash": self.current_prompt_hash,
                "model_used": Config.OPENAI_MODEL,
                "max_tokens": 2000,
                "temperature": 0.7,
                "processing_stats": {
                    "total_words_processed": 0,
                    "words_from_cache": 0,
                    "words_from_api": 0,
                    "violation_rate": 0.0,
                    "most_common_violations": []
                }
            },
            "clues": {}
        }
    
    def _generate_prompt_hash(self) -> str:
        """Generate hash of current prompt for validation"""
        # This would be the actual prompt used in the AI generation
        prompt_template = (
            f"CRITICAL: You are a '{self.secret_word}' speaking about guess words in a word guessing game. "
            f"For each guess word, write a riddle from YOUR PERSPECTIVE describing your RELATIONSHIP to that word in 7 words or less."
        )
        return hashlib.sha256(prompt_template.encode()).hexdigest()[:16]
    
    def _is_cache_valid(self, cache_data: Dict) -> bool:
        """Validate cache compatibility"""
        try:
            metadata = cache_data.get("metadata", {})
            
            # Check cache version
            if metadata.get("cache_version") != "1.0":
                return False
            
            # Check if prompt has changed significantly
            cached_hash = metadata.get("prompt_hash", "")
            if cached_hash and cached_hash != self.current_prompt_hash:
                quick_log(self.secret_word, f"⚠️ Prompt changed - hash mismatch")
                return False
            
            # Check if model changed
            if metadata.get("model_used") != Config.OPENAI_MODEL:
                quick_log(self.secret_word, f"⚠️ Model changed from {metadata.get('model_used')} to {Config.OPENAI_MODEL}")
                return False
            
            return True
            
        except (KeyError, TypeError):
            return False
    
    def lookup_batch(self, words_with_ranks: List[Tuple[str, int]]) -> Tuple[Dict[str, Dict], List[Tuple[str, int]]]:
        """
        Batch lookup: return cached results and list of words needing API calls
        
        Args:
            words_with_ranks: List of (word, rank) tuples
            
        Returns:
            Tuple of (cached_results, words_needing_api_calls)
        """
        cached_results = {}
        api_needed = []
        
        for word, rank in words_with_ranks:
            word_lower = word.lower()
            
            # Check specific cache first
            if word_lower in self.cache["clues"]:
                cached_entry = self.cache["clues"][word_lower].copy()
                cached_entry["source"] = "cache"
                cached_results[word] = cached_entry
                self.hits += 1
                continue
            
            # Check generic cache for universally weak words
            if word_lower in self.generic_cache["clues"]:
                cached_entry = self.generic_cache["clues"][word_lower].copy()
                cached_entry["source"] = "generic_cache"
                cached_results[word] = cached_entry
                self.hits += 1
                continue
            
            # Not found in any cache
            api_needed.append((word, rank))
            self.misses += 1
        
        if cached_results:
            quick_log(self.secret_word, f"💾 Cache hit: {len(cached_results):,} words | API needed: {len(api_needed):,} words")
        
        return cached_results, api_needed
    
    def store_batch(self, api_results: Dict[str, Dict[str, Any]], words_with_ranks: List[Tuple[str, int]]):
        """Store API results in cache"""
        if not api_results:
            return
        
        current_time = datetime.now(timezone.utc).isoformat()
        word_rank_map = {word: rank for word, rank in words_with_ranks}
        
        for word, result in api_results.items():
            word_lower = word.lower()
            
            # Determine if this should go in generic cache (universally weak words)
            is_generic = (
                result.get('strength') == 'weak' and 
                word_rank_map.get(word, 0) > 10000 and
                self._is_generic_word(word_lower)
            )
            
            cache_entry = {
                "clue": result.get('clue', ''),
                "strength": result.get('strength', 'medium'),
                "rank": word_rank_map.get(word, 0),
                "cached_at": current_time,
                "violations": result.get('violations', []),
                "original_clue": result.get('original_clue', result.get('clue', ''))
            }
            
            if is_generic:
                self.generic_cache["clues"][word_lower] = cache_entry
            else:
                self.cache["clues"][word_lower] = cache_entry
        
        # Update metadata
        self.cache["metadata"]["last_updated"] = current_time
        self.cache["metadata"]["total_cached_clues"] = len(self.cache["clues"])
        self.cache["metadata"]["api_calls_saved"] = self.api_calls_saved
        
        # Auto-save every 50 new entries or at end of batch
        if len(api_results) >= 10 or self.misses % 50 == 0:
            self.save_cache()
    
    def _is_generic_word(self, word: str) -> bool:
        """Check if word is universally weak (good for generic cache)"""
        generic_indicators = [
            'spoon', 'zipper', 'stapler', 'paperclip', 'calculator', 
            'telephone', 'computer', 'keyboard', 'mouse', 'printer',
            'television', 'radio', 'camera', 'clock', 'lamp'
        ]
        return word in generic_indicators or len(word) > 12  # Very long words tend to be generic
    
    def save_cache(self):
        """Save cache to disk"""
        try:
            # Ensure directory exists
            Config.ensure_directories()
            
            # Save main cache
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
            
            # Save generic cache if it has entries
            if self.generic_cache["clues"]:
                with open(self.generic_cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.generic_cache, f, indent=2, ensure_ascii=False)
            
            quick_log(self.secret_word, f"💾 Cache saved: {len(self.cache['clues']):,} entries")
            
        except Exception as e:
            quick_log(self.secret_word, f"❌ Failed to save cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Return comprehensive cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        # Calculate estimated savings
        estimated_time_saved_minutes = self.hits * 2.5 / 60  # 2.5s per API call
        estimated_cost_saved = self.hits * 0.002  # ~$0.002 per API call
        
        # Get cache file size
        cache_size_mb = 0
        if self.cache_file.exists():
            cache_size_mb = self.cache_file.stat().st_size / 1024 / 1024
        
        return {
            "total_entries": len(self.cache["clues"]),
            "generic_entries": len(self.generic_cache["clues"]),
            "session_hits": self.hits,
            "session_misses": self.misses,
            "hit_rate": hit_rate,
            "estimated_time_saved_minutes": estimated_time_saved_minutes,
            "estimated_cost_saved_usd": estimated_cost_saved,
            "cache_size_mb": cache_size_mb,
            "cache_age_days": self._get_cache_age_days(),
            "api_calls_saved_total": self.cache["metadata"]["api_calls_saved"]
        }
    
    def _get_cache_age_days(self) -> int:
        """Get cache age in days"""
        try:
            created_str = self.cache["metadata"]["created_at"]
            created_dt = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
            return (datetime.now(timezone.utc) - created_dt).days
        except (KeyError, ValueError):
            return 0
    
    def cleanup_old_entries(self, max_age_days: int = 30):
        """Remove entries older than specified days"""
        if max_age_days <= 0:
            return
        
        cutoff_date = datetime.now(timezone.utc).timestamp() - (max_age_days * 24 * 60 * 60)
        removed_count = 0
        
        # Clean main cache
        entries_to_remove = []
        for word, entry in self.cache["clues"].items():
            try:
                cached_at = datetime.fromisoformat(entry["cached_at"].replace('Z', '+00:00'))
                if cached_at.timestamp() < cutoff_date:
                    entries_to_remove.append(word)
            except (KeyError, ValueError):
                entries_to_remove.append(word)  # Remove malformed entries
        
        for word in entries_to_remove:
            del self.cache["clues"][word]
            removed_count += 1
        
        if removed_count > 0:
            quick_log(self.secret_word, f"🧹 Cleaned {removed_count:,} old cache entries")
            self.save_cache()


def create_cache_stats_report(secret_words: List[str]) -> Dict[str, Any]:
    """Generate comprehensive cache statistics across all secret words"""
    total_stats = {
        "total_cache_files": 0,
        "total_cached_entries": 0,
        "total_size_mb": 0,
        "oldest_cache_days": 0,
        "newest_cache_days": float('inf'),
        "word_stats": {}
    }
    
    for word in secret_words:
        cache = ClueCache(word)
        stats = cache.get_stats()
        
        total_stats["total_cache_files"] += 1
        total_stats["total_cached_entries"] += stats["total_entries"]
        total_stats["total_size_mb"] += stats["cache_size_mb"]
        total_stats["oldest_cache_days"] = max(total_stats["oldest_cache_days"], stats["cache_age_days"])
        total_stats["newest_cache_days"] = min(total_stats["newest_cache_days"], stats["cache_age_days"])
        total_stats["word_stats"][word] = stats
    
    return total_stats


if __name__ == "__main__":
    # Test cache system
    print("=== ClueCache Test ===")
    
    # Test with fish
    cache = ClueCache("fish")
    print(f"Cache loaded for 'fish': {len(cache.cache['clues'])} entries")
    
    # Test batch lookup
    test_words = [("salmon", 34), ("tuna", 45), ("spoon", 50000)]
    cached, api_needed = cache.lookup_batch(test_words)
    
    print(f"Cached results: {len(cached)}")
    print(f"API needed: {len(api_needed)}")
    
    # Test storing results
    if api_needed:
        fake_results = {
            word: {
                "clue": f"I am related to {word}",
                "strength": "weak" if rank > 10000 else "medium",
                "violations": []
            }
            for word, rank in api_needed
        }
        cache.store_batch(fake_results, api_needed)
    
    # Show stats
    stats = cache.get_stats()
    print(f"\nCache Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Cache system test completed!")
