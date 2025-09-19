#!/usr/bin/env python3
"""
OpenAI Similar Words Module
Gets top 250 words similar to a secret word from OpenAI and caches results
"""

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple
import openai

try:
    from .config import Config
    from .progress_tracker import quick_log
    from .word_utils import clean_word
except ImportError:
    # For standalone execution
    import sys
    sys.path.append(str(Path(__file__).parent))
    from config import Config
    from progress_tracker import quick_log
    from word_utils import clean_word

def clean_word(word: str) -> Optional[str]:
    """
    Clean a word according to strict rules:
    - Keep only single words (no spaces, no hyphens)
    - Remove words with numbers, punctuation, or non-ASCII characters
    - Remove words with brackets (drop entire word)
    - Convert to lowercase
    - Return None if word should be dropped
    """
    if not word or not isinstance(word, str):
        return None
    
    # Remove any leading/trailing whitespace
    word = word.strip()
    
    # Drop if empty
    if not word:
        return None
    
    # Drop if contains brackets
    if '[' in word or ']' in word or '(' in word or ')' in word or '{' in word or '}' in word:
        return None
    
    # Drop if contains spaces or hyphens (not single words)
    if ' ' in word or '-' in word or '_' in word:
        return None
    
    # Drop if contains numbers
    if any(char.isdigit() for char in word):
        return None
    
    # Drop if contains punctuation (except letters)
    if not word.isalpha():
        return None
    
    # Convert to lowercase
    word = word.lower()
    
    # Check for non-ASCII characters and try to normalize
    try:
        # Normalize unicode characters (é → e)
        normalized = unicodedata.normalize('NFD', word)
        ascii_word = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
        
        # If still contains non-ASCII, drop it
        if not ascii_word.isascii():
            return None
        
        word = ascii_word
        
    except Exception:
        return None
    
    # Final check - must be alphabetic and at least 2 characters
    if not word.isalpha() or len(word) < 2:
        return None
    
    return word

class OpenAISimilarWords:
    """Handles OpenAI similar words retrieval and caching"""
    
    def __init__(self, secret_word: str):
        self.secret_word = secret_word.lower().strip()
        
        # Validate word
        valid, result = Config.validate_word(self.secret_word)
        if not valid:
            raise ValueError(f"Invalid secret word: {result}")
        
        self.secret_word = result
        
        # Cache file path
        self.cache_file = Config.SECRETWORD_DIR / f"openai-{self.secret_word}.txt"
        
        # Initialize OpenAI
        if Config.check_openai_key():
            openai.api_key = os.getenv('OPENAI_API_KEY')
            self.ai_available = True
        else:
            self.ai_available = False
            quick_log(self.secret_word, "⚠️ WARNING: OpenAI API key not available")
    
    def load_cached_words(self) -> Optional[List[str]]:
        """Load cached similar words if they exist"""
        if not self.cache_file.exists():
            return None
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Skip header and extract words
            words = []
            for line in lines[3:]:  # Skip header lines
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract word from numbered list format "123. word"
                    match = re.match(r'^\d+\.\s+(.+)$', line)
                    if match:
                        words.append(match.group(1).lower())
            
            quick_log(self.secret_word, f"✅ Loaded {len(words)} cached OpenAI words from {self.cache_file}")
            return words
            
        except Exception as e:
            quick_log(self.secret_word, f"⚠️ WARNING: Failed to load cached words: {e}")
            return None
    
    def get_openai_similar_words(self) -> List[str]:
        """Get similar words from OpenAI with fallback strategy"""
        if not self.ai_available:
            quick_log(self.secret_word, "❌ Cannot get OpenAI words - API key not available")
            return []
        
        # Try different strategies in order of preference
        strategies = [
            {
                'name': 'full_request',
                'max_words': 1000,
                'max_tokens': 4000,
                'prompt_template': """List words similar to "{word}". Include synonyms, related concepts, associated objects/actions/qualities, and contextually related terms. Order by similarity (most similar first). Format: numbered list only.

1. word1
2. word2
etc.

Provide as many as possible up to {max_words} words maximum."""
            },
            {
                'name': 'medium_request', 
                'max_words': 500,
                'max_tokens': 2000,
                'prompt_template': """List {max_words} words similar to "{word}". Order by similarity. Format: numbered list only.

1. word1
2. word2
etc."""
            },
            {
                'name': 'small_request',
                'max_words': 250,
                'max_tokens': 1000,
                'prompt_template': """List {max_words} words similar to "{word}". Numbered list only."""
            }
        ]
        
        for strategy in strategies:
            quick_log(self.secret_word, f"🤖 Trying {strategy['name']}: max {strategy['max_words']} words...")
            
            prompt = strategy['prompt_template'].format(
                word=self.secret_word,
                max_words=strategy['max_words']
            )
            
            words = self._try_openai_request(prompt, strategy['max_tokens'])
            if words:
                quick_log(self.secret_word, f"✅ {strategy['name']} successful: got {len(words)} words")
                return words
            else:
                quick_log(self.secret_word, f"❌ {strategy['name']} failed, trying next strategy...")
        
        quick_log(self.secret_word, "❌ All OpenAI strategies failed")
        return []
    
    def _try_openai_request(self, prompt: str, max_tokens: int) -> List[str]:
        """Try a single OpenAI request with given parameters"""
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",  # Use GPT-3.5-turbo for higher token limits and lower cost
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3  # Lower temperature for consistency
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse and clean the response
            return self._parse_and_clean_response(content)
            
        except Exception as e:
            return []  # Return empty list on error
    
    def _parse_and_clean_response(self, content: str) -> List[str]:
        """Parse OpenAI response and clean words"""
        words = []
        raw_words = []
        cleaned_count = 0
        dropped_count = 0
        
        for line in content.split('\n'):
            line = line.strip()
            if line:
                # Extract word from numbered format "123. word"
                match = re.match(r'^\d+\.\s+(.+)$', line)
                if match:
                    raw_word = match.group(1).strip()
                    raw_words.append(raw_word)
                    
                    # Clean the word
                    cleaned_word = clean_word(raw_word)
                    if cleaned_word:
                        words.append(cleaned_word)
                        cleaned_count += 1
                    else:
                        dropped_count += 1
        
        if raw_words:  # Only log if we found some words
            quick_log(self.secret_word, f"📋 Cleaning: {len(raw_words)} raw → {cleaned_count} clean → {dropped_count} dropped")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_words = []
        for word in words:
            if word not in seen:
                seen.add(word)
                unique_words.append(word)
        
        if len(unique_words) != len(words):
            duplicates_removed = len(words) - len(unique_words)
            quick_log(self.secret_word, f"📋 Removed {duplicates_removed} duplicates")
        
        return unique_words
    
    def _validate_against_embeddings(self, words: List[str]) -> List[str]:
        """Check OpenAI words against ENABLE2 and report which are new additions"""
        if not words:
            return []
        
        quick_log(self.secret_word, f"🔍 Analyzing {len(words)} OpenAI words against ENABLE2 dataset...")
        
        # Use ENABLE2 word list for comparison
        enable2_file = Config.ENABLE2_FILE
        if not enable2_file.exists():
            quick_log(self.secret_word, f"⚠️ WARNING: enable2.txt not found at {enable2_file}")
            quick_log(self.secret_word, f"   Keeping all OpenAI words without comparison")
            return words
        
        try:
            # Load ENABLE2 word list
            with open(enable2_file, 'r', encoding='utf-8') as f:
                enable2_words = set(word.strip().lower() for word in f.readlines() if word.strip())
            
            quick_log(self.secret_word, f"✅ Loaded {len(enable2_words):,} words from ENABLE2 list")
            
            # Categorize words
            in_enable2 = []
            new_from_openai = []
            
            for word in words:
                if word in enable2_words:
                    in_enable2.append(word)
                else:
                    new_from_openai.append(word)
            
            # Report results
            quick_log(self.secret_word, f"📊 Word analysis results:")
            quick_log(self.secret_word, f"   Words in ENABLE2: {len(in_enable2)}")
            quick_log(self.secret_word, f"   New words from OpenAI: {len(new_from_openai)}")
            
            if new_from_openai:
                if len(new_from_openai) <= 10:
                    quick_log(self.secret_word, f"   New OpenAI words: {', '.join(new_from_openai)}")
                else:
                    quick_log(self.secret_word, f"   New OpenAI words (first 10): {', '.join(new_from_openai[:10])}")
                
                # Add new words to enable2.txt
                if self._add_words_to_enable2(new_from_openai):
                    quick_log(self.secret_word, f"✅ Added {len(new_from_openai)} new words to ENABLE2.txt")
                else:
                    quick_log(self.secret_word, f"⚠️ WARNING: Failed to add new words to ENABLE2.txt")
                
                quick_log(self.secret_word, f"✅ Keeping all {len(words)} words (including {len(new_from_openai)} new from OpenAI)")
            else:
                quick_log(self.secret_word, f"✅ All OpenAI words are in ENABLE2 dataset")
            
            # Return ALL words (both existing and new)
            return words
            
        except Exception as e:
            quick_log(self.secret_word, f"❌ ERROR: Failed to load ENABLE2 for comparison: {e}")
            quick_log(self.secret_word, f"   Keeping all OpenAI words without comparison")
            return words
    
    def _add_words_to_enable2(self, new_words: List[str]) -> bool:
        """Add new words to enable2.txt file"""
        if not new_words:
            return True
        
        try:
            # Read current enable2.txt to avoid duplicates
            current_words = set()
            if Config.ENABLE2_FILE.exists():
                with open(Config.ENABLE2_FILE, 'r', encoding='utf-8') as f:
                    current_words = set(word.strip().lower() for word in f.readlines() if word.strip())
            
            # Filter out words that are already in the file
            words_to_add = [word for word in new_words if word not in current_words]
            
            if not words_to_add:
                quick_log(self.secret_word, f"📝 All new words already in ENABLE2.txt")
                return True
            
            # Append new words to enable2.txt
            with open(Config.ENABLE2_FILE, 'a', encoding='utf-8') as f:
                for word in words_to_add:
                    f.write(f"{word}\n")
            
            # Update the word count
            total_words = len(current_words) + len(words_to_add)
            quick_log(self.secret_word, f"📝 ENABLE2.txt updated: {len(current_words):,} → {total_words:,} words (+{len(words_to_add)})")
            
            return True
            
        except Exception as e:
            quick_log(self.secret_word, f"❌ ERROR: Failed to update ENABLE2.txt: {e}")
            return False
    
    def save_words_to_cache(self, words: List[str]) -> bool:
        """Save words to cache file"""
        if not words:
            return False
        
        try:
            Config.ensure_directories()
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                f.write(f"# OpenAI Similar Words for '{self.secret_word}'\n")
                f.write(f"# Generated: {Config.get_progress_filename(self.secret_word).parent}\n")
                f.write(f"# Total words: {len(words)}\n\n")
                
                for i, word in enumerate(words, 1):
                    f.write(f"{i}. {word}\n")
            
            quick_log(self.secret_word, f"💾 Saved {len(words)} OpenAI words to {self.cache_file}")
            return True
            
        except Exception as e:
            quick_log(self.secret_word, f"❌ ERROR: Failed to save cache: {e}")
            return False
    
    def get_similar_words(self) -> List[str]:
        """Get similar words with recursive expansion, validation and caching"""
        quick_log(self.secret_word, f"🔍 Getting OpenAI similar words for '{self.secret_word}'")
        
        # Check for expanded cache first
        expanded_cache_file = Config.SECRETWORD_DIR / f"openai-{self.secret_word}-expanded.txt"
        if expanded_cache_file.exists():
            quick_log(self.secret_word, f"📂 Loading recursive expansion cache")
            try:
                expanded_words = []
                with open(expanded_cache_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '(' in line:
                            word = line.split('(')[0].strip()
                            if word:
                                expanded_words.append(word)
                
                if expanded_words:
                    quick_log(self.secret_word, f"✅ Loaded {len(expanded_words)} words from recursive expansion cache")
                    return self._validate_against_embeddings(expanded_words)
            except Exception as e:
                quick_log(self.secret_word, f"⚠️ Failed to load expansion cache: {e}")
        
        # Try to load from regular cache
        cached_words = self.load_cached_words()
        if cached_words:
            initial_words = cached_words
        else:
            # Get from OpenAI if not cached
            initial_words = self.get_openai_similar_words()
            if not initial_words:
                quick_log(self.secret_word, "❌ No initial words retrieved from OpenAI")
                return []
            
            # Cache the initial results
            self.save_words_to_cache(initial_words)
        
        # Perform recursive expansion (import dynamically to avoid circular import)
        quick_log(self.secret_word, f"🔄 Starting recursive expansion with {len(initial_words)} initial words")
        try:
            from .recursive_expansion import create_recursive_expansion
        except ImportError:
            from recursive_expansion import create_recursive_expansion
        
        expanded_words = create_recursive_expansion(self.secret_word, initial_words)
        
        if expanded_words and len(expanded_words) > len(initial_words):
            quick_log(self.secret_word, f"✅ Recursive expansion: {len(initial_words)} → {len(expanded_words)} words")
            # Validate the expanded set
            return self._validate_against_embeddings(expanded_words)
        else:
            quick_log(self.secret_word, f"⚠️ Recursive expansion failed, using initial words")
            # Validate initial words
            return self._validate_against_embeddings(initial_words)

def get_openai_similar_words(secret_word: str) -> List[str]:
    """Convenience function to get OpenAI similar words"""
    try:
        module = OpenAISimilarWords(secret_word)
        return module.get_similar_words()
    except Exception as e:
        quick_log(secret_word, f"❌ ERROR in OpenAI similar words module: {e}")
        return []

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python openai_similar_words.py <secret_word>")
        print("Example: python openai_similar_words.py forest")
        print("Note: Requests 2000 similar words from OpenAI")
        sys.exit(1)
    
    secret_word = sys.argv[1]
    
    try:
        module = OpenAISimilarWords(secret_word)
        words = module.get_similar_words()
        
        if words:
            print(f"\n🎉 Got {len(words)} similar words for '{secret_word}'!")
            print("Top 10:")
            for i, word in enumerate(words[:10], 1):
                print(f"  {i}. {word}")
        else:
            print(f"\n💥 Failed to get similar words for '{secret_word}'")
            
    except Exception as e:
        print(f"\n💥 Error: {e}")
        sys.exit(1)
