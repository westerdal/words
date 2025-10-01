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


def deduplicate_preserve_order(words: List[str]) -> List[str]:
    """
    Remove duplicates while preserving order.
    First occurrence of each word is kept.
    """
    seen = set()
    result = []
    
    for word in words:
        if word and word not in seen:
            seen.add(word)
            result.append(word)
    
    return result


def contains_seed_word(word: str, seed_word: str) -> bool:
    """
    Check if word contains seed_word as a contiguous substring.
    Example: if seed = "queen", exclude "queenly"
    """
    if not word or not seed_word:
        return False
    
    return seed_word.lower() in word.lower()


def filter_words_comprehensive(words: List[str], seed_word: str) -> List[str]:
    """
    Apply comprehensive filtering rules for two-pass expansion:
    - Clean each word according to strict rules
    - Remove words containing seed word as substring
    - Remove generic words from filter list
    - Remove proper nouns (basic detection)
    - Deduplicate while preserving order
    """
    filtered_words = []
    
    for word in words:
        # Clean the word
        cleaned = clean_word(word)
        if not cleaned:
            continue
        
        # Skip if contains seed word
        if contains_seed_word(cleaned, seed_word):
            continue
        
        # Skip generic words
        if cleaned in Config.EXPANSION_GENERIC_FILTER:
            continue
        
        # Basic proper noun detection (starts with capital in original)
        if word and word[0].isupper() and word.lower() != word:
            continue
        
        # Skip very short words (likely not meaningful)
        if len(cleaned) < 2:
            continue
        
        filtered_words.append(cleaned)
    
    # Deduplicate while preserving order
    return deduplicate_preserve_order(filtered_words)


class OpenAISimilarWords:
    """Handles OpenAI similar words retrieval and caching"""
    
    def __init__(self, secret_word: str):
        self.secret_word = secret_word.lower().strip()
        
        # Validate word - temporarily disabled due to import issue
        # valid, result = Config.validate_word(self.secret_word)
        # if not valid:
        #     raise ValueError(f"Invalid secret word: {result}")
        
        # Simple validation
        if not self.secret_word or not self.secret_word.isalpha() or len(self.secret_word) < 2:
            raise ValueError(f"Invalid secret word: {self.secret_word}")
        
        self.secret_word = self.secret_word
        
        # Cache file path
        from pathlib import Path
        secretword_dir = Path("secretword")
        self.cache_file = secretword_dir / f"openai-{self.secret_word}.txt"
        
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
            
            try:
                words = self._try_openai_request(prompt, strategy['max_tokens'])
                if words:
                    quick_log(self.secret_word, f"✅ {strategy['name']} successful: got {len(words)} words")
                    return words
                else:
                    quick_log(self.secret_word, f"❌ {strategy['name']} failed, trying next strategy...")
            except Exception as e:
                # Connection/API errors should abort immediately
                quick_log(self.secret_word, f"🛑 ABORTING: {e}")
                raise Exception(f"openai_failure")  # Special marker for run_csv_prompt.py
        
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
            
        except openai.APIConnectionError as e:
            quick_log(self.secret_word, f"❌ FATAL: OpenAI connection error - {e}")
            raise Exception(f"OpenAI connection failed: {e}")
        except openai.APIError as e:
            quick_log(self.secret_word, f"❌ FATAL: OpenAI API error - {e}")
            raise Exception(f"OpenAI API error: {e}")
        except openai.RateLimitError as e:
            quick_log(self.secret_word, f"❌ FATAL: OpenAI rate limit exceeded - {e}")
            raise Exception(f"OpenAI rate limit exceeded: {e}")
        except Exception as e:
            quick_log(self.secret_word, f"❌ FATAL: Unexpected OpenAI error - {e}")
            raise Exception(f"OpenAI request failed: {e}")
    
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
                
                # Store new words for later addition to ENABLE2.txt (after plural conversion)
                self.new_words_to_add = new_from_openai
                
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
    
    def update_enable2_with_final_words(self, final_words: List[str]) -> bool:
        """Update ENABLE2.txt with final processed words (after plural conversion)"""
        if not hasattr(self, 'new_words_to_add') or not self.new_words_to_add:
            return True
        
        # Filter new words to only include singular forms that are still in final_words
        from plural_converter import pluralize_to_singular
        
        final_new_words = []
        for orig_word in self.new_words_to_add:
            singular_word = pluralize_to_singular(orig_word)
            if singular_word in final_words and singular_word != orig_word:
                # Only add the singular form if it's different from original
                final_new_words.append(singular_word)
            elif orig_word in final_words:
                # Add original if it's not plural or wasn't converted
                final_new_words.append(orig_word)
        
        if final_new_words:
            if self._add_words_to_enable2(final_new_words):
                quick_log(self.secret_word, f"✅ Added {len(final_new_words)} processed words to ENABLE2.txt (post-conversion)")
                return True
            else:
                quick_log(self.secret_word, f"⚠️ WARNING: Failed to add processed words to ENABLE2.txt")
                return False
        return True
    
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
    
    def get_primary_associations(self) -> List[str]:
        """
        First pass: Get at least 300 direct semantic associations from OpenAI
        """
        if not self.ai_available:
            quick_log(self.secret_word, "❌ Cannot get primary associations - API key not available")
            return []
        
        prompt = f'''You are given a single seed word: "{self.secret_word}".

Task: Generate AT LEAST {Config.PRIMARY_MIN_WORDS} unique single dictionary words that are semantically associated with the seed word, ordered from closest (most relevant) to farthest (most tangential).

Hard rules (must follow exactly):
- Output text only (no images, no extra commentary).  
- Return a single comma-separated list of words and nothing else.  
- Use only valid dictionary words (common dictionary entries). No slang, abbreviations, invented words, acronyms, or brand names.  
- Do NOT include proper nouns (people, place names, brands, etc.). If an association is normally a proper noun, convert to a neutral common-noun equivalent or omit.  
- Use only the 26 lowercase English letters (a–z). Convert accented letters to ASCII (e.g., arête → arete). Remove all punctuation, hyphens, spaces, digits, and any non-letter characters.  
- Only single words allowed (no spaces). If the best association is a phrase, choose the single most representative single word from that phrase (e.g., "queen termite" → "termite").  
- Do NOT include the seed word or any word that contains the seed word as a contiguous substring (e.g., if seed = "queen", exclude "queenly").  
- No duplicates. Generate MORE than {Config.PRIMARY_MIN_WORDS} if you can find valid associative words.

Generation procedure:
1. PRIMARY PASS — Generate direct, high-confidence associations: synonyms, near-synonyms, antonyms, hypernyms, hyponyms, direct objects, roles, places, tools, animals, verbs, adjectives, and immediate contextual words tied to the seed.
2. Normalization & filtering — For each candidate, normalize to lowercase ASCII letters, drop words violating the hard rules, and prefer singular/common forms where sensible.
3. Scoring & ordering — Assign an implicit proximity/confidence score to each remaining candidate (how closely it relates to the seed). Sort the list from highest to lowest proximity. When proximity is similar, prefer more common/obvious words earlier.
4. Deduplicate

Return as comma-separated list only.'''

        quick_log(self.secret_word, f"🤖 Getting primary associations (minimum {Config.PRIMARY_MIN_WORDS} words)")
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4",  # Use GPT-4 for better quality
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
                temperature=0.1  # Lower temperature for consistency
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse comma-separated response
            raw_words = [w.strip() for w in content.split(',') if w.strip()]
            
            # Apply comprehensive filtering and deduplication
            filtered_words = filter_words_comprehensive(raw_words, self.secret_word)
            
            quick_log(self.secret_word, f"✅ Primary pass: {len(raw_words)} raw → {len(filtered_words)} filtered words")
            
            if len(filtered_words) < Config.PRIMARY_MIN_WORDS:
                quick_log(self.secret_word, f"⚠️ WARNING: Got {len(filtered_words)} words, expected minimum {Config.PRIMARY_MIN_WORDS}")
            
            return filtered_words
            
        except Exception as e:
            quick_log(self.secret_word, f"❌ ERROR: Primary associations failed: {e}")
            return []
    
    def get_synonym_expansion(self, primary_words: List[str]) -> List[str]:
        """
        Second pass: Get synonyms and near-synonyms for all primary words
        """
        if not self.ai_available or not primary_words:
            return []
        
        # Create batches to avoid token limits
        batch_size = 50  # Process 50 words at a time
        all_synonyms = []
        
        for i in range(0, len(primary_words), batch_size):
            batch = primary_words[i:i + batch_size]
            batch_words = ', '.join(batch)
            
            prompt = f'''Given this list of words: {batch_words}

For EACH word in this list, generate 3-8 synonyms and near-synonyms. Follow these hard rules:
- Output text only, return a single comma-separated list of words and nothing else
- Use only valid dictionary words (common dictionary entries). No slang, abbreviations, invented words, acronyms, or brand names
- Do NOT include proper nouns (people, place names, brands, etc.)
- Use only the 26 lowercase English letters (a–z). Convert accented letters to ASCII
- Only single words allowed (no spaces)
- Do NOT include the original seed word "{self.secret_word}" or any word that contains it as a substring
- No duplicates
- Return ALL synonym results as a single comma-separated list

Generate synonyms for: {batch_words}'''

            quick_log(self.secret_word, f"🤖 Getting synonyms for batch {i//batch_size + 1} ({len(batch)} words)")
            
            try:
                response = openai.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=3000,
                    temperature=0.1
                )
                
                content = response.choices[0].message.content.strip()
                
                # Parse comma-separated response
                batch_synonyms = [w.strip() for w in content.split(',') if w.strip()]
                
                # Apply comprehensive filtering and deduplication
                filtered_synonyms = filter_words_comprehensive(batch_synonyms, self.secret_word)
                
                all_synonyms.extend(filtered_synonyms)
                
                quick_log(self.secret_word, f"✅ Batch {i//batch_size + 1}: {len(batch_synonyms)} raw → {len(filtered_synonyms)} filtered synonyms")
                
            except Exception as e:
                quick_log(self.secret_word, f"❌ ERROR: Synonym batch {i//batch_size + 1} failed: {e}")
                continue
        
        # Final deduplication of all synonyms
        unique_synonyms = deduplicate_preserve_order(all_synonyms)
        
        quick_log(self.secret_word, f"✅ Synonym expansion: {len(all_synonyms)} total → {len(unique_synonyms)} unique synonyms")
        
        return unique_synonyms
    
    def get_two_pass_expansion(self) -> List[str]:
        """
        Perform complete three-method expansion: 
        1. Primary associations (detailed contextual relationships)
        2. Simplified direct associations (broad coverage)
        3. Synonym expansion (comprehensive expansion of all discovered words)
        """
        quick_log(self.secret_word, f"🚀 Starting enhanced three-method expansion for '{self.secret_word}'")
        
        # Method 1: Get primary associations (detailed contextual relationships)
        primary_words = self.get_primary_associations()
        if not primary_words:
            quick_log(self.secret_word, "⚠️ No primary associations found, continuing with other methods")
            primary_words = []
        else:
            quick_log(self.secret_word, f"✅ Method 1 completed: {len(primary_words)} primary words")
        
        # Method 2: Get simplified direct associations (broad coverage)
        simplified_words = self.get_simplified_associations()
        quick_log(self.secret_word, f"✅ Method 2 completed: {len(simplified_words)} simplified words")
        
        # Combine primary and simplified words
        combined_words = primary_words + simplified_words
        
        # Deduplicate combined words before synonym expansion
        combined_words = deduplicate_preserve_order(combined_words)
        quick_log(self.secret_word, f"📋 After combining methods 1+2: {len(combined_words)} unique words")
        
        # Method 3: Get synonyms for all discovered words
        synonym_words = self.get_synonym_expansion(combined_words)
        quick_log(self.secret_word, f"✅ Method 3 completed: {len(synonym_words)} synonym words")
        
        # Final combination: primary + simplified + synonyms
        all_words = combined_words + synonym_words
        
        # Final deduplication while preserving order (primary words ranked highest)
        final_words = deduplicate_preserve_order(all_words)
        
        quick_log(self.secret_word, f"✅ Three-method expansion completed!")
        quick_log(self.secret_word, f"📊 Results: {len(primary_words)} primary + {len(simplified_words)} simplified + {len(synonym_words)} synonyms = {len(final_words)} unique total")
        
        return final_words
    
    def get_simplified_associations(self) -> List[str]:
        """
        Get simplified direct associations using single-prompt method
        This provides broad coverage with compound word explosion
        """
        if not self.ai_available:
            quick_log(self.secret_word, "⚠️ Cannot get simplified associations - API key not available")
            return []
        
        prompt = f"""List 200 words and phrases similar to, related to, or associated with '{self.secret_word}'. 
Include synonyms, related objects, actions, qualities, categories, and contextual terms.
Return as a comma-separated list only, no numbers or formatting.

Example format: word1, word2, phrase with spaces, hyphenated-term, word3, etc."""

        try:
            quick_log(self.secret_word, f"🔍 Getting simplified associations...")
            
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3
            )
            
            content = response.choices[0].message.content.strip()
            
            # Split by commas and process each word/phrase
            raw_words = [w.strip() for w in content.split(',') if w.strip()]
            
            # Explode compound words and clean
            exploded_words = []
            for word_phrase in raw_words:
                # Split on spaces and hyphens to explode compound terms
                parts = re.split(r'[\s\-_]+', word_phrase)
                for part in parts:
                    cleaned = clean_word(part)
                    if cleaned:
                        exploded_words.append(cleaned)
            
            # Deduplicate while preserving order
            unique_words = deduplicate_preserve_order(exploded_words)
            
            if unique_words:
                quick_log(self.secret_word, f"📊 Simplified: {len(raw_words)} phrases → {len(exploded_words)} exploded → {len(unique_words)} unique words")
                quick_log(self.secret_word, f"   Sample: {', '.join(unique_words[:10])}")
            
            return unique_words
            
        except Exception as e:
            quick_log(self.secret_word, f"⚠️ WARNING: Simplified associations failed: {e}")
            return []
    
    def get_similar_words(self) -> List[str]:
        """Get similar words with two-pass expansion and caching"""
        quick_log(self.secret_word, f"🔍 Getting OpenAI similar words for '{self.secret_word}'")
        
        # Check for two-pass expansion cache first
        two_pass_cache_file = Config.SECRETWORD_DIR / Config.get_openai_twopass_filename(self.secret_word)
        if two_pass_cache_file.exists():
            quick_log(self.secret_word, f"📂 Loading two-pass expansion cache")
            try:
                with open(two_pass_cache_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                    # Find the line with comma-separated words (skip header comments)
                    word_line = None
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            word_line = line
                            break
                    
                    if word_line:
                        # Parse comma-separated words
                        cached_words = [w.strip() for w in word_line.split(',') if w.strip()]
                        if cached_words:
                            quick_log(self.secret_word, f"✅ Loaded {len(cached_words)} words from two-pass cache")
                            return self._validate_against_embeddings(cached_words)
            except Exception as e:
                quick_log(self.secret_word, f"⚠️ Failed to load two-pass cache: {e}")
        
        # Perform two-pass expansion
        expanded_words = self.get_two_pass_expansion()
        
        if expanded_words:
            # Save to two-pass cache
            self.save_two_pass_cache(expanded_words)
            
            # Validate the expanded set
            return self._validate_against_embeddings(expanded_words)
        else:
            quick_log(self.secret_word, "❌ Two-pass expansion failed")
            return []
    
    def save_two_pass_cache(self, words: List[str]) -> bool:
        """Save two-pass expansion results to cache as comma-separated list"""
        if not words:
            return False
        
        try:
            Config.ensure_directories()
            
            two_pass_cache_file = Config.SECRETWORD_DIR / Config.get_openai_twopass_filename(self.secret_word)
            
            with open(two_pass_cache_file, 'w', encoding='utf-8') as f:
                # Write header comment
                f.write(f"# Two-pass expansion for '{self.secret_word}'\n")
                f.write(f"# Total words: {len(words)}\n")
                f.write(f"# Generated: {__import__('datetime').datetime.now().isoformat()}\n\n")
                
                # Write comma-separated words
                f.write(', '.join(words))
            
            quick_log(self.secret_word, f"💾 Saved {len(words)} words to two-pass cache")
            return True
            
        except Exception as e:
            quick_log(self.secret_word, f"❌ ERROR: Failed to save two-pass cache: {e}")
            return False

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
            print(f"\nSUCCESS: Got {len(words)} similar words for '{secret_word}'!")
            print("Top 10:")
            for i, word in enumerate(words[:10], 1):
                print(f"  {i}. {word}")
        else:
            print(f"\n💥 Failed to get similar words for '{secret_word}'")
            sys.exit(1)
            
    except Exception as e:
        try:
            print(f"\n💥 Error: {e}")
        except UnicodeEncodeError:
            print(f"\nERROR: {e}")
        sys.exit(1)
