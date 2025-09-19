#!/usr/bin/env python3
"""
Recursive Semantic Word Expansion Module
Creates multi-level semantic networks by recursively finding similar words
"""

import openai
import json
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass

try:
    from .config import Config
    from .progress_tracker import quick_log
    from .plural_converter import convert_plurals_to_singular
    from .word_utils import clean_word, is_generic_word
except ImportError:
    # For standalone execution
    import sys
    sys.path.append(str(Path(__file__).parent))
    from config import Config
    from progress_tracker import quick_log
    from plural_converter import convert_plurals_to_singular
    from word_utils import clean_word, is_generic_word

@dataclass
class ExpandedWord:
    """Represents a word with its expansion metadata"""
    word: str
    level: int
    source_word: str
    similarity: float = 0.0
    
class RecursiveExpander:
    """Handles recursive semantic expansion of words"""
    
    def __init__(self, secret_word: str):
        self.secret_word = secret_word.lower().strip()
        
        # File paths
        self.cache_file = Config.SECRETWORD_DIR / f"openai-{self.secret_word}-expanded.txt"
        
        # Data structures
        self.all_words: Dict[str, ExpandedWord] = {}  # word -> ExpandedWord
        self.level_words: Dict[int, List[ExpandedWord]] = {}  # level -> words
        self.processed_words: Set[str] = set()  # Words we've already expanded
        
        # Stats
        self.api_calls = 0
        self.total_words_found = 0
        
    def load_cached_expansion(self) -> bool:
        """Load existing expanded words from cache"""
        if not self.cache_file.exists():
            return False
        
        quick_log(self.secret_word, f"📂 Loading cached expansion from {self.cache_file}")
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                current_level = 0
                
                for line in f:
                    line = line.strip()
                    
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        # Check for level headers
                        if 'Level' in line:
                            try:
                                current_level = int(line.split('Level')[1].split()[0])
                            except:
                                pass
                        continue
                    
                    # Parse word entry: word (similarity: 0.85, source: queen)
                    if '(' in line and ')' in line:
                        word_part = line.split('(')[0].strip()
                        meta_part = line.split('(')[1].split(')')[0]
                        
                        # Extract metadata
                        similarity = 0.0
                        source_word = self.secret_word
                        
                        for item in meta_part.split(','):
                            item = item.strip()
                            if 'similarity:' in item:
                                try:
                                    similarity = float(item.split(':')[1].strip())
                                except:
                                    pass
                            elif 'source:' in item:
                                source_word = item.split(':')[1].strip()
                        
                        # Create expanded word
                        expanded_word = ExpandedWord(
                            word=word_part,
                            level=current_level,
                            source_word=source_word,
                            similarity=similarity
                        )
                        
                        self.all_words[word_part] = expanded_word
                        if current_level not in self.level_words:
                            self.level_words[current_level] = []
                        self.level_words[current_level].append(expanded_word)
            
            self.total_words_found = len(self.all_words)
            quick_log(self.secret_word, f"✅ Loaded {self.total_words_found} cached expanded words across {len(self.level_words)} levels")
            return True
            
        except Exception as e:
            quick_log(self.secret_word, f"❌ ERROR: Failed to load cached expansion: {e}")
            return False
    
    def expand_words_batch(self, words: List[str], level: int) -> List[ExpandedWord]:
        """Get similar words for a batch of words from OpenAI"""
        if not words:
            return []
        
        # Create batch prompt
        words_str = ", ".join(words)
        prompt = f"Give me up to {Config.EXPANSION_MAX_PER_WORD} words similar in meaning to each of these words: {words_str}. Format as a simple list, one word per line, no numbers or explanations."
        
        quick_log(self.secret_word, f"🤖 Level {level} batch expansion: {len(words)} words")
        
        try:
            self.api_calls += 1
            response = openai.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse response
            expanded_words = []
            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # Clean the word
                cleaned = clean_word(line)
                if not cleaned:
                    continue
                
                # Skip if already found or generic
                if cleaned in self.all_words or is_generic_word(cleaned):
                    continue
                
                # Create expanded word with estimated similarity based on level
                estimated_similarity = max(0.1, 0.8 - (level - 1) * 0.2)  # Decreasing similarity by level
                expanded_word = ExpandedWord(
                    word=cleaned,
                    level=level,
                    source_word=words[0],  # Simplified - could be more sophisticated
                    similarity=estimated_similarity
                )
                
                expanded_words.append(expanded_word)
                self.all_words[cleaned] = expanded_word
            
            quick_log(self.secret_word, f"✅ Level {level} batch: got {len(expanded_words)} new words")
            return expanded_words
            
        except Exception as e:
            quick_log(self.secret_word, f"❌ ERROR: Level {level} batch expansion failed: {e}")
            return []
    
    def expand_level(self, source_words: List[ExpandedWord], target_level: int) -> List[ExpandedWord]:
        """Expand all words from a source level to create the next level"""
        if not source_words:
            return []
        
        quick_log(self.secret_word, f"🔄 Expanding Level {target_level-1} → Level {target_level}")
        
        # Filter source words for expansion with progressive thresholds
        expandable_words = []
        
        # Use different similarity thresholds for different levels
        if target_level == 1:
            # Level 1: Use all initial words
            similarity_threshold = 0.0
        elif target_level == 2:
            # Level 2: Use high-similarity words from Level 1
            similarity_threshold = Config.EXPANSION_MIN_SIMILARITY
        else:
            # Level 3+: Use medium-similarity words from previous level
            similarity_threshold = max(0.2, Config.EXPANSION_MIN_SIMILARITY - 0.2)
        
        for word_obj in source_words:
            # Apply similarity threshold based on level
            if word_obj.similarity >= similarity_threshold or target_level == 1:
                expandable_words.append(word_obj.word)
            
            # Limit expansions based on config
            if len(expandable_words) >= Config.EXPANSION_MAX_WORDS_PER_LEVEL:
                break
        
        if not expandable_words:
            quick_log(self.secret_word, f"⏭️ No words meet expansion criteria for Level {target_level}")
            return []
        
        # Batch the expansions
        all_expanded = []
        for i in range(0, len(expandable_words), Config.EXPANSION_BATCH_SIZE):
            batch = expandable_words[i:i + Config.EXPANSION_BATCH_SIZE]
            batch_expanded = self.expand_words_batch(batch, target_level)
            all_expanded.extend(batch_expanded)
            
            # Limit total words per level
            if len(all_expanded) >= Config.EXPANSION_MAX_WORDS_PER_LEVEL:
                all_expanded = all_expanded[:Config.EXPANSION_MAX_WORDS_PER_LEVEL]
                break
        
        # Convert plurals to singular
        if all_expanded:
            words_list = [w.word for w in all_expanded]
            converted_words = convert_plurals_to_singular(self.secret_word, words_list)
            
            # Update expanded words with converted forms
            for i, converted in enumerate(converted_words):
                if i < len(all_expanded) and converted:
                    original_word = all_expanded[i].word
                    all_expanded[i].word = converted
                    
                    # Update all_words mapping
                    if original_word in self.all_words:
                        del self.all_words[original_word]
                    self.all_words[converted] = all_expanded[i]
        
        self.level_words[target_level] = all_expanded
        self.total_words_found += len(all_expanded)
        
        quick_log(self.secret_word, f"✅ Level {target_level}: {len(all_expanded)} words")
        return all_expanded
    
    def perform_recursive_expansion(self, initial_words: List[str]) -> bool:
        """Perform the complete recursive expansion"""
        quick_log(self.secret_word, f"🚀 Starting recursive expansion for '{self.secret_word}'")
        quick_log(self.secret_word, f"⚙️ Config: {Config.EXPANSION_MAX_LEVELS} levels, {Config.EXPANSION_MAX_WORDS_PER_LEVEL} words/level")
        
        # Level 1: Initial words (already provided)
        level_1_words = []
        for word in initial_words:
            expanded_word = ExpandedWord(
                word=word,
                level=1,
                source_word=self.secret_word,
                similarity=0.9  # Assume high similarity for initial words
            )
            level_1_words.append(expanded_word)
            self.all_words[word] = expanded_word
        
        self.level_words[1] = level_1_words
        self.total_words_found = len(level_1_words)
        
        quick_log(self.secret_word, f"✅ Level 1: {len(level_1_words)} initial words")
        
        # Expand to additional levels
        current_level_words = level_1_words
        for level in range(2, Config.EXPANSION_MAX_LEVELS + 1):
            new_words = self.expand_level(current_level_words, level)
            
            if not new_words:
                quick_log(self.secret_word, f"🛑 Stopping expansion at Level {level-1} (no new words)")
                break
            
            current_level_words = new_words
        
        quick_log(self.secret_word, f"✅ Recursive expansion completed: {self.total_words_found} total words, {self.api_calls} API calls")
        return True
    
    def save_expanded_words(self) -> bool:
        """Save expanded words to cache file"""
        if not self.all_words:
            return False
        
        quick_log(self.secret_word, f"💾 Saving expanded words to {self.cache_file}")
        
        try:
            Config.ensure_directories()
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                f.write(f"# Recursive Expansion for '{self.secret_word}'\n")
                f.write(f"# Total words: {self.total_words_found}\n")
                f.write(f"# API calls: {self.api_calls}\n")
                f.write(f"# Levels: {len(self.level_words)}\n\n")
                
                # Write words by level
                for level in sorted(self.level_words.keys()):
                    words = self.level_words[level]
                    if not words:
                        continue
                    
                    if level == 1:
                        level_name = "Direct"
                    elif level == 2:
                        level_name = "1-degree"
                    elif level == 3:
                        level_name = "2-degree"
                    else:
                        level_name = f"{level-1}-degree"
                    f.write(f"# Level {level} ({level_name}): {len(words)} words\n")
                    
                    for word_obj in words:
                        f.write(f"{word_obj.word} (similarity: {word_obj.similarity:.2f}, source: {word_obj.source_word})\n")
                    
                    f.write("\n")
            
            file_size = self.cache_file.stat().st_size
            quick_log(self.secret_word, f"✅ Saved {self.total_words_found} expanded words ({file_size/1024:.1f} KB)")
            return True
            
        except Exception as e:
            quick_log(self.secret_word, f"❌ ERROR: Failed to save expanded words: {e}")
            return False
    
    def get_all_words_list(self) -> List[str]:
        """Get flat list of all expanded words"""
        return list(self.all_words.keys())
    
    def get_words_by_level(self, level: int) -> List[str]:
        """Get words from a specific level"""
        if level not in self.level_words:
            return []
        return [w.word for w in self.level_words[level]]

def create_recursive_expansion(secret_word: str, initial_words: List[str]) -> List[str]:
    """Convenience function to perform recursive expansion"""
    try:
        expander = RecursiveExpander(secret_word)
        
        # Try to load from cache first
        if expander.load_cached_expansion():
            quick_log(secret_word, f"📋 Using cached recursive expansion")
            return expander.get_all_words_list()
        
        # Perform fresh expansion
        if expander.perform_recursive_expansion(initial_words):
            expander.save_expanded_words()
            return expander.get_all_words_list()
        else:
            quick_log(secret_word, f"❌ Recursive expansion failed, returning initial words")
            return initial_words
            
    except Exception as e:
        quick_log(secret_word, f"❌ ERROR in recursive expansion: {e}")
        return initial_words

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python recursive_expansion.py <secret_word>")
        print("Example: python recursive_expansion.py queen")
        sys.exit(1)
    
    secret_word = sys.argv[1]
    
    # Get the full OpenAI word list for realistic testing
    try:
        # Import the OpenAI similar words module
        try:
            from openai_similar_words import get_openai_similar_words
        except ImportError:
            sys.path.append(str(Path(__file__).parent))
            from openai_similar_words import get_openai_similar_words
        
        print(f"🔍 Getting OpenAI words for '{secret_word}' first...")
        initial_words = get_openai_similar_words(secret_word)
        
        if not initial_words:
            print(f"❌ No OpenAI words found for '{secret_word}'. Using fallback test words.")
            # Fallback test words
            if secret_word.lower() == "fire":
                initial_words = ["flame", "burn", "heat", "ignite", "blaze"]
            elif secret_word.lower() == "king":
                initial_words = ["monarch", "empress", "ruler", "sovereign", "princess"]
            elif secret_word.lower() == "ocean":
                initial_words = ["sea", "water", "wave", "marine", "deep"]
            else:
                initial_words = ["similar", "related", "connected", "associated", "linked"]
        
        print(f"✅ Starting with {len(initial_words)} OpenAI words")
        
    except Exception as e:
        print(f"⚠️ Error getting OpenAI words: {e}")
        print(f"🔄 Using fallback test words for '{secret_word}'")
        # Fallback test words
        if secret_word.lower() == "fire":
            initial_words = ["flame", "burn", "heat", "ignite", "blaze"]
        elif secret_word.lower() == "king":
            initial_words = ["monarch", "empress", "ruler", "sovereign", "princess"]
        elif secret_word.lower() == "ocean":
            initial_words = ["sea", "water", "wave", "marine", "deep"]
        else:
            initial_words = ["similar", "related", "connected", "associated", "linked"]
    
    try:
        expanded_words = create_recursive_expansion(secret_word, initial_words)
        print(f"\n🎉 Recursive expansion completed!")
        print(f"📊 Total words: {len(expanded_words)}")
        print(f"📝 First 10 words: {expanded_words[:10]}")
        
    except Exception as e:
        print(f"\n💥 Error: {e}")
        sys.exit(1)
