#!/usr/bin/env python3
"""
Plural to Singular Converter Module
Converts plural words to singular form and removes duplicates
"""

from typing import List, Set
from pathlib import Path

try:
    from .progress_tracker import quick_log
except ImportError:
    # For standalone execution
    import sys
    sys.path.append(str(Path(__file__).parent))
    from progress_tracker import quick_log

def is_plural(word: str) -> bool:
    """
    Determines if a given word is likely plural based on English pluralization rules.
    """
    # Convert word to lowercase for case-insensitive checks
    w_lower = word.lower()

    # Rule 5: Check against a set of common irregular plurals
    irregular_plurals = {"men", "women", "children", "feet", "teeth", "mice", "people"}
    if w_lower in irregular_plurals:
        return True

    # Rule 6: Check against nouns where singular and plural forms are the same
    no_change_nouns = {"sheep", "deer", "fish", "series", "species"}
    if w_lower in no_change_nouns:
        return True

    # Rule 4: Check for words ending in "-ves" (e.g., wives, leaves)
    if w_lower.endswith("ves"):
        return True
        
    # Rule 3: Check for words ending in "-ies" (e.g., babies, parties)
    if w_lower.endswith("ies"):
        return True
        
    # Rule 2: Check for words ending in "-es" (e.g., boxes, buses)
    if w_lower.endswith("es"):
        return True
        
    # Rule 1: Check for the most common "-s" ending, excluding common singulars
    # to reduce false positives (e.g., "status", "bus", "class").
    if w_lower.endswith("s") and not w_lower.endswith(("ss", "us", "is")):
        return True

    # If none of the plural rules match, assume the word is singular.
    return False

def pluralize_to_singular(word: str) -> str:
    """
    Convert a plural word to its singular form.
    """
    w_lower = word.lower()
    
    # Handle irregular plurals
    irregular_map = {
        "men": "man",
        "women": "woman", 
        "children": "child",
        "feet": "foot",
        "teeth": "tooth",
        "mice": "mouse",
        "people": "person"
    }
    
    if w_lower in irregular_map:
        return irregular_map[w_lower]
    
    # Handle no-change nouns (already singular)
    no_change_nouns = {"sheep", "deer", "fish", "series", "species"}
    if w_lower in no_change_nouns:
        return word
    
    # Handle -ves endings (wives -> wife, leaves -> leaf)
    if w_lower.endswith("ves"):
        if w_lower.endswith("ives"):
            return w_lower[:-4] + "ife"  # wives -> wife
        else:
            return w_lower[:-3] + "f"    # leaves -> leaf
    
    # Handle -ies endings (babies -> baby, parties -> party)  
    if w_lower.endswith("ies"):
        return w_lower[:-3] + "y"
    
    # Handle -es endings (but not words that just end in -es as part of the root)
    if w_lower.endswith("es"):
        # Special cases for -ches, -shes, -xes, -zes, -ses (true -es plurals)
        if w_lower.endswith(("ches", "shes", "xes", "zes", "ses")):
            return w_lower[:-2]  # boxes -> box, dishes -> dish, buses -> bus
        # Skip words that just happen to end in -es but aren't -es plurals
        # These will be handled by the -s rule if appropriate
    
    # Handle regular -s endings
    if w_lower.endswith("s") and not w_lower.endswith(("ss", "us", "is")):
        return w_lower[:-1]
    
    # If no plural pattern matched, return original word
    return word

class PluralConverter:
    """Handles conversion of plural words to singular and deduplication"""
    
    def __init__(self, secret_word: str):
        self.secret_word = secret_word.lower().strip()
    
    def convert_and_deduplicate(self, words: List[str]) -> List[str]:
        """Convert plurals to singular and remove duplicates while preserving order"""
        quick_log(self.secret_word, f"🔄 Converting plurals to singular for {len(words)} words...")
        
        seen_words: Set[str] = set()
        converted_words: List[str] = []
        plural_count = 0
        duplicate_count = 0
        
        for word in words:
            original_word = word.strip().lower()
            
            # Skip empty words
            if not original_word:
                continue
            
            # Convert plural to singular if needed
            if is_plural(original_word):
                singular_word = pluralize_to_singular(original_word)
                plural_count += 1
            else:
                singular_word = original_word
            
            # Check for duplicates
            if singular_word in seen_words:
                duplicate_count += 1
                continue
            
            # Add to results
            seen_words.add(singular_word)
            converted_words.append(singular_word)
        
        quick_log(self.secret_word, f"✅ Conversion complete:")
        quick_log(self.secret_word, f"   Original words: {len(words)}")
        quick_log(self.secret_word, f"   Plurals converted: {plural_count}")
        quick_log(self.secret_word, f"   Duplicates removed: {duplicate_count}")
        quick_log(self.secret_word, f"   Final unique words: {len(converted_words)}")
        
        return converted_words
    
    def process_word_list(self, words: List[str]) -> List[str]:
        """Main processing function"""
        if not words:
            return []
        
        return self.convert_and_deduplicate(words)

def convert_plurals_to_singular(secret_word: str, words: List[str]) -> List[str]:
    """Convenience function to convert plurals and remove duplicates"""
    try:
        converter = PluralConverter(secret_word)
        return converter.process_word_list(words)
    except Exception as e:
        quick_log(secret_word, f"❌ ERROR in plural converter: {e}")
        return words  # Return original list on error

if __name__ == "__main__":
    # Test the converter
    test_words = [
        "trees", "tree", "forests", "forest", "leaves", "leaf",
        "mice", "mouse", "children", "child", "boxes", "box",
        "dishes", "dish", "parties", "party", "wives", "wife",
        "sheep", "deer", "fish", "status", "bus", "class"
    ]
    
    print("=== Plural Converter Test ===")
    
    converter = PluralConverter("test")
    result = converter.convert_and_deduplicate(test_words)
    
    print(f"\nOriginal: {test_words}")
    print(f"Converted: {result}")
    
    print("\n=== Individual Tests ===")
    for word in test_words:
        plural = is_plural(word)
        singular = pluralize_to_singular(word) if plural else word
        print(f"{word:<12} -> Plural: {plural:<5} -> Singular: {singular}")
    
    print("\n✅ Test completed!")
