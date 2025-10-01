#!/usr/bin/env python3
"""
Standalone OpenAI Keywords Generator
Generates OpenAI twopass keywords for a single word using enhanced three-method expansion
"""

import sys
from pathlib import Path
from typing import Tuple, List, Optional

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent / "scripts" / "utilities"))

try:
    from scripts.utilities.openai_similar_words import OpenAISimilarWords
    from scripts.utilities.config import Config
    from scripts.utilities.progress_tracker import quick_log
except ImportError:
    # Fallback for direct execution
    from openai_similar_words import OpenAISimilarWords
    from config import Config
    from progress_tracker import quick_log


def generate_openai_keywords(
    word: str, 
    force_regenerate: bool = False,
    save_to_file: bool = True
) -> Tuple[bool, List[str], str]:
    """
    Generate OpenAI twopass keywords for a single word using enhanced three-method expansion
    
    Args:
        word: The secret word to generate keywords for
        force_regenerate: If True, regenerate even if cache exists
        save_to_file: If True, save results to twopass file
    
    Returns:
        Tuple of (success: bool, word_list: List[str], error_message: str)
    """
    try:
        word = word.lower().strip()
        
        # Validate word
        if not word or not word.isalpha() or len(word) < 2:
            return False, [], f"Invalid word: '{word}'"
        
        quick_log(word, f"🚀 Starting OpenAI keyword generation for '{word}'")
        
        # Check if twopass file already exists (unless force regenerate)
        twopass_file = Config.SECRETWORD_DIR / Config.get_openai_twopass_filename(word)
        if not force_regenerate and twopass_file.exists():
            quick_log(word, f"📂 Twopass file already exists, loading from cache")
            try:
                with open(twopass_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                # Find the line with comma-separated words (skip header comments)
                word_line = None
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        word_line = line
                        break
                
                if word_line:
                    cached_words = [w.strip() for w in word_line.split(',') if w.strip()]
                    if cached_words:
                        quick_log(word, f"✅ Loaded {len(cached_words)} words from existing twopass file")
                        return True, cached_words, ""
            except Exception as e:
                quick_log(word, f"⚠️ Failed to load existing twopass file: {e}")
        
        # Generate new keywords using enhanced three-method expansion
        openai_generator = OpenAISimilarWords(word)
        
        # Use the enhanced three-method expansion
        word_list = openai_generator.get_two_pass_expansion()
        
        if not word_list:
            error_msg = f"Failed to generate OpenAI keywords for '{word}'"
            quick_log(word, f"❌ {error_msg}")
            return False, [], error_msg
        
        # Save to file if requested
        if save_to_file:
            success = openai_generator.save_two_pass_cache(word_list)
            if not success:
                error_msg = f"Generated keywords but failed to save to file for '{word}'"
                quick_log(word, f"⚠️ {error_msg}")
                return True, word_list, error_msg  # Still return the words
        
        quick_log(word, f"✅ Successfully generated {len(word_list)} OpenAI keywords for '{word}'")
        return True, word_list, ""
        
    except Exception as e:
        error_msg = f"Error generating OpenAI keywords for '{word}': {e}"
        quick_log(word, f"❌ {error_msg}")
        return False, [], error_msg


def main():
    """Command line interface for standalone usage"""
    if len(sys.argv) != 2:
        print("Usage: python generate_openai_keywords.py <word>")
        print("Example: python generate_openai_keywords.py juice")
        sys.exit(1)
    
    word = sys.argv[1]
    
    print(f"🚀 Generating OpenAI keywords for: '{word}'")
    print("=" * 60)
    
    success, word_list, error_message = generate_openai_keywords(word, force_regenerate=True)
    
    if success:
        print(f"\n✅ SUCCESS: Generated {len(word_list)} keywords")
        print(f"📋 First 20 words: {', '.join(word_list[:20])}")
        if error_message:
            print(f"⚠️ Warning: {error_message}")
    else:
        print(f"\n❌ FAILED: {error_message}")
        sys.exit(1)


if __name__ == "__main__":
    main()

