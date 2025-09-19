#!/usr/bin/env python3
"""
Word utility functions shared across modules
"""

import re
import unicodedata
from typing import Optional

def clean_word(word: str) -> Optional[str]:
    """
    Clean a word according to strict rules:
    - Keep only single words (no spaces, no hyphens)
    - Remove numbers, punctuation, non-ASCII characters
    - Remove brackets and their contents
    - Convert to lowercase
    - Return None if word doesn't meet criteria
    """
    if not word or not isinstance(word, str):
        return None
    
    # Remove any text in brackets first
    word = re.sub(r'\[.*?\]', '', word)
    word = re.sub(r'\(.*?\)', '', word)
    
    # Remove leading numbers and dots (e.g., "1. word" -> "word")
    word = re.sub(r'^\d+\.\s*', '', word)
    
    # Remove any remaining punctuation and whitespace
    word = re.sub(r'[^\w]', '', word)
    
    if not word:
        return None
    
    # Convert to lowercase
    word = word.lower()
    
    # Remove non-ASCII characters by converting to closest ASCII
    try:
        # Normalize unicode characters
        word = unicodedata.normalize('NFD', word)
        # Keep only ASCII characters
        word = word.encode('ascii', 'ignore').decode('ascii')
    except:
        return None
    
    if not word:
        return None
    
    # Check if it's a single word (no spaces, reasonable length)
    if ' ' in word or len(word) < 2 or len(word) > 20:
        return None
    
    # Skip common non-words
    skip_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    if word in skip_words:
        return None
    
    return word

def is_generic_word(word: str) -> bool:
    """Check if a word is too generic to be useful"""
    generic_words = {
        'thing', 'item', 'stuff', 'object', 'things', 'items', 
        'something', 'anything', 'everything', 'nothing',
        'person', 'people', 'someone', 'anyone', 'everyone',
        'place', 'somewhere', 'anywhere', 'everywhere',
        'way', 'ways', 'method', 'approach', 'technique'
    }
    return word.lower() in generic_words
