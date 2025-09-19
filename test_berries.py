#!/usr/bin/env python3
"""
Test how the plural converter handles "berries"
"""

import sys
from pathlib import Path

# Add utilities to path
sys.path.append(str(Path(__file__).parent / "scripts" / "utilities"))
from plural_converter import is_plural, pluralize_to_singular

test_words = ["berries", "berry", "trees", "tree", "parties", "party", "stories", "story"]

print("=== Testing Berries and Similar Words ===")
for word in test_words:
    plural = is_plural(word)
    singular = pluralize_to_singular(word) if plural else word
    print(f"{word:<10} -> Plural: {plural:<5} -> Singular: {singular}")
