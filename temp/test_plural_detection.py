#!/usr/bin/env python3
"""
Test the plural detection logic
"""

def is_plural(word):
    """
    Determines if a given word is likely plural based on a comprehensive
    set of English pluralization rules.

    Args:
      word: A string representing the word to check.

    Returns:
      True if the word is likely plural, False otherwise.
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

def test_plural_detection():
    """Test the plural detection function"""
    
    print("=== Testing Plural Detection ===")
    
    # Test cases: (word, expected_result)
    test_cases = [
        # Should be detected as plural
        ("dogs", True),
        ("cats", True),
        ("boxes", True),
        ("babies", True),
        ("wives", True),
        ("men", True),
        ("children", True),
        ("feet", True),
        ("mice", True),
        
        # Should be detected as singular
        ("dog", False),
        ("cat", False),
        ("box", False),
        ("baby", False),
        ("wife", False),
        ("man", False),
        ("child", False),
        ("foot", False),
        ("mouse", False),
        
        # Edge cases - should be singular
        ("class", False),    # ends in 'ss'
        ("bus", False),      # ends in 'us' 
        ("status", False),   # ends in 'us'
        ("analysis", False), # ends in 'is'
        ("glass", False),    # ends in 'ss'
        
        # Same form singular/plural
        ("sheep", True),     # treated as plural by rule
        ("deer", True),      # treated as plural by rule
        ("fish", True),      # treated as plural by rule
    ]
    
    print(f"Testing {len(test_cases)} cases...\n")
    
    passed = 0
    failed = 0
    
    for word, expected in test_cases:
        result = is_plural(word)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        print(f"{status} | {word:<12} | Expected: {str(expected):<5} | Got: {str(result):<5}")
        
        if result == expected:
            passed += 1
        else:
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    
    # Test with some words from the ENABLE list
    print(f"\n=== Sample ENABLE Words ===")
    sample_words = ["dog", "dogs", "forest", "trees", "animal", "animals", "book", "books"]
    
    for word in sample_words:
        result = is_plural(word)
        print(f"{word:<10} -> {'Plural' if result else 'Singular'}")
    
    return failed == 0

if __name__ == "__main__":
    success = test_plural_detection()
    print(f"\n{'✅ All tests passed!' if success else '❌ Some tests failed!'}")
