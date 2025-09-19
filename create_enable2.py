#!/usr/bin/env python3
"""
Create ENABLE2.txt - a filtered version of ENABLE1.txt with no plural words
"""

def is_plural(word):
    """
    Determines if a given word is likely plural based on a comprehensive
    set of English pluralization rules.
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

def create_enable2():
    """Create ENABLE2.txt with no plural words"""
    
    print("=== Creating ENABLE2.txt (No Plurals) ===")
    
    # Read original ENABLE list
    try:
        with open("data/enable1.txt", 'r', encoding='utf-8') as f:
            all_words = [w.strip().lower() for w in f.readlines()]
        print(f"✅ Loaded {len(all_words):,} words from ENABLE1.txt")
    except FileNotFoundError:
        print("❌ ENABLE1.txt not found at data/enable1.txt")
        return False
    
    # Filter out plural words
    print("🔍 Filtering out plural words...")
    singular_words = []
    plural_count = 0
    
    for word in all_words:
        if is_plural(word):
            plural_count += 1
        else:
            singular_words.append(word)
    
    print(f"✅ Filtered to {len(singular_words):,} singular words")
    print(f"📊 Removed {plural_count:,} plural words ({plural_count/len(all_words)*100:.1f}%)")
    
    # Write ENABLE2.txt
    output_file = "data/enable2.txt"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for word in singular_words:
                f.write(word + '\n')
        
        print(f"✅ Created {output_file}")
        print(f"📏 File contains {len(singular_words):,} singular words")
        
        # Show some examples of what was removed
        print(f"\nExamples of plural words removed:")
        plural_examples = [w for w in all_words if is_plural(w)][:10]
        for word in plural_examples:
            print(f"  - {word}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error writing {output_file}: {e}")
        return False

if __name__ == "__main__":
    create_enable2()
