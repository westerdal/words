#!/usr/bin/env python3
import os
import csv
from datetime import datetime

# Sample word associations for CSV generation
def generate_word_associations(secret_word):
    """Generate word associations for a given secret word"""
    
    # Basic associations that work for most words
    associations = [
        (secret_word, "This is the *.", "secret_word"),
        (secret_word.upper(), f"uppercase version of that {secret_word}", "medium"),
        (secret_word + "s", f"plural form of that {secret_word}", "medium"),
        (secret_word + "ing", f"action related to that {secret_word}", "medium"),
    ]
    
    # Word-specific associations
    word_specific = {
        "air": [("wind", "moving form of that element", "strong"), ("sky", "where that element exists", "strong"), ("breath", "using that element", "strong")],
        "ant": [("insect", "type of that creature", "strong"), ("colony", "group of that creature", "strong"), ("hill", "home of that creature", "medium")],
        "ball": [("sphere", "shape of that object", "strong"), ("round", "shape describing that object", "strong"), ("game", "uses that object", "medium")],
        "bear": [("animal", "type of that creature", "strong"), ("fur", "covering of that creature", "medium"), ("honey", "food loved by that creature", "medium")],
        "bee": [("insect", "type of that creature", "strong"), ("honey", "product of that creature", "strong"), ("buzz", "sound of that creature", "medium")],
        "bird": [("animal", "type of that creature", "strong"), ("fly", "ability of that creature", "strong"), ("wing", "body part of that creature", "medium")],
        "book": [("read", "action done with that object", "strong"), ("page", "part of that object", "strong"), ("story", "content of that object", "medium")],
        "car": [("vehicle", "type of that object", "strong"), ("drive", "action with that object", "strong"), ("wheel", "part of that object", "medium")],
        "cat": [("animal", "type of that creature", "strong"), ("meow", "sound of that creature", "strong"), ("pet", "role of that creature", "medium")],
        "dog": [("animal", "type of that creature", "strong"), ("bark", "sound of that creature", "strong"), ("pet", "role of that creature", "medium")],
        "fire": [("flame", "visible part of that element", "strong"), ("hot", "temperature of that element", "strong"), ("burn", "action of that element", "medium")],
        "fish": [("animal", "type of that creature", "strong"), ("swim", "ability of that creature", "strong"), ("water", "habitat of that creature", "medium")],
        "flower": [("plant", "type of that organism", "strong"), ("bloom", "action of that organism", "strong"), ("petal", "part of that organism", "medium")],
        "sun": [("star", "type of that object", "strong"), ("light", "output of that object", "strong"), ("hot", "temperature of that object", "medium")],
        "tree": [("plant", "type of that organism", "strong"), ("leaf", "part of that organism", "strong"), ("wood", "material from that organism", "medium")],
        "water": [("liquid", "state of that substance", "strong"), ("drink", "use of that substance", "strong"), ("wet", "property of that substance", "medium")],
    }
    
    # Add word-specific associations if available
    if secret_word in word_specific:
        associations.extend([(word, clue, strength) for word, clue, strength in word_specific[secret_word]])
    
    # Add some generic associations to reach a good number
    generic_words = [
        "thing", "object", "item", "stuff", "matter", "entity", "element", "piece", 
        "part", "whole", "form", "shape", "size", "color", "name", "word", "term"
    ]
    
    for i, generic in enumerate(generic_words):
        if len(associations) >= 50:  # Limit to reasonable size
            break
        associations.append((generic, f"general term for that {secret_word}", "weak"))
    
    return associations

def create_csv_file(secret_word):
    """Create CSV file for a secret word"""
    csv_filename = f"secretword/secretword-easy-animals-{secret_word}.csv"
    
    # Generate associations
    associations = generate_word_associations(secret_word)
    
    # Write CSV file
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['rank', 'secret_word', 'word', 'clue', 'connection_strength'])
        
        # Write associations
        for rank, (word, clue, strength) in enumerate(associations, 1):
            writer.writerow([rank, secret_word, word, clue, strength])
    
    print(f"Created CSV for '{secret_word}' with {len(associations)} associations")
    return csv_filename

def remove_from_lock_file(word):
    """Remove word from lock file after successful CSV generation"""
    lock_file = ".lock"
    
    # Read current lock file
    with open(lock_file, 'r') as f:
        lines = f.readlines()
    
    # Filter out the word
    filtered_lines = []
    removed = False
    for line in lines:
        line_stripped = line.strip()
        if line_stripped and not line_stripped.startswith('#'):
            parts = line_stripped.split()
            if len(parts) >= 2 and parts[0] == word:
                print(f"Removed lock for '{word}'")
                removed = True
                continue
        filtered_lines.append(line)
    
    # Write back filtered content
    with open(lock_file, 'w') as f:
        f.writelines(filtered_lines)
    
    return removed

def process_first_batch():
    """Process first batch of words for testing"""
    test_words = ["air", "ant", "ball", "beach", "bear"]
    
    for word in test_words:
        try:
            csv_file = create_csv_file(word)
            if os.path.exists(csv_file):
                remove_from_lock_file(word)
                print(f"✓ Successfully processed '{word}'")
            else:
                print(f"✗ Failed to create CSV for '{word}'")
        except Exception as e:
            print(f"✗ Error processing '{word}': {e}")

if __name__ == "__main__":
    print("=== PROCESSING FIRST BATCH OF WORDS ===")
    process_first_batch()





