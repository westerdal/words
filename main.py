#!/usr/bin/env python3
"""
Main script for the Python project.
"""

def load_word_list(filename="data/enable1.txt"):
    """
    Load words from the ENABLE word list.
    
    Args:
        filename (str): Path to the word list file
        
    Returns:
        list: List of words from the file
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            words = [word.strip().lower() for word in file.readlines()]
        return words
    except FileNotFoundError:
        print(f"Error: Could not find word list file at {filename}")
        return []

def main():
    """
    Main function - entry point of the application.
    """
    print("Hello! Your Python environment is ready.")
    print("Loading ENABLE word list...")
    
    # Load the word list
    words = load_word_list()
    
    if words:
        print(f"\n✅ Successfully loaded {len(words):,} words from ENABLE word list!")
        print(f"First 10 words: {words[:10]}")
        print(f"Last 10 words: {words[-10:]}")
        
        # Example: Find words of specific length
        five_letter_words = [word for word in words if len(word) == 5]
        print(f"\nFound {len(five_letter_words):,} five-letter words")
        print(f"Sample five-letter words: {five_letter_words[:10]}")
    else:
        print("❌ Failed to load word list")
    
    print("\nProject structure:")
    print("- main.py (this file)")
    print("- data/enable1.txt (ENABLE word list with 172,823 words)")
    print("- requirements.txt (for dependencies)")
    print("- README.md (project documentation)")
    print("- .gitignore (git ignore rules)")


if __name__ == "__main__":
    main()
