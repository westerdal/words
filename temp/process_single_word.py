#!/usr/bin/env python3
"""
Process a single word from the master list to generate its semantic ranking CSV
"""

import os
import sys
from semantic_embedding_generator import SemanticEmbeddingGenerator

def process_single_word(word_line):
    """Process a single word line from master-list.txt"""
    
    print(f"=== Processing Single Word ===")
    print(f"Word line: {word_line}")
    
    # Parse the line: easy-animals-dog
    try:
        parts = word_line.strip().split('-')
        if len(parts) != 3:
            print(f"❌ Invalid format. Expected: difficulty-category-word, got: {word_line}")
            return False
        
        difficulty, category, word = parts
        print(f"Difficulty: {difficulty}")
        print(f"Category: {category}")
        print(f"Word: {word}")
        
    except Exception as e:
        print(f"❌ Error parsing word line: {e}")
        return False
    
    # Check if CSV already exists
    csv_filename = f"secretword/secretword-{difficulty}-{category}-{word}.csv"
    if os.path.exists(csv_filename):
        print(f"✅ CSV already exists: {csv_filename}")
        return True
    
    # Create the generator
    print(f"\n🔄 Creating semantic generator for '{word}'...")
    generator = SemanticEmbeddingGenerator(word, batch_size=50)
    
    # Load words
    print("📚 Loading ENABLE word list...")
    if not generator.load_words():
        print("❌ Failed to load word list")
        return False
    
    # Compute rankings using cached embeddings
    print("🧮 Computing semantic rankings from cached embeddings...")
    if not generator.compute_semantic_rankings_from_cache():
        print("⚠️  Failed to use cached embeddings, falling back to API...")
        if not generator.compute_semantic_rankings():
            print("❌ Failed to compute semantic rankings")
            return False
    
    # Generate CSV with custom filename
    print(f"📝 Generating CSV file: {csv_filename}")
    
    # Update the generator to use NULL for distant clues
    generator.ai_cutoff_rank = 10000  # Only AI clues for top 10K
    
    success = generator.generate_csv(csv_filename)
    
    if success:
        # Get file size for confirmation
        file_size = os.path.getsize(csv_filename)
        print(f"🎉 Successfully created {csv_filename}")
        print(f"📏 File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        return True
    else:
        print(f"❌ Failed to generate CSV for '{word}'")
        return False

def main():
    """Main function to process the first word from master-list.txt"""
    
    # Read first line from master list
    master_list_file = "secretword/master-list.txt"
    
    try:
        with open(master_list_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        
        if not first_line:
            print("❌ Master list is empty")
            return False
        
        print(f"Processing first word from {master_list_file}")
        success = process_single_word(first_line)
        
        if success:
            print("\n✅ Single word processing completed successfully!")
        else:
            print("\n❌ Single word processing failed!")
        
        return success
        
    except FileNotFoundError:
        print(f"❌ Master list file not found: {master_list_file}")
        return False
    except Exception as e:
        print(f"❌ Error reading master list: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
