#!/usr/bin/env python3
"""
Execute the complete CSV generation prompt for a specific word
Based on run_csv_prompt.py but targets a specific word
"""

import os
import sys
import json
import subprocess
import numpy as np
import re
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm

def process_specific_word(target_word):
    """Process a specific word through the complete CSV generation pipeline"""
    print(f"🚀 EXECUTING CSV GENERATION FOR SPECIFIC WORD: {target_word}")
    print("Following csv-prompt.md specifications\n")
    
    try:
        # Check if word already has CSV
        secretword_dir = Path("secretword")
        csv_file = secretword_dir / f"secretword-easy-animals-{target_word}.csv"
        if csv_file.exists():
            print(f"✅ CSV file already exists for '{target_word}': {csv_file}")
            return True
        
        # Check if embeddings exist
        embeddings_file = secretword_dir / f"{target_word}-embeddings.txt"
        
        if not embeddings_file.exists():
            print(f"❌ No embeddings file found for '{target_word}'")
            print(f"   Looked for: {embeddings_file}")
            return False
        
        # Step 4.1: Load ENABLE words
        print("\n=== 4.1. LOADING ENABLE WORD LIST ===")
        enable_file = Path("data/enable2.txt")
        if not enable_file.exists():
            print(f"❌ ENABLE word list not found: {enable_file}")
            return False
        
        with open(enable_file, 'r', encoding='utf-8') as f:
            all_words = [w.strip().lower() for w in f.readlines() if w.strip()]
        
        print(f"📚 Loaded {len(all_words):,} words from ENABLE2")
        
        # Filter out plural words
        def is_plural(word):
            w_lower = word.lower()
            irregular_plurals = {"men", "women", "children", "feet", "teeth", "mice", "people"}
            if w_lower in irregular_plurals:
                return True
            no_change_nouns = {"sheep", "deer", "fish", "series", "species"}
            if w_lower in no_change_nouns:
                return True
            if w_lower.endswith("ves") or w_lower.endswith("ies") or w_lower.endswith("es"):
                return True
            if w_lower.endswith("s") and not w_lower.endswith(("ss", "us", "is")):
                return True
            return False
        
        singular_words = [word for word in all_words if not is_plural(word)]
        print(f"🔍 Filtered to {len(singular_words):,} singular words")
        
        # Verify target word is in enable words
        if target_word not in singular_words:
            print(f"❌ Target word '{target_word}' not in filtered ENABLE list")
            return False
        
        # Step 4.2: Load embeddings and compute rankings
        print(f"\n=== 4.2. LOADING EMBEDDINGS FOR '{target_word}' ===")
        
        target_file = embeddings_file
        print(f"📂 Using embeddings: {target_file}")
        print(f"📥 Loading embeddings from: {target_file}")
        
        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"📄 Read {len(lines):,} lines from embeddings file")
            
            # Parse rankings
            rankings = {}
            first_line = lines[0].strip()
            
            if first_line.startswith('rank,word,similarity') or ',' in first_line:
                print(f"📋 Detected new CSV format")
                import csv
                reader = csv.reader(lines)
                if first_line.startswith('rank,word,similarity'):
                    next(reader)
                
                for row in reader:
                    if len(row) >= 3:
                        rank = int(row[0])
                        word = row[1]
                        similarity = float(row[2])
                        rankings[word] = {'rank': rank, 'similarity': similarity}
            else:
                print(f"📋 Detected old space-separated format")
                processed_lines = 0
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 3:
                            rank = int(parts[0])
                            similarity = float(parts[1])
                            word = parts[2]
                            rankings[word] = {'rank': rank, 'similarity': similarity}
                            processed_lines += 1
                            
                            if processed_lines % 500 == 0:
                                print(f"📊 Processed {processed_lines:,} rankings...")
            
            print(f"✅ Loaded rankings for {len(rankings):,} words")
            
            if target_word not in rankings:
                print(f"❌ Target word '{target_word}' not found in rankings")
                return False
            
            print(f"🎯 '{target_word}' has rank: {rankings[target_word]['rank']}")
            
            # Show top 10
            top_words = sorted(rankings.items(), key=lambda x: x[1]['rank'])[:10]
            print(f"\nTop 10 most similar words to '{target_word}':")
            for i, (word, data) in enumerate(top_words, 1):
                print(f"  {i:2d}. {word:<15} (similarity: {data['similarity']:.6f})")
            
        except Exception as e:
            print(f"❌ Error loading embeddings file: {e}")
            return False
        
        # Step 4.3: OpenAI expansion
        print(f"\n=== 4.3. OPENAI SIMILAR WORDS EXPANSION FOR '{target_word}' ===")
        
        original_dir = os.getcwd()
        try:
            os.chdir("scripts/utilities")
            
            process = subprocess.Popen([
                sys.executable, "openai_similar_words.py", target_word
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
               text=True, bufsize=1, universal_newlines=True, encoding='utf-8', errors='replace')
            
            # Track OpenAI statistics
            api_calls_made = 0
            primary_words = 0
            synonym_words = 0
            total_words = 0
            
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    line = output.strip()
                    print(line)
                    
                    # Parse statistics
                    if "⚠️ WARNING: OpenAI API key not available" in line:
                        print(f"❌ CRITICAL: OpenAI API key not available")
                        return False
                    elif "❌ Cannot get primary associations - API key not available" in line:
                        print(f"❌ CRITICAL: OpenAI primary associations failed")
                        return False
                    elif "Primary pass:" in line and "filtered words" in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if "filtered" in part and i > 0:
                                primary_words = int(parts[i-1])
                        api_calls_made += 1
                    elif "✅ Loaded" in line and "words from two-pass cache" in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "Loaded" and i+1 < len(parts):
                                try:
                                    cached_count = int(parts[i+1])
                                    total_words = cached_count
                                    break
                                except (ValueError, IndexError):
                                    pass
                    elif "unique total" in line:
                        try:
                            match = re.search(r'= (\d+) unique total', line)
                            if match:
                                total_words = int(match.group(1))
                        except (ValueError, AttributeError):
                            pass
            
            return_code = process.wait()
            
            if return_code == 0:
                print(f"✅ OpenAI expansion completed successfully for '{target_word}'")
                print(f"📊 Final Stats: {api_calls_made} API calls, {total_words} total words")
                
                if api_calls_made == 0 and total_words == 0:
                    print(f"❌ OpenAI expansion failed - no data available")
                    return False
                elif api_calls_made == 0 and total_words > 0:
                    print(f"✅ Using cached OpenAI data with {total_words} words")
            else:
                print(f"❌ OpenAI expansion failed for '{target_word}' (exit code: {return_code})")
                return False
        finally:
            os.chdir(original_dir)
        
        # Step 5: CSV generation
        print(f"\n=== 5. CSV GENERATION FOR '{target_word}' ===")
        
        original_dir = os.getcwd()
        try:
            os.chdir("scripts/processing")
            
            process = subprocess.Popen([
                sys.executable, "generate_csv.py", target_word
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
               text=True, bufsize=1, universal_newlines=True, encoding='utf-8', errors='replace')
            
            # Track CSV statistics
            ai_clues_generated = 0
            
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    line = output.strip()
                    print(line)
                    
                    if "AI clues:" in line:
                        try:
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part == "clues:" and i > 0 and parts[i-1] == "AI":
                                    ai_clues_generated = int(parts[i+1].replace(",", "").replace("|", ""))
                        except:
                            pass
            
            return_code = process.wait()
            
            if return_code == 0:
                print(f"✅ CSV generation completed successfully for '{target_word}'")
                print(f"📊 Final Stats: {ai_clues_generated:,} AI clues generated")
                return True
            else:
                print(f"❌ CSV generation failed for '{target_word}' (exit code: {return_code})")
                return False
        finally:
            os.chdir(original_dir)
        
    except Exception as e:
        print(f"❌ Error processing '{target_word}': {e}")
        return False

def main():
    """Main execution for specific word"""
    if len(sys.argv) != 2:
        print("Usage: python run_csv_prompt_word.py <word>")
        print("Example: python run_csv_prompt_word.py meat")
        sys.exit(1)
    
    target_word = sys.argv[1].lower()
    success = process_specific_word(target_word)
    
    if success:
        print(f"\n🎉 SUCCESSFULLY COMPLETED PROCESSING FOR: {target_word}")
    else:
        print(f"\n❌ FAILED TO PROCESS: {target_word}")
        sys.exit(1)

if __name__ == "__main__":
    main()
