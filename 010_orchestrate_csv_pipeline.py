#!/usr/bin/env python3
"""
Execute the complete CSV generation prompt with all main program steps
Following csv-prompt.md specifications
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

def clean_lock_file():
    """Step 1: Garbage collection on lock file"""
    print("=== 1. GARBAGE COLLECTION ON LOCK FILE ===")
    
    lock_file = Path("secretword") / ".lock-csv.lock"
    current_time = datetime.now()
    one_hour_ago = current_time - timedelta(hours=1)
    
    if not lock_file.exists():
        lock_file.parent.mkdir(exist_ok=True)
        with open(lock_file, 'w') as f:
            f.write("# Lock file for CSV generation\n")
            f.write("# Format: [word] [timestamp]\n")
        print(f"✅ Created lock file: {lock_file}")
        return []
    
    # Read current lock file
    with open(lock_file, 'r') as f:
        lines = f.readlines()
    
    # Filter out stale entries
    active_locks = []
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            cleaned_lines.append(line)
            continue
            
        parts = line.split()
        if len(parts) >= 2:
            word = parts[0]
            timestamp_str = parts[1]
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', ''))
                if timestamp > one_hour_ago:
                    active_locks.append(word)
                    cleaned_lines.append(line)
                else:
                    print(f"🗑️ Removing stale lock for '{word}' (timestamp: {timestamp_str})")
            except ValueError:
                print(f"⚠️ Invalid timestamp format: {timestamp_str}")
    
    # Write cleaned lock file
    with open(lock_file, 'w') as f:
        for line in cleaned_lines:
            f.write(line + '\n')
    
    print(f"✅ Active locks after cleanup: {active_locks}")
    return active_locks

def scan_target_words():
    """Step 2: Find words that need CSV generation"""
    print("\n=== 2. SCANNING FOR TARGET WORDS ===")
    
    secretword_dir = Path("secretword")
    
    # Get all embedding files and extract base words
    embedding_files = []
    for file in secretword_dir.glob("*-embeddings.txt"):
        word = file.stem.replace("-embeddings", "")
        # Remove suffixes like '2', '-clean' to get base word
        import re
        base_word = re.sub(r'2$', '', word)
        base_word = re.sub(r'-clean$', '', base_word)
        # Only include if it's a valid base word (no numbers or special chars)
        if base_word.isalpha() and len(base_word) > 1:
            embedding_files.append(base_word)
    
    embedding_words = sorted(set(embedding_files))
    print(f"📂 Found {len(embedding_words)} words with embeddings")
    
    # Get existing CSV files
    csv_files = []
    for file in secretword_dir.glob("*-secret.csv"):
        word = file.stem.replace("-secret", "")
        # Skip temp/backup/incomplete files
        if not re.search(r'_(temp|backup|incomplete)$', word):
            csv_files.append(word)
    
    csv_words = sorted(set(csv_files))
    print(f"📊 Found {len(csv_words)} existing CSV files")
    
    # Find words needing CSV generation
    words_needing_csv = [word for word in embedding_words if word not in csv_words]
    print(f"🎯 Words needing CSV generation: {len(words_needing_csv)}")
    
    if words_needing_csv:
        print(f"📋 Next words to process: {', '.join(words_needing_csv[:10])}")
        if len(words_needing_csv) > 10:
            print(f"    ... and {len(words_needing_csv) - 10} more")
    
    return words_needing_csv

def reserve_single_word(words_needing_csv, active_locks):
    """Step 3: Reserve ONE word in lock file"""
    print("\n=== 3. RESERVING SINGLE WORD ===")
    
    lock_file = Path("secretword") / ".lock-csv.lock"
    current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # Find first available word
    available_word = None
    for word in words_needing_csv:
        if word not in active_locks:
            available_word = word
            break
    
    if not available_word:
        print("❌ No words available for reservation")
        return None
    
    # Reserve the word
    with open(lock_file, 'a') as f:
        f.write(f"{available_word} {current_time}\n")
    
    print(f"🔒 Reserved word: {available_word}")
    return available_word

def load_enable_words():
    """Step 4.1: Load and filter ENABLE word list"""
    print("\n=== 4.1. LOADING ENABLE WORD LIST ===")
    
    enable_file = Path("data/enable2.txt")
    if not enable_file.exists():
        print(f"❌ ENABLE word list not found: {enable_file}")
        return None
    
    # Load all words
    with open(enable_file, 'r', encoding='utf-8') as f:
        all_words = [w.strip().lower() for w in f.readlines() if w.strip()]
    
    print(f"📚 Loaded {len(all_words):,} words from ENABLE2")
    
    # Filter out plural words using comprehensive rules
    def is_plural(word):
        w_lower = word.lower()
        
        # Irregular plurals
        irregular_plurals = {"men", "women", "children", "feet", "teeth", "mice", "people"}
        if w_lower in irregular_plurals:
            return True
        
        # Same singular/plural
        no_change_nouns = {"sheep", "deer", "fish", "series", "species"}
        if w_lower in no_change_nouns:
            return True
        
        # Pluralization patterns
        if w_lower.endswith("ves") or w_lower.endswith("ies") or w_lower.endswith("es"):
            return True
        
        if w_lower.endswith("s") and not w_lower.endswith(("ss", "us", "is")):
            return True
        
        return False
    
    # Filter plurals
    singular_words = []
    plural_count = 0
    
    for word in all_words:
        if is_plural(word):
            plural_count += 1
        else:
            singular_words.append(word)
    
    print(f"🔍 Filtered to {len(singular_words):,} singular words ({plural_count:,} plurals removed)")
    return singular_words

def run_meta_generation(secret_word):
    """Step 4.1.5: Generate meta file with OpenAI classification"""
    print(f"\n=== 4.1.5. GENERATING META FILE ===")
    
    # Check if meta file already exists
    meta_file = Path("secretword") / f"{secret_word}-meta.json"
    if meta_file.exists():
        print(f"📄 Meta file already exists: {meta_file}")
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
            category = meta_data.get('category', 'default')
            print(f"📋 Existing category: {category}")
            return category
        except Exception as e:
            print(f"⚠️ Error reading existing meta file: {e}")
            print(f"🔄 Will regenerate meta file...")
    
    print(f"🤖 Generating meta file for '{secret_word}' using OpenAI classification...")
    print(f"⏱️ This typically takes 10-30 seconds depending on API response time")
    print(f"" + "="*60)
    
    try:
        # Run meta generation as subprocess
        original_dir = os.getcwd()
        os.chdir("scripts/processing")
        
        # Set environment variables for UTF-8 encoding
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        
        result = subprocess.run([
            sys.executable, "015_generate_word_meta.py", secret_word
        ], capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
        
        os.chdir(original_dir)
        
        success = result.returncode == 0
        if success:
            print(result.stdout)  # Show the script output
            
            # Load the generated meta file to get the category
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta_data = json.load(f)
                category = meta_data.get('category', 'default')
                
                print(f"" + "="*60)
                print(f"✅ Meta generation completed successfully for '{secret_word}'")
                print(f"📊 Classified as: {category}")
                print(f"💾 Meta file saved: {meta_file}")
                
                return category
                
            except Exception as e:
                print(f"❌ Error reading generated meta file: {e}")
                return "default"
        else:
            print(f"❌ Meta generation failed: {result.stderr}")
            print(f"⚠️ Using default category for '{secret_word}'")
            return "default"
        
    except Exception as e:
        print(f"❌ Meta generation subprocess failed: {e}")
        print(f"⚠️ Using default category for '{secret_word}'")
        return "default"

def load_embeddings_and_compute_rankings(secret_word, enable_words):
    """Step 4.2: Load embeddings and compute semantic rankings"""
    print("\n=== 4.2. LOADING EMBEDDINGS AND COMPUTING RANKINGS ===")
    
    secretword_dir = Path("secretword")
    
    # Priority: 1) Enhanced version, 2) Standard version
    embeddings2_file = secretword_dir / f"{secret_word}2-embeddings.txt"
    embeddings_file = secretword_dir / f"{secret_word}-embeddings.txt"
    
    target_file = None
    if embeddings2_file.exists():
        target_file = embeddings2_file
        print(f"📂 Using enhanced embeddings: {target_file}")
    elif embeddings_file.exists():
        target_file = embeddings_file
        print(f"📂 Using standard embeddings: {target_file}")
    else:
        print(f"❌ No embeddings file found for '{secret_word}'")
        print(f"   Looked for: {embeddings2_file}")
        print(f"   Looked for: {embeddings_file}")
        return None
    
    # Load pre-computed rankings from embeddings file
    print(f"📥 Loading embeddings from: {target_file}")
    print(f"⏱️ Loading and parsing embeddings file...")
    
    try:
        with open(target_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"📄 Read {len(lines):,} lines from embeddings file")
        
        if not lines:
            print(f"❌ Embeddings file is empty")
            return None
        
        # Detect format by checking first line
        first_line = lines[0].strip()
        rankings = {}
        
        if first_line.startswith('rank,word,similarity') or ',' in first_line:
            # New CSV format
            print(f"📋 Detected new CSV format - parsing rankings...")
            import csv
            reader = csv.reader(lines)
            if first_line.startswith('rank,word,similarity'):
                next(reader)  # Skip header
            
            for row in reader:
                if len(row) >= 3:
                    rank = int(row[0])
                    word = row[1]
                    similarity = float(row[2])
                    rankings[word] = {'rank': rank, 'similarity': similarity}
        
        else:
            # Old space-separated format: "1 1.000000 rock"
            print(f"📋 Detected old space-separated format - parsing rankings...")
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
                        
                        # Progress indicator for large files
                        if processed_lines % 500 == 0:
                            print(f"📊 Processed {processed_lines:,} rankings...")
        
        print(f"✅ Loaded rankings for {len(rankings):,} words")
        
        # Check if secret word exists
        if secret_word not in rankings:
            print(f"❌ Secret word '{secret_word}' not found in rankings")
            return None
        
        print(f"🎯 '{secret_word}' has rank: {rankings[secret_word]['rank']}")
        
        # Show top 10
        top_words = sorted(rankings.items(), key=lambda x: x[1]['rank'])[:10]
        print(f"\nTop 10 most similar words to '{secret_word}':")
        for i, (word, data) in enumerate(top_words, 1):
            print(f"  {i:2d}. {word:<15} (similarity: {data['similarity']:.6f})")
        
        return rankings
        
    except Exception as e:
        print(f"❌ Error loading embeddings file: {e}")
        return None

def run_openai_expansion(secret_word):
    """Step 4.3: Run OpenAI similar words processing using standalone function"""
    print(f"\n=== 4.3. OPENAI SIMILAR WORDS EXPANSION ===")
    
    print(f"🤖 Running OpenAI enhanced three-method expansion for '{secret_word}'...")
    print(f"⏱️ This typically takes 2-5 minutes depending on word complexity")
    print(f"📊 Will show real-time progress from OpenAI API calls...")
    print(f"🔍 Monitoring for: API calls, word counts, filtering stats, cache usage")
    print(f"" + "="*60)
    
    try:
        # Run OpenAI expansion as subprocess
        original_dir = os.getcwd()
        os.chdir("scripts/utilities")
        
        # Set environment variables for UTF-8 encoding
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        
        result = subprocess.run([
            sys.executable, "020_expand_vocabulary.py", secret_word
        ], capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
        
        os.chdir(original_dir)
        
        success = result.returncode == 0
        if success:
            # Parse the output to check if words were generated
            output_lines = result.stdout.strip().split('\n')
            success_line = [line for line in output_lines if line.startswith("SUCCESS:")]
            if success_line:
                # Extract word count from line like "SUCCESS: Got 928 similar words for 'glass'!"
                import re
                match = re.search(r'Got (\d+) similar words', success_line[0])
                word_count = int(match.group(1)) if match else 0
                success = word_count > 0
                error_message = None if success else f"No words generated (count: {word_count})"
            else:
                success = False
                error_message = "Could not find success confirmation in output"
        else:
            error_message = result.stderr or "OpenAI expansion subprocess failed"
        
        if not success:
            print(f"❌ OpenAI expansion failed: {error_message}")
            return False
        
        # Report final statistics
        print(f"" + "="*60)
        print(f"✅ OpenAI expansion completed successfully for '{secret_word}'")
        
        if error_message:
            print(f"⚠️ Warning: {error_message}")
        
        print(f"📊 Final Stats: OpenAI expansion completed with {word_count if 'word_count' in locals() else 'unknown'} words")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAI expansion failed: {e}")
        return False

def run_csv_generation(secret_word):
    """Step 5: Generate final CSV file"""
    print(f"\n=== 5. CSV GENERATION ===")
    
    print(f"📄 Generating CSV for '{secret_word}'...")
    print(f"⏱️ This typically takes 10-20 minutes for full processing")
    print(f"🤖 Will generate AI clues for top-ranked words (up to 5,000)")
    print(f"📊 Will show real-time progress with batch updates...")
    print(f"🔍 Monitoring for: AI clue generation, processing speed, cache hits, completion stats")
    print(f"" + "="*60)
    
    # Change to processing directory and run
    original_dir = os.getcwd()
    try:
        os.chdir("scripts/processing")
        
        # Use Popen to capture output in real-time and parse for CSV generation stats
        import subprocess
        
        # Set environment variables for UTF-8 encoding
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        
        process = subprocess.Popen([
            sys.executable, "030_generate_final_csv.py", secret_word
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
           text=True, bufsize=1, universal_newlines=True, encoding='utf-8', errors='replace', env=env)
        
        # Track CSV generation statistics
        ai_clues_generated = 0
        total_rows_processed = 0
        cache_hits = 0
        cache_misses = 0
        hot_words_found = 0
        processing_rate = 0
        
        # Process output line by line
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                print(line)  # Show original output
                
                # Parse for CSV generation statistics
                if "processed" in line and "/" in line:
                    # Extract from lines like "4,501/5,000 processed"
                    try:
                        if "processed" in line:
                            parts = line.split()
                            for part in parts:
                                if "/" in part and "processed" not in part:
                                    current, total = part.split("/")
                                    current = current.replace(",", "").replace("#", "")
                                    total = total.replace(",", "")
                                    if current.isdigit():
                                        total_rows_processed = int(current)
                                        print(f"📊 CSV STATUS: Processed {total_rows_processed:,} words so far...")
                    except:
                        pass
                elif "🔥" in line and "Super close, sizzling hot" in line:
                    hot_words_found += 1
                    print(f"🔥 HOT WORD DISCOVERY: Found {hot_words_found} super close words!")
                elif "AI clues:" in line:
                    # Extract from "AI clues: 4,950 | NULL clues: 109,479"
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "clues:" and i > 0 and parts[i-1] == "AI":
                                ai_clues_generated = int(parts[i+1].replace(",", "").replace("|", ""))
                                print(f"🤖 AI CLUE STATUS: Generated {ai_clues_generated:,} AI clues")
                    except:
                        pass
                elif "Cache:" in line and "hits" in line and "misses" in line:
                    # Extract from "Cache: 0 hits | 4,950 misses | 0.0% hit rate"
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "hits" and i > 0:
                                cache_hits = int(parts[i-1])
                            elif part == "misses" and i > 0:
                                cache_misses = int(parts[i-1])
                        print(f"💾 CACHE STATUS: {cache_hits:,} hits, {cache_misses:,} misses")
                    except:
                        pass
                elif "Generated" in line and "CSV rows" in line:
                    # Extract final row count
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "Generated" and i+1 < len(parts):
                                total_csv_rows = parts[i+1].replace(",", "")
                                if total_csv_rows.isdigit():
                                    print(f"📋 FINAL CSV: Generated {int(total_csv_rows):,} total CSV rows")
                    except:
                        pass
        
        return_code = process.wait()
        print(f"" + "="*60)
        
        if return_code == 0:
            print(f"✅ CSV generation completed successfully for '{secret_word}'")
            print(f"📊 Final Stats: {ai_clues_generated:,} AI clues, {hot_words_found} hot words found")
            return True
        else:
            print(f"❌ CSV generation failed for '{secret_word}' (exit code: {return_code})")
            return False
    finally:
        os.chdir(original_dir)

def remove_lock(word):
    """Step 6: Remove lock after successful completion"""
    print(f"\n=== 6. CLEANUP LOCK ===")
    
    lock_file = Path("secretword") / ".lock-csv.lock"
    
    # Read current lock file
    with open(lock_file, 'r') as f:
        lines = f.readlines()
    
    # Filter out the completed word
    updated_lines = []
    removed = False
    for line in lines:
        line_stripped = line.strip()
        if line_stripped and not line_stripped.startswith('#'):
            parts = line_stripped.split()
            if len(parts) >= 2 and parts[0] == word:
                print(f"🗑️ Removing lock for: {word}")
                removed = True
                continue
        updated_lines.append(line)
    
    # Write updated lock file
    with open(lock_file, 'w') as f:
        f.writelines(updated_lines)
    
    return removed

def main():
    """Main execution following csv-prompt.md"""
    print("🚀 EXECUTING COMPLETE CSV GENERATION PROMPT")
    print("Following csv-prompt.md specifications\n")
    
    try:
        # Step 1: Garbage collection
        active_locks = clean_lock_file()
        
        # Step 2: Scan for target words
        words_needing_csv = scan_target_words()
        if not words_needing_csv:
            print("✅ All words already have CSV files")
            return
        
        # Step 3: Reserve single word
        secret_word = reserve_single_word(words_needing_csv, active_locks)
        if not secret_word:
            return
        
        # Step 4.1: Load ENABLE words
        enable_words = load_enable_words()
        if not enable_words:
            print(f"❌ Keeping lock for {secret_word} due to ENABLE loading failure")
            return
        
        # Verify secret word is in enable words
        if secret_word not in enable_words:
            print(f"❌ Secret word '{secret_word}' not in filtered ENABLE list")
            print(f"❌ Keeping lock for {secret_word}")
            return
        
        # Step 4.1.5: Generate meta file with category classification
        word_category = run_meta_generation(secret_word)
        print(f"📋 Word category determined: {word_category}")
        
        # Step 4.2: Load embeddings and compute rankings
        rankings = load_embeddings_and_compute_rankings(secret_word, enable_words)
        if not rankings:
            print(f"❌ Keeping lock for {secret_word} due to embeddings failure")
            return
        
        # Step 4.3: OpenAI expansion
        if not run_openai_expansion(secret_word):
            print(f"❌ Keeping lock for {secret_word} due to OpenAI expansion failure")
            return
        
        # Step 5: CSV generation
        if not run_csv_generation(secret_word):
            print(f"❌ Keeping lock for {secret_word} due to CSV generation failure")
            return
        
        # Step 6: Cleanup
        if remove_lock(secret_word):
            print(f"\n🎉 SUCCESSFULLY COMPLETED FULL PROCESSING FOR: {secret_word}")
        else:
            print(f"⚠️ Warning: Could not remove lock for {secret_word}")
    
    except Exception as e:
        print(f"❌ Error in main processing: {e}")
        print("Process may have been interrupted")

if __name__ == "__main__":
    main()
