#!/usr/bin/env python3
"""
Loop the csv-prompt process until all words are processed
"""

import subprocess
import sys
import time
from pathlib import Path

def count_words_needing_csv():
    """Count how many words still need CSV generation"""
    secretword_dir = Path("secretword")
    
    # Get all embedding files
    embedding_files = []
    for file in secretword_dir.glob("embeddings-*.txt"):
        word = file.stem.replace("embeddings-", "")
        # Remove suffixes like '2', '-clean' to get base word
        import re
        base_word = re.sub(r'2$', '', word)
        base_word = re.sub(r'-clean$', '', base_word)
        embedding_files.append(base_word)
    
    embedding_words = sorted(set(embedding_files))
    
    # Get existing CSV files
    csv_files = []
    for file in secretword_dir.glob("secretword-easy-animals-*.csv"):
        word = file.stem.replace("secretword-easy-animals-", "")
        # Skip temp/backup/incomplete files
        if not re.search(r'_(temp|backup|incomplete)$', word):
            csv_files.append(word)
    
    csv_words = sorted(set(csv_files))
    
    # Find words needing CSV generation
    words_needing_csv = [word for word in embedding_words if word not in csv_words]
    
    return len(words_needing_csv), words_needing_csv, len(embedding_words), len(csv_words)

def main():
    """Main loop to process all words"""
    print("🔄 STARTING CSV GENERATION LOOP")
    print("Will process all words until none remain")
    print("="*60)
    
    iteration = 0
    total_processed = 0
    
    while True:
        iteration += 1
        
        # Check current status
        words_remaining, words_list, total_embeddings, total_csvs = count_words_needing_csv()
        
        print(f"\n🔍 ITERATION {iteration} - STATUS CHECK")
        print(f"📂 Total words with embeddings: {total_embeddings}")
        print(f"✅ Words with CSV files: {total_csvs}")
        print(f"⏳ Words remaining to process: {words_remaining}")
        
        if words_remaining == 0:
            print(f"\n🎉 ALL WORDS PROCESSED!")
            print(f"📊 Total iterations: {iteration - 1}")
            print(f"📈 Total words processed in this session: {total_processed}")
            break
        
        print(f"🎯 Next words to process: {', '.join(words_list[:5])}")
        if len(words_list) > 5:
            print(f"    ... and {len(words_list) - 5} more")
        
        print(f"\n🚀 RUNNING CSV-PROMPT (Iteration {iteration})")
        print("="*60)
        
        # Run the csv-prompt process
        try:
            result = subprocess.run([
                sys.executable, "run_csv_prompt.py"
            ], capture_output=False, text=True, encoding='utf-8', errors='replace')
            
            if result.returncode == 0:
                print(f"✅ Iteration {iteration} completed successfully")
                total_processed += 1
            else:
                print(f"❌ Iteration {iteration} failed with exit code: {result.returncode}")
                print("⏸️ Continuing to next iteration...")
                
        except KeyboardInterrupt:
            print(f"\n⏹️ Process interrupted by user")
            print(f"📊 Completed {total_processed} words in {iteration - 1} iterations")
            break
        except Exception as e:
            print(f"❌ Error running csv-prompt: {e}")
            print("⏸️ Waiting 5 seconds before retry...")
            time.sleep(5)
        
        # Brief pause between iterations
        print(f"\n⏱️ Pausing 3 seconds before next iteration...")
        time.sleep(3)
    
    print(f"\n📋 FINAL SUMMARY:")
    final_remaining, _, final_total_embeddings, final_total_csvs = count_words_needing_csv()
    print(f"📂 Total words with embeddings: {final_total_embeddings}")
    print(f"✅ Total CSV files created: {final_total_csvs}")
    print(f"⏳ Words still remaining: {final_remaining}")
    print(f"🔄 Loop completed after {iteration} iterations")

if __name__ == "__main__":
    main()

