#!/usr/bin/env python3
"""
Regenerate 20 words using proper two-pass OpenAI expansion logic
This implements the same logic as the original csv-prompt system
"""

import sys
import os
import openai
from pathlib import Path
from datetime import datetime
import time
import re

class ProperTwoPassExpander:
    """Implements the real two-pass expansion logic"""
    
    def __init__(self, secret_word):
        self.secret_word = secret_word.lower().strip()
        self.api_calls = 0
        
        # Set up OpenAI
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise Exception("No OpenAI API key found")
        openai.api_key = api_key
    
    def get_primary_associations(self, min_words=300):
        """Pass 1: Get primary associations from OpenAI"""
        print(f"🔍 Getting primary associations (minimum {min_words} words)")
        
        # Use the same prompt strategy as the original system
        strategies = [
            {
                'name': 'Comprehensive Strategy',
                'max_words': 500,
                'max_tokens': 2000,
                'prompt_template': """Give me {max_words} words that are semantically related to "{word}". Include:
- Direct synonyms and related terms
- Associated concepts and ideas  
- Objects, places, actions connected to this word
- Emotional or conceptual associations
- Words that would appear in similar contexts

Format as a simple numbered list (1. word, 2. word, etc.). Be comprehensive and diverse."""
            },
            {
                'name': 'Contextual Strategy', 
                'max_words': 400,
                'max_tokens': 1500,
                'prompt_template': """List {max_words} words related to "{word}" including synonyms, associated objects, related concepts, and contextual connections. Numbered list format."""
            },
            {
                'name': 'Simple Strategy',
                'max_words': 300,
                'max_tokens': 1000, 
                'prompt_template': """List {max_words} words similar to "{word}". Numbered list only."""
            }
        ]
        
        for strategy in strategies:
            print(f"🤖 Trying {strategy['name']}: max {strategy['max_words']} words...")
            
            prompt = strategy['prompt_template'].format(
                word=self.secret_word,
                max_words=strategy['max_words']
            )
            
            try:
                words = self._try_openai_request(prompt, strategy['max_tokens'])
                if words and len(words) >= min_words * 0.8:  # Accept if we get at least 80% of target
                    print(f"✅ {strategy['name']} successful: got {len(words)} words")
                    return words
                else:
                    print(f"❌ {strategy['name']} insufficient: got {len(words) if words else 0} words, trying next...")
            except Exception as e:
                print(f"❌ {strategy['name']} failed: {e}")
                continue
        
        print("❌ All primary strategies failed")
        return []
    
    def get_synonym_expansion(self, primary_words):
        """Pass 2: Get synonyms for each primary word"""
        print(f"🔄 Getting synonyms for {len(primary_words)} primary words")
        
        all_synonyms = []
        batch_size = 50
        
        for i in range(0, len(primary_words), batch_size):
            batch = primary_words[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            print(f"🤖 Getting synonyms for batch {batch_num} ({len(batch)} words)")
            
            # Create batch prompt for synonyms
            words_str = ", ".join(batch)
            prompt = f"""For each of these words: {words_str}

Give me 3-8 synonyms or closely related words for each. Format as a simple list, one word per line, no numbers or explanations. Include the original words too."""
            
            try:
                batch_synonyms = self._try_openai_request(prompt, 2000)
                if batch_synonyms:
                    all_synonyms.extend(batch_synonyms)
                    print(f"✅ Batch {batch_num}: got {len(batch_synonyms)} synonyms")
                else:
                    print(f"❌ Batch {batch_num}: failed")
                
                # Small delay between batches
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Batch {batch_num} error: {e}")
                continue
        
        # Remove duplicates while preserving order
        unique_synonyms = []
        seen = set()
        for word in all_synonyms:
            if word.lower() not in seen:
                unique_synonyms.append(word)
                seen.add(word.lower())
        
        print(f"✅ Synonym expansion: {len(all_synonyms)} total → {len(unique_synonyms)} unique synonyms")
        return unique_synonyms
    
    def _try_openai_request(self, prompt, max_tokens):
        """Make an OpenAI API request and parse the response"""
        try:
            self.api_calls += 1
            response = openai.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=max_tokens,
                temperature=0.3
            )
            
            content = response.choices[0].message.content.strip()
            return self._parse_and_clean_response(content)
            
        except Exception as e:
            print(f"❌ OpenAI API error: {e}")
            raise e
    
    def _parse_and_clean_response(self, content):
        """Parse OpenAI response and extract clean words"""
        words = []
        
        # Split into lines and process each
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Remove numbered list format (1. word, 2. word, etc.)
            line = re.sub(r'^\d+\.\s*', '', line)
            
            # Remove bullet points and other formatting
            line = re.sub(r'^[-•*]\s*', '', line)
            
            # Split by commas if present
            if ',' in line:
                parts = [p.strip() for p in line.split(',')]
            else:
                parts = [line]
            
            for part in parts:
                part = part.strip()
                if part and len(part) > 1 and part.replace(' ', '').replace('-', '').isalpha():
                    # Clean up the word
                    clean_word = part.lower().strip()
                    if clean_word and len(clean_word) > 1:
                        words.append(clean_word)
        
        return words
    
    def get_two_pass_expansion(self):
        """Perform complete two-pass expansion"""
        print(f"🚀 Starting two-pass expansion for '{self.secret_word}'")
        
        # Pass 1: Primary associations
        primary_words = self.get_primary_associations(300)
        if not primary_words:
            print("❌ Primary pass failed")
            return []
        
        print(f"✅ First pass completed: {len(primary_words)} primary words")
        
        # Remove duplicates from primary words
        unique_primary = []
        seen = set()
        for word in primary_words:
            if word.lower() not in seen:
                unique_primary.append(word)
                seen.add(word.lower())
        
        print(f"✅ After deduplication: {len(unique_primary)} unique primary words")
        
        # Pass 2: Synonym expansion
        synonyms = self.get_synonym_expansion(unique_primary)
        
        # Combine primary and synonyms
        all_words = unique_primary + synonyms
        
        # Final deduplication
        final_words = []
        seen = set()
        for word in all_words:
            if word.lower() not in seen:
                final_words.append(word)
                seen.add(word.lower())
        
        print(f"✅ Two-pass expansion completed!")
        print(f"📊 Results: {len(unique_primary)} primary + {len(synonyms)} synonyms = {len(final_words)} unique total")
        
        return final_words

def regenerate_word(word):
    """Regenerate a single word with proper two-pass logic"""
    print(f"\n{'='*60}")
    print(f"🔄 REGENERATING: {word}")
    print(f"{'='*60}")
    
    try:
        # Create expander
        expander = ProperTwoPassExpander(word)
        
        # Get two-pass expansion
        words = expander.get_two_pass_expansion()
        
        if not words:
            print(f"❌ Failed to generate words for {word}")
            return False
        
        # Save to file
        twopass_file = Path(f'secretword/{word}-openai-twopass.txt')
        
        with open(twopass_file, 'w', encoding='utf-8') as f:
            f.write(f"# Two-pass expansion for '{word}'\n")
            f.write(f"# Total words: {len(words)}\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
            f.write(', '.join(words))
        
        print(f"✅ SUCCESS: Generated {len(words)} words using {expander.api_calls} API calls")
        print(f"📁 Saved to: {word}-openai-twopass.txt")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR regenerating {word}: {e}")
        return False

def main():
    """Regenerate the 20 words with proper two-pass logic"""
    
    # The 20 words we generated with simple logic
    words_to_regenerate = [
        'island', 'joy', 'juice', 'jump', 'key', 'kitchen', 'lake', 'lamp', 
        'laugh', 'leaf', 'library', 'life', 'light', 'lion', 'listen', 
        'love', 'luck', 'magnet', 'market', 'meat'
    ]
    
    print(f"🚀 REGENERATING {len(words_to_regenerate)} WORDS WITH PROPER TWO-PASS LOGIC")
    print(f"Words: {', '.join(words_to_regenerate)}")
    print(f"This will use significantly more API calls but generate much richer results!")
    
    success_count = 0
    failed_words = []
    total_api_calls = 0
    
    for i, word in enumerate(words_to_regenerate, 1):
        print(f"\n[{i}/{len(words_to_regenerate)}] Processing: {word}")
        
        success = regenerate_word(word)
        if success:
            success_count += 1
        else:
            failed_words.append(word)
        
        # Small delay between words
        if i < len(words_to_regenerate):
            print("⏸️ Pausing 3 seconds between words...")
            time.sleep(3)
    
    # Summary
    print(f"\n🎯 REGENERATION COMPLETE")
    print(f"✅ Successful: {success_count}/{len(words_to_regenerate)}")
    
    if failed_words:
        print(f"❌ Failed: {len(failed_words)} words")
        print(f"Failed words: {', '.join(failed_words)}")
    else:
        print(f"🎉 All words regenerated successfully!")

if __name__ == "__main__":
    main()

