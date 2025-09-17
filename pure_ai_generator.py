#!/usr/bin/env python3
"""
Pure AI Semantic Rank Generator - AI writes ALL clues up to rank 10,000
"""

import pandas as pd
import csv
import json
from difflib import SequenceMatcher
import time
import os

# Try to import OpenAI, but provide fallback if not available
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

class PureAISemanticRankGenerator:
    def __init__(self, secret_word="planet", batch_size=50):
        """Initialize the generator with a secret word and batch size."""
        self.secret_word = secret_word.lower()
        self.words = []
        self.rankings = None
        self.use_ai = False
        self.client = None
        self.batch_size = batch_size
        self.ai_cutoff_rank = 10000  # AI for ranks 1-10000, static message for 10001+
        
        # Try to set up AI
        if HAS_OPENAI:
            try:
                api_key = os.getenv('OPENAI_API_KEY')
                if api_key:
                    self.client = OpenAI(api_key=api_key)
                    self.use_ai = True
                    print(f"✅ AI enabled - will generate clues for ranks 1-{self.ai_cutoff_rank:,}")
                    print(f"📊 Batch size: {batch_size}")
                else:
                    print("❌ No OpenAI API key found - cannot generate AI clues")
                    return
            except Exception as e:
                print(f"❌ AI setup failed: {e}")
                return
        else:
            print("❌ OpenAI library not available")
            return
        
    def load_words(self, filename="data/enable1.txt"):
        """Load words from the ENABLE word list."""
        print(f"Loading words from {filename}...")
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                self.words = [word.strip().lower() for word in file.readlines()]
            print(f"Loaded {len(self.words):,} words")
            
            if self.secret_word not in self.words:
                print(f"Warning: Secret word '{self.secret_word}' not found in word list!")
                return False
            return True
        except FileNotFoundError:
            print(f"Error: Could not find word list file at {filename}")
            return False
    
    def calculate_similarity_score(self, word):
        """Calculate a similarity score for a word relative to the secret word."""
        if word == self.secret_word:
            return 1000
        
        score = 0
        
        # String similarity
        string_sim = SequenceMatcher(None, word, self.secret_word).ratio()
        score += string_sim * 100
        
        # Length similarity bonus
        len_diff = abs(len(word) - len(self.secret_word))
        if len_diff == 0:
            score += 50
        elif len_diff <= 2:
            score += 25
        
        # Common prefixes/suffixes
        if len(word) > 2 and len(self.secret_word) > 2:
            if word.startswith(self.secret_word[:2]) or self.secret_word.startswith(word[:2]):
                score += 30
            if word.endswith(self.secret_word[-2:]) or self.secret_word.endswith(word[-2:]):
                score += 30
        
        # Common letters bonus
        common_letters = set(word) & set(self.secret_word)
        score += len(common_letters) * 5
        
        # Rhyming bonus
        if len(word) > 3 and len(self.secret_word) > 3:
            if word[-3:] == self.secret_word[-3:]:
                score += 40
        
        # Add some randomness to avoid identical scores
        import random
        score += random.random()
        
        return score
    
    def compute_rankings(self):
        """Compute similarity rankings for all words."""
        print(f"Computing rankings relative to secret word: '{self.secret_word}'")
        
        # Calculate similarity scores
        word_scores = []
        for word in self.words:
            score = self.calculate_similarity_score(word)
            word_scores.append((word, score))
        
        # Sort by score (descending) and assign ranks
        word_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Create ranking dictionary
        self.rankings = {}
        for rank, (word, score) in enumerate(word_scores, 1):
            self.rankings[word] = {
                'rank': rank,
                'score': score
            }
        
        print(f"Rankings computed! Secret word '{self.secret_word}' has rank {self.rankings[self.secret_word]['rank']}")
        
        # Show distribution
        ai_words = sum(1 for data in self.rankings.values() if data['rank'] <= self.ai_cutoff_rank)
        static_words = len(self.words) - ai_words
        print(f"📊 AI clues needed: {ai_words:,} words (ranks 1-{self.ai_cutoff_rank:,})")
        print(f"📊 Static clues: {static_words:,} words (ranks {self.ai_cutoff_rank+1:,}+)")
    
    def generate_batched_ai_clues(self, word_batch):
        """Generate clues for a batch of words using AI."""
        if not self.use_ai:
            return {}
        
        try:
            # Filter out words that should get static clues
            ai_words = [(word, rank) for word, rank in word_batch if rank <= self.ai_cutoff_rank and rank > 1]
            
            if not ai_words:
                return {}
            
            # Create batch prompt
            word_descriptions = []
            for word, rank in ai_words:
                word_descriptions.append(f'"{word}" (rank {rank})')
            
            prompt = f"""You are writing clues for a word guessing game. The secret word is "{self.secret_word}".

For each word below, write a SHORT clue (maximum 5 words) that hints at the relationship between that word and "{self.secret_word}". Use "*" as a placeholder for "{self.secret_word}".

Be creative and specific. Think about:
- How the word relates to {self.secret_word}
- What connection players might recognize
- Keep it under 5 words
- Use * instead of "{self.secret_word}"

Words to create clues for:
{chr(10).join(word_descriptions)}

Respond ONLY with valid JSON in this exact format:
{{"word1": "clue text", "word2": "clue text"}}

Example good clues:
- "Orbits around *"
- "Red neighbor of *"
- "Tool to observe *"
- "Force that holds *"
"""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You write concise word game clues. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.8  # Slightly higher for more creative clues
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean up JSON response
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
            elif content.startswith('```'):
                content = content.replace('```', '').strip()
            
            try:
                clue_dict = json.loads(content)
                
                # Clean and validate clues
                cleaned_clues = {}
                for word, clue in clue_dict.items():
                    if isinstance(clue, str):
                        clue = clue.strip().strip('"').strip("'")
                        # Ensure max 5 words
                        words_in_clue = clue.split()
                        if len(words_in_clue) > 5:
                            clue = ' '.join(words_in_clue[:5])
                        cleaned_clues[word.lower()] = clue
                
                return cleaned_clues
                
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON parsing failed: {e}")
                print(f"Response was: {content[:200]}...")
                return {}
                
        except Exception as e:
            print(f"⚠️  AI batch request failed: {e}")
            return {}
    
    def generate_csv(self, output_filename=None):
        """Generate the complete CSV file with pure AI clues."""
        if output_filename is None:
            # Ensure secretword directory exists
            import os
            if not os.path.exists("secretword"):
                os.makedirs("secretword")
            output_filename = f"secretword/secretword-{self.secret_word}.csv"
        
        print(f"Generating CSV file: {output_filename}")
        
        if not self.use_ai:
            print("❌ Cannot generate - AI not available")
            return None
        
        # Get words that need AI clues (ranks 1 to ai_cutoff_rank)
        ai_words = [(word, self.rankings[word]['rank']) for word in self.words 
                   if self.rankings[word]['rank'] <= self.ai_cutoff_rank]
        ai_words.sort(key=lambda x: x[1])  # Sort by rank
        
        print(f"🤖 Generating AI clues for {len(ai_words):,} words (ranks 1-{self.ai_cutoff_rank:,})")
        
        # Estimate API calls and cost
        estimated_calls = (len(ai_words) // self.batch_size) + 1
        estimated_cost = estimated_calls * 0.003  # Rough estimate
        print(f"📞 Estimated API calls: ~{estimated_calls:,}")
        print(f"💰 Estimated cost: ~${estimated_cost:.2f}")
        
        # Store all clues
        all_clues = {}
        
        # Process AI words in batches
        successful_ai_clues = 0
        failed_ai_clues = 0
        
        for i in range(0, len(ai_words), self.batch_size):
            batch = ai_words[i:i + self.batch_size]
            
            if i % (self.batch_size * 5) == 0:  # Progress every 5 batches
                print(f"  Progress: {i:,} / {len(ai_words):,} AI words processed")
            
            # Handle secret word separately
            secret_word_in_batch = False
            for word, rank in batch:
                if rank == 1:
                    all_clues[word] = "This is the *."
                    secret_word_in_batch = True
                    break
            
            # Get AI clues for this batch
            ai_clues = self.generate_batched_ai_clues(batch)
            
            # Process each word in the batch
            for word, rank in batch:
                if rank == 1:
                    continue  # Already handled
                elif word in ai_clues:
                    all_clues[word] = ai_clues[word]
                    successful_ai_clues += 1
                else:
                    # If AI failed for this word, we have a problem since we want pure AI
                    print(f"⚠️  AI failed for '{word}' (rank {rank}) - this shouldn't happen with pure AI mode")
                    all_clues[word] = f"Connected to *"  # Emergency fallback
                    failed_ai_clues += 1
            
            # Small delay between batches
            time.sleep(0.5)
        
        print(f"✅ AI clue generation complete!")
        print(f"   Successful AI clues: {successful_ai_clues:,}")
        print(f"   Failed AI clues: {failed_ai_clues:,}")
        
        # Add static clues for words beyond the cutoff
        static_clue_count = 0
        for word in self.words:
            rank = self.rankings[word]['rank']
            if rank > self.ai_cutoff_rank:
                all_clues[word] = "* is more than 10K+ words away"
                static_clue_count += 1
        
        print(f"✅ Added {static_clue_count:,} static clues for distant words")
        
        # Create CSV data
        csv_data = []
        for word in self.words:
            rank = self.rankings[word]['rank']
            clue = all_clues.get(word, "Missing clue")
            
            csv_data.append({
                'rank': rank,
                'secret_word': self.secret_word,
                'word': word,
                'clue': clue
            })
        
        # Sort by rank
        csv_data.sort(key=lambda x: x['rank'])
        
        # Write to CSV
        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['rank', 'secret_word', 'word', 'clue']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in csv_data:
                writer.writerow(row)
        
        print(f"CSV file generated successfully: {output_filename}")
        print(f"Total rows: {len(csv_data):,}")
        
        # Show sample rows
        print("\nSample rows from the CSV:")
        df = pd.read_csv(output_filename)
        print("Top 15 (AI-generated clues):")
        print(df.head(15).to_string(index=False))
        
        print(f"\nSample distant words (static clues):")
        distant_sample = df[df['rank'] > self.ai_cutoff_rank].head(5)
        print(distant_sample.to_string(index=False))
        
        return output_filename
    
    def run_full_generation(self, output_filename=None):
        """Run the complete generation process."""
        print("=== Pure AI Semantic Rank CSV Generator ===")
        print(f"Secret word: {self.secret_word}")
        print(f"AI clue cutoff: rank {self.ai_cutoff_rank:,}")
        print(f"Batch size: {self.batch_size}")
        print()
        
        if not self.use_ai:
            print("❌ Cannot proceed without AI - please check your OpenAI API key")
            return False
        
        # Step 1: Load words
        if not self.load_words():
            return False
        
        # Step 2: Compute rankings
        self.compute_rankings()
        
        # Step 3: Generate CSV
        result = self.generate_csv(output_filename)
        
        if result:
            print("\n=== Generation Complete! ===")
            return True
        else:
            print("\n❌ Generation failed!")
            return False

def main():
    """Main function to run the generator."""
    secret_word = "elephant"
    batch_size = 50
    
    generator = PureAISemanticRankGenerator(secret_word, batch_size)
    
    # Use default naming convention: secretword-[secretword].csv
    success = generator.run_full_generation()
    
    if success:
        output_file = f"secretword/secretword-{secret_word}.csv"
        print(f"\n🎉 Success! Generated {output_file}")
        print("The CSV file is ready to use in your Semantic Rank game!")
        print(f"\nFile contains rankings for all {len(generator.words):,} ENABLE words")
        print("Columns: rank, secret_word, word, clue")
        print(f"✨ AI-generated clues for ranks 1-{generator.ai_cutoff_rank:,}")
        print(f"📝 Static message for ranks {generator.ai_cutoff_rank+1:,}+")
    else:
        print("❌ Generation failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
