#!/usr/bin/env python3
"""
Batched AI Semantic Rank Generator - Efficient API usage with word batching
"""

import pandas as pd
import random
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

class BatchedAISemanticRankGenerator:
    def __init__(self, secret_word="planet", batch_size=50):
        """Initialize the generator with a secret word and batch size."""
        self.secret_word = secret_word.lower()
        self.words = []
        self.rankings = None
        self.use_ai = False
        self.client = None
        self.quota_exceeded = False
        self.batch_size = batch_size
        
        # Try to set up AI
        if HAS_OPENAI:
            try:
                api_key = os.getenv('OPENAI_API_KEY')
                if api_key:
                    self.client = OpenAI(api_key=api_key)
                    self.use_ai = True
                    print(f"✅ AI enabled with batch size: {batch_size}")
                else:
                    print("⚠️  No OpenAI API key found")
            except Exception as e:
                print(f"⚠️  AI setup failed: {e}")
        
        if not self.use_ai:
            print("📝 Using pattern-based clue generation")
        
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
    
    def generate_batched_ai_clues(self, word_batch):
        """Generate clues for a batch of words using AI."""
        if not self.use_ai or self.quota_exceeded:
            return {}
        
        try:
            # Create batch prompt
            word_list = []
            for word, rank in word_batch:
                if rank == 1:
                    continue  # Skip secret word, we handle it separately
                
                # Determine tier context
                if rank <= 1000:
                    tier = "very closely related"
                elif rank <= 5000:
                    tier = "somewhat related"
                elif rank <= 50000:
                    tier = "distantly related"
                else:
                    tier = "not meaningfully related"
                
                word_list.append(f'"{word}" (rank {rank}, {tier})')
            
            if not word_list:
                return {}
            
            prompt = f"""Generate short clues (max 5 words each) for these words relative to "{self.secret_word}".

Use "*" as placeholder for "{self.secret_word}".

Rules:
- Max 5 words per clue
- Use * instead of "{self.secret_word}"
- If no meaningful connection exists, use: "* are nothing like [word]"

Words to create clues for:
{chr(10).join(word_list)}

Respond with JSON format:
{{"word1": "clue text", "word2": "clue text", ...}}

Examples of good clues:
- "Part of * system"
- "Often found near *"
- "Similar to *"
- "* are nothing like spoon"
"""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a word game expert. Generate concise clues in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,  # Increased for batch processing
                temperature=0.7
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                # Clean up the response (remove markdown formatting if present)
                if content.startswith('```json'):
                    content = content.replace('```json', '').replace('```', '').strip()
                elif content.startswith('```'):
                    content = content.replace('```', '').strip()
                
                clue_dict = json.loads(content)
                
                # Validate and clean clues
                cleaned_clues = {}
                for word, clue in clue_dict.items():
                    if isinstance(clue, str):
                        clue = clue.strip().strip('"').strip("'")
                        # Ensure max 5 words
                        if len(clue.split()) > 5:
                            clue = ' '.join(clue.split()[:5])
                        cleaned_clues[word.lower()] = clue
                
                return cleaned_clues
                
            except json.JSONDecodeError:
                print(f"⚠️  Failed to parse AI response as JSON: {content[:100]}...")
                return {}
                
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                self.quota_exceeded = True
                print("⚠️  OpenAI quota exceeded - switching to pattern-based clues")
            else:
                print(f"⚠️  AI batch request failed: {e}")
            return {}
    
    def generate_pattern_clue(self, word, rank):
        """Generate a pattern-based clue."""
        if rank == 1:
            return "This is the *."
        elif rank <= 1000:
            patterns = [
                "A type of *.", "Related to *.", "Found near *.", 
                "Associated with *.", "Similar to *.", "Connected to *.",
                "Part of * system.", "Often with *."
            ]
            return random.choice(patterns)
        elif rank <= 5000:
            patterns = [
                "Close to *.", "Often with *.", "Connected to *.", 
                "Similar to *.", "Like *."
            ]
            return random.choice(patterns)
        elif rank <= 50000:
            patterns = [
                "Far from *.", "Not usually with *.", "Different than *.", 
                "Rarely linked to *.", "Unlike *."
            ]
            return random.choice(patterns)
        else:
            return f"* are nothing like {word}."
    
    def generate_csv(self, output_filename="batched_semantic_rank_data.csv"):
        """Generate the complete CSV file with batched AI clues."""
        print(f"Generating CSV file: {output_filename}")
        
        if self.use_ai:
            print(f"🤖 Using batched AI clue generation (batch size: {self.batch_size})")
            estimated_calls = len(self.words) // self.batch_size + 1
            print(f"📊 Estimated API calls: ~{estimated_calls:,} (vs {len(self.words):,} individual calls)")
        else:
            print("📝 Using pattern-based clue generation")
        
        # Prepare data with ranks
        word_rank_pairs = [(word, self.rankings[word]['rank']) for word in self.words]
        
        # Sort by rank for processing
        word_rank_pairs.sort(key=lambda x: x[1])
        
        # Store all clues
        all_clues = {}
        
        if self.use_ai and not self.quota_exceeded:
            # Process in batches for AI
            print("Generating batched AI clues...")
            
            for i in range(0, len(word_rank_pairs), self.batch_size):
                batch = word_rank_pairs[i:i + self.batch_size]
                
                if i % (self.batch_size * 10) == 0:  # Progress every 10 batches
                    print(f"  Progress: {i:,} / {len(word_rank_pairs):,} words processed")
                
                # Get AI clues for this batch
                ai_clues = self.generate_batched_ai_clues(batch)
                
                # Process each word in the batch
                for word, rank in batch:
                    if rank == 1:
                        all_clues[word] = "This is the *."
                    elif word in ai_clues:
                        all_clues[word] = ai_clues[word]
                    else:
                        # Fallback to pattern
                        all_clues[word] = self.generate_pattern_clue(word, rank)
                
                # Small delay between batches to be respectful
                if not self.quota_exceeded:
                    time.sleep(0.5)
                
                # If quota exceeded, switch to patterns for remaining words
                if self.quota_exceeded:
                    print(f"  Switching to patterns for remaining {len(word_rank_pairs) - i} words...")
                    for word, rank in word_rank_pairs[i:]:
                        if word not in all_clues:
                            all_clues[word] = self.generate_pattern_clue(word, rank)
                    break
        
        # Fill in any missing clues with patterns
        for word, rank in word_rank_pairs:
            if word not in all_clues:
                all_clues[word] = self.generate_pattern_clue(word, rank)
        
        # Create CSV data
        csv_data = []
        for word, rank in word_rank_pairs:
            csv_data.append({
                'rank': rank,
                'secret_word': self.secret_word,
                'word': word,
                'clue': all_clues[word]
            })
        
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
        print(df.head(15).to_string(index=False))
        
        # Count AI vs pattern clues
        if self.use_ai:
            ai_count = sum(1 for clue in all_clues.values() 
                          if not any(pattern in clue for pattern in ["A type of", "Related to", "Found near", "Associated with", "Similar to", "Connected to", "Part of", "Often with", "Close to", "Like", "Far from", "Not usually", "Different than", "Rarely linked", "Unlike", "nothing like"]))
            print(f"\n📊 Clue Statistics:")
            print(f"  AI-generated clues: ~{ai_count:,}")
            print(f"  Pattern-based clues: ~{len(all_clues) - ai_count:,}")
        
        return output_filename
    
    def run_full_generation(self, output_filename="batched_semantic_rank_data.csv"):
        """Run the complete generation process."""
        print("=== Batched AI Semantic Rank CSV Generator ===")
        print(f"Secret word: {self.secret_word}")
        print(f"Batch size: {self.batch_size}")
        print()
        
        # Step 1: Load words
        if not self.load_words():
            return False
        
        # Step 2: Compute rankings
        self.compute_rankings()
        
        # Step 3: Generate CSV
        self.generate_csv(output_filename)
        
        print("\n=== Generation Complete! ===")
        return True

def main():
    """Main function to run the generator."""
    secret_word = "planet"
    batch_size = 50  # Adjust this based on your needs (10-100 words per batch)
    
    generator = BatchedAISemanticRankGenerator(secret_word, batch_size)
    
    output_file = f"batched_ai_semantic_rank_{secret_word}.csv"
    success = generator.run_full_generation(output_file)
    
    if success:
        print(f"\n🎉 Success! Generated {output_file}")
        print("The CSV file is ready to use in your Semantic Rank game!")
        print(f"\nFile contains rankings for all {len(generator.words):,} ENABLE words")
        print("Columns: rank, secret_word, word, clue")
        
        if generator.use_ai:
            estimated_calls = len(generator.words) // batch_size + 1
            estimated_cost = estimated_calls * 0.002  # Rough estimate
            print(f"💰 Estimated API cost: ~${estimated_cost:.2f} (vs ~${len(generator.words) * 0.0001:.2f} individual)")
            print(f"📞 API calls made: ~{estimated_calls:,} (vs {len(generator.words):,} individual)")
        
    else:
        print("❌ Generation failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
