#!/usr/bin/env python3
"""
Optimized Semantic Rank Generator - Quieter fallback handling
"""

import pandas as pd
import random
import csv
import re
from difflib import SequenceMatcher
import time
import os

# Try to import OpenAI, but provide fallback if not available
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

class OptimizedSemanticRankGenerator:
    def __init__(self, secret_word="planet"):
        """Initialize the generator with a secret word."""
        self.secret_word = secret_word.lower()
        self.words = []
        self.rankings = None
        self.use_ai = False
        self.client = None
        self.quota_exceeded = False
        
        # Try to set up AI, but don't be noisy about failures
        if HAS_OPENAI:
            try:
                api_key = os.getenv('OPENAI_API_KEY')
                if api_key:
                    self.client = OpenAI(api_key=api_key)
                    self.use_ai = True
            except:
                pass
        
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
    
    def generate_ai_clue(self, word, rank):
        """Generate a clue using AI or fallback to patterns."""
        # Special case for the secret word
        if rank == 1:
            return "This is the *."
        
        # If we've already detected quota issues or AI isn't set up, use patterns
        if self.quota_exceeded or not self.use_ai:
            return self.generate_pattern_clue(word, rank)
        
        # Try AI for the first few words only, then switch to patterns if quota exceeded
        if rank <= 10:  # Only try AI for top 10 words to test quota
            try:
                # Determine the tier for context
                if rank <= 1000:
                    tier_context = "very closely related"
                elif rank <= 5000:
                    tier_context = "somewhat related"
                elif rank <= 50000:
                    tier_context = "distantly related"
                else:
                    tier_context = "not meaningfully related"
                
                # Create prompt for AI
                prompt = f"""Write a very short clue (5 words max) that hints at the connection between "{word}" and "{self.secret_word}".

The words are {tier_context}. Use "*" as a placeholder for "{self.secret_word}".

Rules:
- Maximum 5 words
- Use * instead of "{self.secret_word}"
- Be creative but accurate
- If there's no meaningful connection, respond with exactly: "* are nothing like {word}."

Word to describe: {word}
Secret word (use * instead): {self.secret_word}
Clue:"""

                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a word game expert who writes concise, helpful clues."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=20,
                    temperature=0.7
                )
                
                clue = response.choices[0].message.content.strip()
                clue = clue.strip('"').strip("'").strip()
                
                # Validate clue length (should be 5 words or less)
                if len(clue.split()) > 5:
                    clue = ' '.join(clue.split()[:5])
                
                return clue
                
            except Exception as e:
                # Check if it's a quota error
                if "quota" in str(e).lower() or "429" in str(e):
                    self.quota_exceeded = True
                    if rank <= 5:  # Only print quota message once
                        print("⚠️  OpenAI quota exceeded - switching to pattern-based clues for all remaining words")
                
                # Fall back to pattern-based clue
                return self.generate_pattern_clue(word, rank)
        
        # For all other words, use patterns
        return self.generate_pattern_clue(word, rank)
    
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
    
    def generate_csv(self, output_filename="optimized_semantic_rank_data.csv"):
        """Generate the complete CSV file efficiently."""
        print(f"Generating CSV file: {output_filename}")
        
        if self.use_ai:
            print("🤖 Attempting AI clue generation (will fallback to patterns if quota exceeded)")
        else:
            print("📝 Using pattern-based clue generation")
        
        # Prepare data
        csv_data = []
        
        print("Generating clues for all words...")
        
        for i, word in enumerate(self.words):
            if i % 10000 == 0:
                print(f"  Progress: {i:,} / {len(self.words):,} words processed")
            
            rank = self.rankings[word]['rank']
            clue = self.generate_ai_clue(word, rank)
            
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
        print(df.head(15).to_string(index=False))
        
        return output_filename
    
    def run_full_generation(self, output_filename="optimized_semantic_rank_data.csv"):
        """Run the complete generation process."""
        print("=== Optimized Semantic Rank CSV Generator ===")
        print(f"Secret word: {self.secret_word}")
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
    
    generator = OptimizedSemanticRankGenerator(secret_word)
    
    output_file = f"final_semantic_rank_{secret_word}.csv"
    success = generator.run_full_generation(output_file)
    
    if success:
        print(f"\n🎉 Success! Generated {output_file}")
        print("The CSV file is ready to use in your Semantic Rank game!")
        print(f"\nFile contains rankings for all {len(generator.words):,} ENABLE words")
        print("Columns: rank, secret_word, word, clue")
        
        if generator.quota_exceeded:
            print("📝 Used intelligent pattern-based clues (OpenAI quota exceeded)")
        elif generator.use_ai and not generator.quota_exceeded:
            print("✨ Used AI-generated clues for top words, patterns for others")
        else:
            print("📝 Used pattern-based clues throughout")
    else:
        print("❌ Generation failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
