#!/usr/bin/env python3
"""
Simple Semantic Rank Game CSV Generator

This script generates a precomputed CSV file for the Semantic Rank word game
using a lightweight approach that doesn't require heavy ML models.
It uses heuristic-based similarity ranking with semantic categories.
"""

import pandas as pd
import random
import csv
import re
from difflib import SequenceMatcher

class SimpleSemanticRankGenerator:
    def __init__(self, secret_word="planet"):
        """Initialize the generator with a secret word."""
        self.secret_word = secret_word.lower()
        self.words = []
        self.rankings = None
        
        # Define semantic categories for better ranking
        self.semantic_categories = {
            'planet': {
                'very_close': ['earth', 'mars', 'jupiter', 'saturn', 'venus', 'mercury', 'neptune', 'uranus', 'pluto'],
                'close': ['moon', 'sun', 'star', 'solar', 'orbit', 'space', 'cosmic', 'celestial', 'galaxy', 'universe', 
                         'asteroid', 'comet', 'meteorite', 'nebula', 'satellite'],
                'related': ['system', 'sphere', 'round', 'circular', 'rotation', 'gravity', 'atmosphere', 'surface',
                           'telescope', 'astronomy', 'science', 'discovery', 'exploration'],
                'somewhat_related': ['sky', 'night', 'light', 'bright', 'distance', 'far', 'big', 'small', 'rock', 'gas']
            }
        }
        
    def load_words(self, filename="data/enable1.txt"):
        """Load words from the ENABLE word list."""
        print(f"Loading words from {filename}...")
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                self.words = [word.strip().lower() for word in file.readlines()]
            print(f"Loaded {len(self.words):,} words")
            
            # Ensure secret word is in the list
            if self.secret_word not in self.words:
                print(f"Warning: Secret word '{self.secret_word}' not found in word list!")
                return False
            return True
        except FileNotFoundError:
            print(f"Error: Could not find word list file at {filename}")
            return False
    
    def calculate_similarity_score(self, word):
        """Calculate a similarity score for a word relative to the secret word."""
        score = 0
        
        # 1. Exact match
        if word == self.secret_word:
            return 1000
        
        # 2. Semantic category matching
        if self.secret_word in self.semantic_categories:
            categories = self.semantic_categories[self.secret_word]
            if word in categories.get('very_close', []):
                score += 800
            elif word in categories.get('close', []):
                score += 600
            elif word in categories.get('related', []):
                score += 400
            elif word in categories.get('somewhat_related', []):
                score += 200
        
        # 3. String similarity (for related words not in categories)
        string_sim = SequenceMatcher(None, word, self.secret_word).ratio()
        score += string_sim * 100
        
        # 4. Length similarity bonus
        len_diff = abs(len(word) - len(self.secret_word))
        if len_diff == 0:
            score += 50
        elif len_diff <= 2:
            score += 25
        
        # 5. Common prefixes/suffixes
        if len(word) > 2 and len(self.secret_word) > 2:
            if word.startswith(self.secret_word[:2]) or self.secret_word.startswith(word[:2]):
                score += 30
            if word.endswith(self.secret_word[-2:]) or self.secret_word.endswith(word[-2:]):
                score += 30
        
        # 6. Common letters bonus
        common_letters = set(word) & set(self.secret_word)
        score += len(common_letters) * 5
        
        # 7. Rhyming bonus (simple check for similar endings)
        if len(word) > 3 and len(self.secret_word) > 3:
            if word[-3:] == self.secret_word[-3:]:
                score += 40
        
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
        
        # Show top 10 for verification
        print("\nTop 10 most similar words:")
        for i, (word, score) in enumerate(word_scores[:10]):
            rank = i + 1
            print(f"{rank:2d}. {word:15s} (score: {score:.1f})")
    
    def generate_clue(self, word, rank):
        """Generate a clue based on the word's rank using tiered logic."""
        if rank == 1:  # Tier 1: Secret word
            return "This is the *."
        
        elif rank <= 1000:  # Tier 2: Closest words (AI-inspired patterns)
            patterns = [
                "A type of *.",
                "Another form of *.",
                "Often linked to *.",
                "Part of * system.",
                "Moves around *.",
                "Related to *.",
                "Found near *.",
                "Associated with *.",
                "Similar to *.",
                "Connected to *."
            ]
            return random.choice(patterns)
        
        elif rank <= 5000:  # Tier 3: Medium closeness
            patterns = [
                "Close to *.",
                "Often with *.",
                "Connected to *.",
                "Similar to *.",
                "Like *."
            ]
            return random.choice(patterns)
        
        elif rank <= 50000:  # Tier 4: Weak associations
            patterns = [
                "Far from *.",
                "Not usually with *.",
                "Different than *.",
                "Rarely linked to *.",
                "Unlike *."
            ]
            return random.choice(patterns)
        
        else:  # Tier 5: Distant words
            return f"* are nothing like {word}."
    
    def generate_csv(self, output_filename="semantic_rank_data.csv"):
        """Generate the complete CSV file with all words, ranks, and clues."""
        print(f"Generating CSV file: {output_filename}")
        
        # Prepare data
        csv_data = []
        
        print("Generating clues for all words...")
        for i, word in enumerate(self.words):
            if i % 10000 == 0:
                print(f"  Progress: {i:,} / {len(self.words):,} words processed")
            
            rank = self.rankings[word]['rank']
            clue = self.generate_clue(word, rank)
            
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
        
        # Show some random middle-tier examples
        print(f"\nSample middle-tier words (ranks 1000-5000):")
        middle_tier = df[(df['rank'] >= 1000) & (df['rank'] <= 5000)].sample(min(10, len(df)))
        print(middle_tier.to_string(index=False))
        
        return output_filename
    
    def run_full_generation(self, output_filename="semantic_rank_data.csv"):
        """Run the complete generation process."""
        print("=== Simple Semantic Rank CSV Generator ===")
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
    # You can change the secret word here
    secret_word = "planet"
    
    generator = SimpleSemanticRankGenerator(secret_word)
    
    output_file = f"semantic_rank_{secret_word}.csv"
    success = generator.run_full_generation(output_file)
    
    if success:
        print(f"\n🎉 Success! Generated {output_file}")
        print("The CSV file is ready to use in your Semantic Rank game!")
        print(f"\nFile contains rankings for all {len(generator.words):,} ENABLE words")
        print("Columns: rank, secret_word, word, clue")
    else:
        print("❌ Generation failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
