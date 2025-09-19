#!/usr/bin/env python3
"""
True Semantic Rank Generator - Uses OpenAI embeddings for semantic similarity
"""

import os
import json
import time
import math
import pandas as pd
from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

class SemanticEmbeddingGenerator:
    def __init__(self, secret_word, batch_size=50, ai_cutoff_rank=10000, openai_api_key=None):
        self.secret_word = secret_word.lower()
        self.batch_size = batch_size
        self.ai_cutoff_rank = ai_cutoff_rank
        self.words = []
        self.rankings = {}
        self.client = None
        self.use_ai = False
        
        if openai_api_key:
            self.client = OpenAI(api_key=openai_api_key)
            self.use_ai = True
        elif os.getenv("OPENAI_API_KEY"):
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.use_ai = True
        else:
            print("Warning: No OPENAI_API_KEY found in environment variables or provided.")
            print("Cannot generate semantic embeddings or AI clues.")

        if not self.use_ai:
            print("❌ Cannot proceed without OpenAI API access")

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
            self.words = []
            return False

    def get_embeddings_batch(self, texts, model="text-embedding-3-large"):
        """Get embeddings for a batch of texts using OpenAI API."""
        if not self.use_ai:
            return None
        
        try:
            response = self.client.embeddings.create(
                model=model,
                input=texts,
                encoding_format="float"
            )
            
            embeddings = []
            for data in response.data:
                embeddings.append(data.embedding)
            
            return np.array(embeddings)
            
        except RateLimitError:
            print("⚠️  OpenAI Rate Limit Exceeded. Retrying after delay...")
            time.sleep(20)
            return self.get_embeddings_batch(texts, model)
        except APIConnectionError as e:
            print(f"⚠️  OpenAI API Connection Error: {e}. Retrying after delay...")
            time.sleep(10)
            return self.get_embeddings_batch(texts, model)
        except APIStatusError as e:
            print(f"⚠️  OpenAI API Status Error: {e.status_code}. Retrying after delay...")
            time.sleep(10)
            return self.get_embeddings_batch(texts, model)
        except Exception as e:
            print(f"⚠️  Embedding request failed: {e}")
            return None

    def load_embeddings_cache(self, cache_path=".env/embeddings.json"):
        """Load precomputed embeddings from cache file."""
        print(f"Loading embeddings cache from {cache_path}...")
        
        try:
            if not os.path.exists(cache_path):
                print(f"❌ Cache file not found at {cache_path}")
                return None
            
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            
            print(f"✅ Loaded embeddings for {len(cache):,} words from cache")
            return cache
            
        except Exception as e:
            print(f"❌ Error loading embeddings cache: {e}")
            return None

    def compute_semantic_rankings_from_cache(self, cache_path=".env/embeddings.json"):
        """Compute semantic rankings using precomputed embeddings from cache."""
        print(f"Computing semantic rankings from cached embeddings...")
        print(f"Secret word: '{self.secret_word}'")
        
        # Load the embeddings cache
        embeddings_cache = self.load_embeddings_cache(cache_path)
        if embeddings_cache is None:
            print("❌ Failed to load embeddings cache")
            return False
        
        # Check if secret word is in cache
        if self.secret_word not in embeddings_cache:
            print(f"❌ Secret word '{self.secret_word}' not found in embeddings cache")
            return False
        
        # Get secret word embedding
        secret_embedding = np.array(embeddings_cache[self.secret_word])
        secret_embedding = secret_embedding / np.linalg.norm(secret_embedding)  # Normalize
        
        print(f"Processing {len(self.words):,} words using cached embeddings...")
        
        # Compute similarities for all words
        word_similarities = []
        
        for word in tqdm(self.words, desc="Computing similarities"):
            if word in embeddings_cache:
                # Get cached embedding
                word_embedding = np.array(embeddings_cache[word])
                word_embedding = word_embedding / np.linalg.norm(word_embedding)  # Normalize
                
                # Compute cosine similarity
                similarity = np.dot(word_embedding, secret_embedding)
                word_similarities.append((word, similarity))
            else:
                # Word not in cache - assign very low similarity
                print(f"⚠️  Word '{word}' not found in cache, assigning low similarity")
                word_similarities.append((word, -1.0))
        
        print(f"Computed similarities for {len(word_similarities):,} words")
        
        # Sort by similarity (descending), then by word alphabetically for tie-breaking
        word_similarities.sort(key=lambda x: (-x[1], x[0]))
        
        # Assign ranks
        self.rankings = {}
        for rank, (word, similarity) in enumerate(word_similarities, 1):
            self.rankings[word] = {
                'rank': rank,
                'similarity': similarity
            }
        
        # Diagnostic logging
        print(f"\n🔍 Semantic similarity analysis:")
        print(f"Secret word '{self.secret_word}' rank: {self.rankings[self.secret_word]['rank']}")
        print(f"Secret word similarity: {self.rankings[self.secret_word]['similarity']:.6f}")
        
        # Show top 10 most similar words
        top_words = sorted(self.rankings.items(), key=lambda x: x[1]['rank'])[:10]
        print(f"\nTop 10 most similar words to '{self.secret_word}':")
        for word, data in top_words:
            print(f"  {data['rank']:4d}. {word:<15} (similarity: {data['similarity']:.6f})")
        
        # Compare with orthographic similarity for diagnostics
        print(f"\nDiagnostic: Comparing semantic vs orthographic similarity...")
        test_words = [word for word, _ in top_words[1:6]]  # Skip secret word itself
        for word in test_words:
            semantic_sim = self.rankings[word]['similarity']
            ortho_sim = SequenceMatcher(None, self.secret_word, word).ratio()
            print(f"  {word:<15}: semantic={semantic_sim:.4f}, orthographic={ortho_sim:.4f}")
        
        return True

    def compute_semantic_rankings(self):
        """Compute semantic rankings using OpenAI embeddings (fallback if no cache)."""
        print(f"Computing semantic rankings relative to secret word: '{self.secret_word}'")
        
        if not self.use_ai:
            print("❌ Cannot compute semantic rankings without OpenAI API")
            return False
        
        # Get embedding for the secret word first
        print("Getting embedding for secret word...")
        secret_embedding = self.get_embeddings_batch([self.secret_word])
        if secret_embedding is None:
            print("❌ Failed to get secret word embedding")
            return False
        
        secret_embedding = secret_embedding[0]  # Extract single embedding
        secret_embedding = secret_embedding / np.linalg.norm(secret_embedding)  # Normalize
        
        print(f"Processing {len(self.words):,} words in batches of {self.batch_size}")
        
        # Process words in batches to get embeddings
        all_similarities = []
        word_order = []
        
        for i in tqdm(range(0, len(self.words), self.batch_size), desc="Computing embeddings"):
            batch_words = self.words[i:i + self.batch_size]
            
            # Get embeddings for this batch
            batch_embeddings = self.get_embeddings_batch(batch_words)
            if batch_embeddings is None:
                print(f"❌ Failed to get embeddings for batch starting at {i}")
                continue
            
            # Normalize embeddings to unit vectors
            batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
            
            # Compute cosine similarities
            similarities = cosine_similarity(batch_embeddings, secret_embedding.reshape(1, -1)).flatten()
            
            all_similarities.extend(similarities)
            word_order.extend(batch_words)
            
            # Small delay to respect rate limits
            time.sleep(0.1)
        
        print(f"Computed similarities for {len(word_order):,} words")
        
        # Create word-similarity pairs and sort
        word_similarity_pairs = list(zip(word_order, all_similarities))
        
        # Sort by similarity (descending), then by word alphabetically for tie-breaking
        word_similarity_pairs.sort(key=lambda x: (-x[1], x[0]))
        
        # Assign ranks
        self.rankings = {}
        for rank, (word, similarity) in enumerate(word_similarity_pairs, 1):
            self.rankings[word] = {
                'rank': rank,
                'similarity': similarity
            }
        
        # Verify secret word is rank 1
        secret_rank = self.rankings[self.secret_word]['rank']
        print(f"✅ Secret word '{self.secret_word}' has rank {secret_rank}")
        
        if secret_rank != 1:
            print("⚠️  Warning: Secret word is not rank 1! This suggests an issue with the similarity calculation.")
        
        # Show sample top neighbors for diagnostic
        print("\n📊 Top 50 semantic neighbors:")
        top_50 = [(word, data['similarity']) for word, data in self.rankings.items() if data['rank'] <= 50]
        top_50.sort(key=lambda x: self.rankings[x[0]]['rank'])
        
        for word, similarity in top_50:
            rank = self.rankings[word]['rank']
            print(f"  {rank:2d}. {word:15s} (similarity: {similarity:.4f})")
        
        # Diagnostic: Compare semantic vs orthographic similarity
        print("\n🔍 Diagnostic: Semantic vs Orthographic similarity")
        print("Checking if rankings are biased by spelling similarity...")
        
        # Sample some high-ranking words
        sample_words = [word for word, data in self.rankings.items() if 2 <= data['rank'] <= 20]
        
        print(f"\nComparing top semantic matches to orthographic similarity:")
        for word in sample_words[:10]:
            semantic_sim = self.rankings[word]['similarity']
            ortho_sim = SequenceMatcher(None, word, self.secret_word).ratio()
            rank = self.rankings[word]['rank']
            print(f"  {rank:2d}. {word:15s} | Semantic: {semantic_sim:.4f} | Orthographic: {ortho_sim:.4f}")
        
        return True

    def _get_ai_clues_batch(self, words_batch, secret_word):
        """Get AI-generated clues for a batch of words."""
        if not self.use_ai:
            return {}

        prompt = (
            f"For each word below, describe the SPECIFIC relationship between the word and '{secret_word}' in 7 words or less. "
            f"Focus on HOW they connect, not what the word is. Don't mention '{secret_word}' directly - use 'that animal/creature/thing' instead. "
            f"Provide the output as a JSON object where keys are the words and values are the relationship descriptions. "
            f"Example: {{'collar': 'restrains and guides that animal', 'bone': 'that animal buries and chews it'}}\n\n"
            f"Words: {json.dumps(words_batch)}"
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides short, creative clues for a word game."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=len(words_batch) * 10,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            clues_json = json.loads(response.choices[0].message.content)
            
            # Validate and truncate clues to max 5 words
            for word, clue in clues_json.items():
                if isinstance(clue, str):
                    clues_json[word] = " ".join(clue.split()[:5])
                else:
                    clues_json[word] = "ERROR"
            
            return clues_json
        except Exception as e:
            print(f"⚠️  AI clue generation failed: {e}")
            return {}

    def generate_csv(self, output_filename=None):
        """Generate the complete CSV file with semantic rankings and AI clues."""
        if output_filename is None:
            if not os.path.exists("secretword"):
                os.makedirs("secretword")
            output_filename = f"secretword/secretword-{self.secret_word}.csv"
        
        print(f"Generating CSV file: {output_filename}")
        
        if not self.use_ai:
            print("⚠️  AI not available - generating CSV with NULL clues only")
            return self.generate_csv_no_ai(output_filename)
        
        # Get words that need AI clues (ranks 1 to ai_cutoff_rank)
        ai_words = [(word, self.rankings[word]['rank']) for word in self.words 
                   if self.rankings[word]['rank'] <= self.ai_cutoff_rank]
        ai_words.sort(key=lambda x: x[1])  # Sort by rank

        all_clues = {}
        successful_ai_clues = 0
        failed_ai_clues = 0
        
        print(f"🤖 Generating AI clues for {len(ai_words):,} words (ranks 1-{self.ai_cutoff_rank:,})")
        estimated_api_calls = math.ceil(len(ai_words) / self.batch_size)
        print(f"📞 Estimated API calls: ~{estimated_api_calls:,}")

        # Process in batches
        for i in tqdm(range(0, len(ai_words), self.batch_size), desc="Generating AI clues"):
            batch = [word for word, _ in ai_words[i:i + self.batch_size]]
            
            # Special handling for secret word
            if any(self.rankings[word]['rank'] == 1 for word in batch):
                for word in batch:
                    if self.rankings[word]['rank'] == 1:
                        all_clues[word] = "This is the *."
                        successful_ai_clues += 1
                        batch.remove(word)
                        break
            
            if batch:  # If there are still words to process
                batch_clues = self._get_ai_clues_batch(batch, self.secret_word)
                
                for word in batch:
                    if word in batch_clues:
                        all_clues[word] = batch_clues[word]
                        successful_ai_clues += 1
                    else:
                        all_clues[word] = "ERROR"
                        failed_ai_clues += 1
        
        print(f"✅ AI clue generation complete!")
        print(f"   Successful AI clues: {successful_ai_clues:,}")
        print(f"   Failed AI clues: {failed_ai_clues:,}")

        # Add NULL for words beyond the cutoff (saves space)
        null_clue_count = 0
        for word in self.words:
            rank = self.rankings[word]['rank']
            if rank > self.ai_cutoff_rank:
                all_clues[word] = None  # NULL value
                null_clue_count += 1
        
        print(f"✅ Set {null_clue_count:,} distant words to NULL (space optimization)")
        
        # Create CSV data
        csv_data = []
        for word in self.words:
            rank_info = self.rankings.get(word, {'rank': 0})
            clue = all_clues.get(word, f"No clue generated for {word}")
            csv_data.append({
                'rank': rank_info['rank'],
                'secret_word': self.secret_word,
                'word': word,
                'clue': clue
            })

        df = pd.DataFrame(csv_data)
        df.sort_values(by='rank', inplace=True)
        
        df.to_csv(output_filename, index=False)
        print(f"CSV file generated successfully: {output_filename}")
        print(f"Total rows: {len(df):,}")

        print("\nSample rows from the CSV:")
        print(f"Top 15 (AI-generated clues):")
        print(df.head(15).to_string(index=False))
        
        distant_sample = df[df['rank'] > self.ai_cutoff_rank].head(5)
        if not distant_sample.empty:
            print("\nSample distant words (static clues):")
            print(distant_sample.to_string(index=False))
        
        return output_filename

    def generate_csv_no_ai(self, output_filename):
        """Generate CSV file with rankings but no AI clues (all NULL)."""
        print(f"Generating CSV without AI clues: {output_filename}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        
        # Create CSV data with NULL clues
        csv_data = []
        for word in self.words:
            rank_info = self.rankings.get(word, {'rank': 0})
            
            # Special case for the secret word itself
            if rank_info['rank'] == 1:
                clue = "This is the *."
            else:
                clue = None  # NULL for all other words
            
            csv_data.append({
                'rank': rank_info['rank'],
                'secret_word': self.secret_word,
                'word': word,
                'clue': clue
            })

        # Create DataFrame and sort by rank
        df = pd.DataFrame(csv_data)
        df.sort_values(by='rank', inplace=True)
        
        # Write to CSV
        df.to_csv(output_filename, index=False)
        
        # Report results
        file_size = os.path.getsize(output_filename)
        print(f"✅ CSV generated successfully: {output_filename}")
        print(f"📏 File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        print(f"📊 Total rows: {len(df):,}")
        print(f"🎯 Secret word '{self.secret_word}' at rank 1")
        print(f"💾 All clues set to NULL (except secret word) for space optimization")
        
        return output_filename

    def generate_csv_with_progress(self, progress_callback):
        """Generate CSV with progress reporting callback."""
        self.progress_callback = progress_callback
        return self.generate_csv()
    
    def run_full_generation(self, output_filename=None):
        """Run the complete generation process."""
        print("=== Semantic Embedding Rank CSV Generator ===")
        print(f"Secret word: {self.secret_word}")
        print(f"Using OpenAI text-embedding-3-large for semantic similarity")
        print(f"AI clue cutoff: rank {self.ai_cutoff_rank:,}")
        print(f"Batch size: {self.batch_size}")

        if not self.use_ai:
            print("\n❌ AI is not enabled. Please set OPENAI_API_KEY.")
            return False

        # Load words
        if not self.load_words():
            return False

        # Compute semantic rankings
        if not self.compute_semantic_rankings():
            return False
        
        # Generate CSV
        generated_file = self.generate_csv(output_filename)
        
        if generated_file:
            print("\n=== Generation Complete! ===")
            print(f"\n🎉 Success! Generated {generated_file}")
            print("The CSV file is ready to use in your Semantic Rank game!")
            print(f"\nFile contains semantic rankings for all {len(self.words):,} ENABLE words")
            print("Columns: rank, secret_word, word, clue")
            print(f"✨ AI-generated clues for ranks 1-{self.ai_cutoff_rank:,}")
            print(f"📝 Static message for ranks {self.ai_cutoff_rank+1:,}+")
            return True
        else:
            print("\n❌ Generation failed!")
            return False

def main():
    """Main function to run the generator."""
    secret_word = "planet"
    batch_size = 50
    
    generator = SemanticEmbeddingGenerator(secret_word, batch_size)
    
    success = generator.run_full_generation()
    
    if success:
        output_file = f"secretword/secretword-{secret_word}.csv"
        print(f"\n🎉 Success! Generated {output_file}")
    else:
        print("❌ Generation failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
