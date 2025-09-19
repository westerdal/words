#!/usr/bin/env python3
"""
Test script to verify cached embeddings functionality
"""

import os
import sys
from semantic_embedding_generator import SemanticEmbeddingGenerator

def test_cached_embeddings():
    """Test the cached embeddings functionality."""
    
    print("=== Testing Cached Embeddings ===")
    
    # Check if embeddings cache exists
    cache_path = ".env/embeddings.json"
    if not os.path.exists(cache_path):
        print(f"❌ Embeddings cache not found at {cache_path}")
        print("Please ensure the embeddings.json file is in the .env directory")
        return False
    
    # Test with a simple secret word
    secret_word = "forest"
    print(f"Testing with secret word: '{secret_word}'")
    
    # Create generator (no API key needed for cached version)
    generator = SemanticEmbeddingGenerator(secret_word, batch_size=50)
    
    # Load words
    print("Loading word list...")
    if not generator.load_words():
        print("❌ Failed to load words")
        return False
    
    # Test cached embeddings
    print("Testing cached embeddings computation...")
    success = generator.compute_semantic_rankings_from_cache(cache_path)
    
    if success:
        print("✅ Cached embeddings test successful!")
        
        # Show some results
        if generator.rankings:
            secret_rank = generator.rankings[secret_word]['rank']
            print(f"Secret word '{secret_word}' has rank: {secret_rank}")
            
            # Show top 20 words
            top_words = sorted(generator.rankings.items(), key=lambda x: x[1]['rank'])[:20]
            print("Top 20 most similar words:")
            for word, data in top_words:
                print(f"  {data['rank']:3d}. {word} (similarity: {data['similarity']:.4f})")
        
        return True
    else:
        print("❌ Cached embeddings test failed")
        return False

def main():
    """Main function."""
    success = test_cached_embeddings()
    
    if success:
        print("\n🎉 All tests passed! Cached embeddings are working correctly.")
        print("The system is ready to generate semantic rankings using cached embeddings.")
    else:
        print("\n❌ Tests failed. Please check the error messages above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
