#!/usr/bin/env python3
"""
Demo script showing AI-generated clues vs pattern-based clues
"""

from ai_semantic_rank import AISemanticRankGenerator
import random

def demo_clue_comparison():
    """Demonstrate the difference between AI and pattern-based clues."""
    
    secret_word = "planet"
    test_words = [
        ("earth", 2), ("mars", 3), ("sun", 19), ("moon", 14), ("orbit", 15),
        ("telescope", 500), ("gravity", 800), ("science", 1200), 
        ("book", 5000), ("car", 15000), ("spoon", 80000), ("happiness", 120000)
    ]
    
    print("=== AI vs Pattern-Based Clue Comparison ===")
    print(f"Secret word: {secret_word}")
    print()
    
    # Initialize generators
    ai_gen = AISemanticRankGenerator(secret_word, use_ai=True)
    pattern_gen = AISemanticRankGenerator(secret_word, use_ai=False)
    
    print(f"{'Word':<12} {'Rank':<6} {'AI Clue (if available)':<25} {'Pattern Clue':<25}")
    print("-" * 75)
    
    for word, rank in test_words:
        ai_clue = ai_gen.generate_ai_clue(word, rank)
        pattern_clue = pattern_gen.generate_fallback_clue(word, rank)
        
        print(f"{word:<12} {rank:<6} {ai_clue:<25} {pattern_clue:<25}")
    
    print()
    print("Note: AI clues would be unique and contextually aware if OpenAI API key is provided.")
    print("Pattern clues use templates but are still functional for the game.")

if __name__ == "__main__":
    demo_clue_comparison()
