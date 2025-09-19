#!/usr/bin/env python3
"""
Test the secret word validation in clue generation
"""

import sys
from pathlib import Path

# Add utilities to path
sys.path.append(str(Path(__file__).parent / "scripts" / "utilities"))
sys.path.append(str(Path(__file__).parent / "scripts" / "processing"))

from generate_csv import CSVGenerator

def test_clue_validation():
    """Test that clues containing the secret word are rejected"""
    
    # Create generator for "art"
    generator = CSVGenerator("art")
    
    # Test with words that might generate clues containing "art"
    test_words = ["place", "gallery", "museum", "canvas", "painting"]
    
    print(f"🧪 Testing clue validation for secret word: 'art'")
    print(f"📝 Test words: {test_words}")
    print()
    
    # Generate clues
    try:
        results = generator.get_ai_clues_batch(test_words)
        
        print("🔍 Results:")
        for word, data in results.items():
            clue = data['clue']
            strength = data['strength']
            
            # Check if clue contains "art"
            contains_secret = 'art' in clue.lower()
            status = "❌ CONTAINS SECRET WORD!" if contains_secret else "✅ Safe"
            
            print(f"  {word}: '{clue}' ({strength}) {status}")
            
    except Exception as e:
        print(f"💥 Error: {e}")

if __name__ == "__main__":
    test_clue_validation()
